"""The ≥3-session gate on the calibration fine-tune (``tools/train/calibrate_finetune.py``).

This is the product rule the experiment bought: frames from one sitting made the
model slightly *worse* (0.932 vs the base's 0.938), frames from three sittings made
it clearly better (0.968). The gate is the only thing standing between an operator
and a fine-tune that silently costs them accuracy, so it gets tested.

It is also the reason this file — alone in ``tools/train/tests/`` — is listed in the
workspace ``testpaths``: the frame-loading and clustering layer of that tool is
deliberately stdlib-only, so the rule is checkable by ``make test`` without dragging
torch into the workspace venv. Everything below it (the LP-FT itself) needs the
scratch torch venv and is verified by running the tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibrate_finetune as cf

HOUR = 3600.0
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
CORNERS = "138,550,812,520,1051,1136,57,1260"


def write_frame(dir_: Path, name: str, ts: float, overrides: dict | None = None) -> Path:
    """A calibrate frame on disk: sidecar + the jpg it must sit beside."""
    meta = {"kind": "calibrate", "ts": ts, "corners": CORNERS, "placement": START}
    meta.update(overrides or {})
    sidecar = dir_ / f"{name}-calibrate.json"
    sidecar.write_text(json.dumps(meta))
    (dir_ / f"{name}-calibrate.jpg").write_bytes(b"not really a jpeg")
    return sidecar


def frames_at(*offsets_s: float) -> list[cf.CalibFrame]:
    base = 1_784_323_624.0
    return [
        cf.CalibFrame(f"f{i}", Path(f"f{i}.jpg"), Path(f"f{i}.json"), base + off, ())
        for i, off in enumerate(offsets_s)
    ]


def test_frames_minutes_apart_are_one_session():
    """The measured sittings had shots two and four minutes apart. Splitting those
    into separate "sessions" would let one sitting pass a three-session gate."""
    sessions = cf.group_sessions(frames_at(0, 62, 238))
    assert [len(s) for s in sessions] == [3]


def test_a_gap_over_an_hour_starts_a_new_session():
    sessions = cf.group_sessions(frames_at(0, 60, 2 * HOUR, 2 * HOUR + 120))
    assert [len(s) for s in sessions] == [2, 2]


def test_the_boundary_is_the_gap_not_the_elapsed_total():
    """Sessions are cut at gaps *between consecutive frames*: a long sitting of
    closely-spaced shots stays one session however long it ran in total."""
    sessions = cf.group_sessions(frames_at(*[i * 0.9 * HOUR for i in range(5)]))
    assert [len(s) for s in sessions] == [5]
    assert sessions[0][-1].ts - sessions[0][0].ts > 3 * HOUR


def test_exactly_an_hour_does_not_split():
    assert len(cf.group_sessions(frames_at(0, HOUR))) == 1
    assert len(cf.group_sessions(frames_at(0, HOUR + 0.001))) == 2


def test_frames_are_ordered_by_capture_time_before_grouping():
    """Explicit --frames arrive in whatever order the shell globbed them."""
    sessions = cf.group_sessions(frames_at(2 * HOUR, 0, 4 * HOUR, 60))
    assert [len(s) for s in sessions] == [2, 1, 1]


def test_the_measured_shape_passes_the_gate():
    """C2′: one frame from each of three sessions, days apart."""
    sessions = cf.group_sessions(frames_at(0, 15 * HOUR, 480 * HOUR))
    assert len(sessions) == 3
    cf.check_session_gate(sessions, force=False)  # does not raise


def test_one_session_is_refused_with_the_measurement_that_says_why():
    sessions = cf.group_sessions(frames_at(0, 60, 120))
    with pytest.raises(cf.Refused) as e:
        cf.check_session_gate(sessions, force=False)
    message = str(e.value)
    assert "1 capture session" in message
    assert "0.9323" in message and "0.9375" in message  # the measurement, quoted


def test_two_sessions_are_refused_too_and_the_sessions_are_named():
    sessions = cf.group_sessions(frames_at(0, 60, 30 * HOUR))
    with pytest.raises(cf.Refused) as e:
        cf.check_session_gate(sessions, force=False)
    assert "2 capture session" in str(e.value)
    assert "f0" in str(e.value) and "f2" in str(e.value)


def test_force_overrides_the_gate_loudly(capsys):
    sessions = cf.group_sessions(frames_at(0, 60))
    cf.check_session_gate(sessions, force=True)
    warning = capsys.readouterr().err
    assert "--force" in warning and "OUTSIDE every measured configuration" in warning
    assert "0.9323" in warning


def test_force_on_a_passing_gate_says_nothing(capsys):
    cf.check_session_gate(cf.group_sessions(frames_at(0, 2 * HOUR, 4 * HOUR)), force=True)
    assert capsys.readouterr().err == ""


# --- loading real sidecars ---------------------------------------------------------


def test_a_directory_of_sidecars_loads_and_clusters(tmp_path):
    base = 1_784_323_624.0
    for i, off in enumerate((0, 120, 20 * HOUR, 480 * HOUR)):
        write_frame(tmp_path, f"2026071{i}-2327", base + off)
    frames = cf.load_frames(cf.resolve_sidecars([], tmp_path, cf.DEFAULT_GLOB))
    assert [len(s) for s in cf.group_sessions(frames)] == [2, 1, 1]


def test_explicit_jpg_paths_resolve_to_their_sidecars(tmp_path):
    write_frame(tmp_path, "shot", 1_784_323_624.0)
    jpg = tmp_path / "shot-calibrate.jpg"
    frames = cf.load_frames(cf.resolve_sidecars([jpg], None, cf.DEFAULT_GLOB))
    assert [f.image_path for f in frames] == [jpg]
    assert frames[0].corners == (138, 550, 812, 520, 1051, 1136, 57, 1260)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"kind": "estimate"}, "not 'calibrate'"),
        ({"placement": "8/8/8/8/8/8/8/8"}, "not the start position"),
        ({"corners": "1,2,3"}, "8 comma-separated"),
        ({"ts": "yesterday"}, "no numeric 'ts'"),
    ],
)
def test_unusable_sidecars_refuse_the_run(tmp_path, overrides, match):
    """Every one of these would produce wrong labels or a wrong session count, and
    a fine-tune that trained on them looks fine right up until it reads a board."""
    sidecar = write_frame(tmp_path, "shot", 1_784_323_624.0, overrides)
    with pytest.raises(cf.Refused, match=match):
        cf.load_frames([sidecar])


def test_a_sidecar_without_its_image_refuses(tmp_path):
    sidecar = write_frame(tmp_path, "shot", 1_784_323_624.0)
    (tmp_path / "shot-calibrate.jpg").unlink()
    with pytest.raises(cf.Refused, match="no image beside it"):
        cf.load_frames([sidecar])


def test_frames_and_dir_together_are_refused(tmp_path):
    with pytest.raises(cf.Refused, match="not both"):
        cf.resolve_sidecars([tmp_path / "a.jpg"], tmp_path, cf.DEFAULT_GLOB)


def test_an_empty_directory_says_so(tmp_path):
    with pytest.raises(cf.Refused, match=r"no \*-calibrate\.json"):
        cf.resolve_sidecars([], tmp_path, cf.DEFAULT_GLOB)
