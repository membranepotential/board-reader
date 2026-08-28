"""The >=3-export-dir gate and empty-cap on the crop fine-tune
(``tools/train/finetune_crops.py``).

Same shape as ``test_calibrate_finetune.py``'s session-gate tests: this is the
product rule (too few *sessions* — here, export directories — teaches one
sitting's conditions instead of the cross-session type distinction), so it's the
part standing between an operator and a fine-tune that silently costs accuracy,
and it's stdlib-only by design so it runs in the torch-free workspace suite
without dragging torch into it. Everything past label-loading (the LP-FT itself)
needs the scratch torch venv and is verified by running the tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import finetune_crops as fc

GEOMETRY = {"kind": "bbox", "headroom": 1.6, "side_pad": 0.15}


def write_session(
    dir_: Path,
    pieces: list[str],
    *,
    geometry: dict | None = None,
    skip_export_json: bool = False,
) -> Path:
    """One export dir on disk: labels_normalized.json + crops/ + export.json,
    exactly the shape `eval-session --export-crops` writes."""
    dir_.mkdir(parents=True, exist_ok=True)
    crops = dir_ / "crops"
    crops.mkdir(exist_ok=True)
    records = []
    for i, piece in enumerate(pieces):
        name = f"{i:03d}_{piece}.jpg"
        (crops / name).write_bytes(b"not really a jpeg")
        records.append({"image": f"crops/{name}", "piece": piece, "chess_set": "phone"})
    (dir_ / fc.MANIFEST_NAME).write_text(json.dumps(records))
    if not skip_export_json:
        (dir_ / fc.EXPORT_META_NAME).write_text(
            json.dumps({"geometry": geometry or GEOMETRY})
        )
    return dir_


# --- directory discovery -----------------------------------------------------------


def test_no_dirs_given_is_refused():
    with pytest.raises(fc.Refused, match="pass --export-dirs"):
        fc.resolve_export_dirs([])


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(fc.Refused, match="not a directory"):
        fc.resolve_export_dirs([tmp_path / "nope"])


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    with pytest.raises(fc.Refused, match=fc.MANIFEST_NAME):
        fc.resolve_export_dirs([tmp_path / "empty_dir"])


def test_duplicate_dirs_are_deduplicated(tmp_path):
    write_session(tmp_path / "a", ["P"])
    resolved = fc.resolve_export_dirs([tmp_path / "a", tmp_path / "a"])
    assert len(resolved) == 1


# --- session loading -----------------------------------------------------------


def test_a_session_loads_its_crops_and_class_counts(tmp_path):
    write_session(tmp_path / "s1", ["P", "P", "n", "empty", "empty", "empty"])
    session = fc.load_session(tmp_path / "s1")
    assert session.name == "s1"
    assert len(session.records) == 6
    assert session.class_counts == {"P": 2, "n": 1, "empty": 3}
    assert session.geometry == GEOMETRY


def test_an_unknown_piece_code_refuses(tmp_path):
    write_session(tmp_path / "s1", ["P"])
    manifest = tmp_path / "s1" / fc.MANIFEST_NAME
    manifest.write_text(json.dumps([{"image": "crops/000_P.jpg", "piece": "X"}]))
    with pytest.raises(fc.Refused, match="unknown piece code"):
        fc.load_session(tmp_path / "s1")


def test_a_missing_crop_image_refuses(tmp_path):
    write_session(tmp_path / "s1", ["P"])
    (tmp_path / "s1" / "crops" / "000_P.jpg").unlink()
    with pytest.raises(fc.Refused, match="crop image missing"):
        fc.load_session(tmp_path / "s1")


def test_an_empty_manifest_refuses(tmp_path):
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / fc.MANIFEST_NAME).write_text("[]")
    with pytest.raises(fc.Refused, match="no crop records"):
        fc.load_session(tmp_path / "s1")


def test_a_missing_export_json_refuses(tmp_path):
    write_session(tmp_path / "s1", ["P"], skip_export_json=True)
    with pytest.raises(fc.Refused, match=fc.EXPORT_META_NAME):
        fc.load_session(tmp_path / "s1")


# --- the session gate -----------------------------------------------------------


def _sessions(tmp_path, n: int) -> list[fc.ExportedSession]:
    return [
        fc.load_session(write_session(tmp_path / f"s{i}", ["P", "empty"]))
        for i in range(n)
    ]


def test_three_export_dirs_pass_the_gate(tmp_path):
    fc.check_session_gate(_sessions(tmp_path, 3), force=False)  # does not raise


def test_fewer_than_three_is_refused_with_the_measurement_quoted(tmp_path):
    with pytest.raises(fc.Refused) as e:
        fc.check_session_gate(_sessions(tmp_path, 2), force=False)
    message = str(e.value)
    assert "2 export dir(s)" in message
    assert "0.9323" in message and "0.9681" in message  # calibrate_finetune's measurement, quoted


def test_force_overrides_the_gate_loudly(tmp_path, capsys):
    fc.check_session_gate(_sessions(tmp_path, 1), force=True)
    warning = capsys.readouterr().err
    assert "--force" in warning and "OUTSIDE calibrate_finetune's measured" in warning


def test_force_on_a_passing_gate_says_nothing(tmp_path, capsys):
    fc.check_session_gate(_sessions(tmp_path, 3), force=True)
    assert capsys.readouterr().err == ""


# --- empty capping -----------------------------------------------------------


def _records(*labels: str) -> list[fc.CropRecord]:
    return [fc.CropRecord(Path(f"{i}.jpg"), label, "s") for i, label in enumerate(labels)]


def test_cap_empties_caps_to_the_multiple_of_the_largest_nonempty_class():
    records = _records(*(["P"] * 10 + ["N"] * 5 + ["empty"] * 50))
    capped = fc.cap_empties(records, multiple=2.0, seed=0)
    counts = {}
    for r in capped:
        counts[r.label] = counts.get(r.label, 0) + 1
    assert counts["P"] == 10 and counts["N"] == 5
    assert counts["empty"] == 20  # 2x the largest non-empty class (P, 10)


def test_cap_empties_is_a_noop_when_already_under_the_cap():
    records = _records(*(["P"] * 10 + ["empty"] * 15))
    capped = fc.cap_empties(records, multiple=2.0, seed=0)
    assert len(capped) == len(records)


def test_cap_empties_multiple_zero_disables_capping():
    records = _records(*(["P"] * 1 + ["empty"] * 100))
    capped = fc.cap_empties(records, multiple=0.0, seed=0)
    assert len(capped) == len(records)


def test_cap_empties_is_deterministic_given_a_seed():
    records = _records(*(["P"] * 10 + ["empty"] * 50))
    a = fc.cap_empties(records, multiple=2.0, seed=7)
    b = fc.cap_empties(records, multiple=2.0, seed=7)
    assert [r.image_path for r in a] == [r.image_path for r in b]


def test_cap_empties_with_no_nonempty_crops_is_a_noop():
    records = _records(*(["empty"] * 10))
    assert fc.cap_empties(records, multiple=2.0, seed=0) == records


# --- geometry matching -----------------------------------------------------------


def test_geometry_matches_identical_bbox_geometry():
    assert fc._geometry_matches(GEOMETRY, dict(GEOMETRY))


def test_geometry_matches_tolerates_float_noise():
    a = {"kind": "bbox", "headroom": 1.6, "side_pad": 0.15}
    b = {"kind": "bbox", "headroom": 1.6 + 1e-9, "side_pad": 0.15}
    assert fc._geometry_matches(a, b)


def test_geometry_mismatch_on_headroom_is_caught():
    a = {"kind": "bbox", "headroom": 1.6, "side_pad": 0.15}
    b = {"kind": "bbox", "headroom": 0.15, "side_pad": 0.15}
    assert not fc._geometry_matches(a, b)


def test_geometry_mismatch_on_kind_is_caught():
    a = {"kind": "bbox", "headroom": 1.6, "side_pad": 0.15}
    b = {"kind": "pose", "piece_height": 1.9, "piece_radius": 0.5}
    assert not fc._geometry_matches(a, b)
