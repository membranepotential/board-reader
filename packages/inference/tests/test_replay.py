"""Tests for the replay harness (:mod:`board_reader_inference.replay`).

The load-bearing claims, in order: a replay is driven by the *recorded* clock and
not the wall one (so a 2.5 s settle window is 2.5 recorded seconds however fast
the GPU runs); the same recording through the same code writes byte-identical
events; the output is a recording directory ``eval-session`` scores unchanged and
the source recording is not written to; a multi-session directory replays one
session rather than splicing two; and the frame-selection policy is genuinely
asked, because the next milestone (move-triggered estimation) plugs in there.

No GPU and no model: the estimator is a scripted fake, which is also the only way
to assert on settle timing — a real model's answers are the thing under test
everywhere else, but here the *timing* is.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from board_reader_core.protocol import (
    START_PLACEMENT,
    ClockState,
    PositionEvent,
    dist_from_placement,
    iso_utc,
)
from board_reader_core.recording import EVENTS_NAME, INDEX_NAME, FrameRecorder, Recording
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference.cli import main
from board_reader_inference.replay import (
    EveryFrame,
    ReplayResult,
    build_session,
    default_name,
    find_sessions,
    replay,
)
from board_reader_inference.session import EstimatorSession
from board_reader_inference.stabilize import SETTLE_SECONDS
from board_reader_inference.trial import REPORT_NAME, TRUTH_NAME

E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"

#: Recorded seconds between frames. Five to a settle window, so the settle can be
#: counted in reads without the test knowing the window's value.
FRAME_DT = 0.5


# --------------------------------------------------------------------------- #
# Fixtures: a tiny recording and a scripted estimator
# --------------------------------------------------------------------------- #


class ScriptedEstimator:
    """Reads a placement per frame ``seq`` off a script, at no cost.

    Deliberately not photo-mode and given corner-less frames, so
    :func:`~board_reader_inference.routing.route_for` picks the whole-frame branch
    and the test exercises the live routing rather than bypassing it.
    """

    def __init__(self, script: dict[int, str], default: str = START_PLACEMENT) -> None:
        self.script = script
        self.default = default
        self.seen: list[int] = []

    def estimate(self, frame: Frame) -> PositionEvent:
        self.seen.append(frame.seq)
        placement = self.script.get(frame.seq, self.default)
        return PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=dist_from_placement(placement, 0.9),
            stability=frame.stability,
            clock=ClockState(frame.clock),
        )


def _write_recording(
    root, *, frames: int = 8, first_ts: float = 1_700_000_000.0, seq_start: int = 0,
    session_id: str = "src", board_id: str = "phone",
) -> None:
    """Append ``frames`` synthetic frames to a recording session on disk."""
    rng = np.random.default_rng(seq_start)
    recorder = FrameRecorder(root, session_id=session_id, writer="test")
    for i in range(frames):
        image = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
        frame = Frame(
            board_id=board_id,
            seq=seq_start + i,
            capture_ts=first_ts + i * FRAME_DT,
            image=image,
            burst_id=f"{board_id}:b{i // 4}",
        )
        recorder.record(frame.header(), encode_jpeg(image))
    recorder.stop()


@pytest.fixture
def source(tmp_path):
    """An 8-frame recording of one board, 0.5 recorded seconds apart."""
    _write_recording(tmp_path / "recordings")
    return Recording(tmp_path / "recordings" / "src")


def _run(source: Recording, name: str, estimator=None, **kwargs) -> ReplayResult:
    return replay(
        source,
        source.path / "replays" / name,
        session=EstimatorSession(estimator or ScriptedEstimator({})),
        log=None,
        **kwargs,
    )


def _events(result: ReplayResult) -> list[dict]:
    text = (result.path / EVENTS_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# The virtual clock
# --------------------------------------------------------------------------- #


def test_settles_on_recorded_time_not_wall_time(source):
    """The debounce window is 2.5 RECORDED seconds, even in a millisecond replay.

    The whole reason a replay is worth running: the stabilizer is fed each frame's
    ``capture_ts``, so a settle that takes 2.5 s live takes 2.5 s of recording
    here — five frames at 0.5 s apart — while the replay itself finishes in
    milliseconds. A wall-clock stabilizer would settle on the very next read.
    """
    # Every frame from seq 1 on reads e4; seq 0 is the board the stream adopts.
    estimator = ScriptedEstimator(dict.fromkeys(range(1, 8), E4))
    started = time.monotonic()
    result = _run(source, "clock", estimator)
    elapsed = time.monotonic() - started

    settled = [e["settled"]["placement"] for e in _events(result)]
    assert settled[0] == START_PLACEMENT
    # seq 1 is the first e4 read (t0 + 0.5s); the window closes at t0 + 3.0s,
    # which is seq 6 — SETTLE_SECONDS after the challenger first appeared.
    first_e4 = 1
    window_frames = int(SETTLE_SECONDS / FRAME_DT)
    assert settled[first_e4 + window_frames - 1] == START_PLACEMENT
    assert settled[first_e4 + window_frames] == E4
    # And it did not take 2.5 s of anybody's wall clock to get there.
    assert elapsed < SETTLE_SECONDS


def test_events_carry_no_published_ts(source):
    """A replay never published anything, so it stamps no publish latency.

    Faking one (with the replay's own wall clock, or with the capture time) would
    put a fabricated capture->publish number in the trial report's timing block.
    """
    result = _run(source, "nostamp")
    assert all("published_ts" not in e for e in _events(result))
    assert all(e["capture_ts"] > 0 for e in _events(result))


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_two_replays_of_one_recording_are_byte_identical(source):
    """Same recording + same code + same estimator -> the same bytes.

    Without this a replay cannot be diffed against another one, which is the only
    thing the harness is for.
    """
    script = dict.fromkeys(range(3, 8), E4)
    a = _run(source, "a", ScriptedEstimator(script))
    b = _run(source, "b", ScriptedEstimator(script))
    assert (a.path / EVENTS_NAME).read_bytes() == (b.path / EVENTS_NAME).read_bytes()
    assert (a.path / INDEX_NAME).read_bytes() == (b.path / INDEX_NAME).read_bytes()


def test_default_name_tracks_the_model_environment(monkeypatch):
    """The default replay name separates model configurations, not runs.

    Re-running the same code against the same models must land in the same
    directory (it supersedes its own answer); pointing BR_MODEL somewhere else
    must land beside it, or one score silently overwrites another's.
    """
    monkeypatch.setenv("BR_MODEL", "models/run_e/model.onnx")
    first = default_name()
    assert first == default_name()
    monkeypatch.setenv("BR_MODEL", "models/run_g/model.onnx")
    assert default_name() != first


# --------------------------------------------------------------------------- #
# The output directory
# --------------------------------------------------------------------------- #


def test_replay_directory_is_scoreable_and_leaves_the_source_alone(source, tmp_path):
    """``eval-session <replay dir>`` works unchanged, and nothing wrote to the source.

    The layout claim end to end: truth and pixels are symlinked (never copied),
    the thin ``frames.ndjson`` keeps the timing block honest, and the only thing
    that appears in the recording is the ``replays/`` directory itself.
    """
    truth = {
        "format": 1,
        "board_id": "phone",
        "segments": [{"from_seq": 0, "placement": START_PLACEMENT}],
        "guard_seqs": 0,
    }
    (source.path / TRUTH_NAME).write_text(json.dumps(truth), encoding="utf-8")
    before = {
        p.name: p.read_bytes()
        for p in source.path.iterdir()
        if p.is_file()
    }

    result = _run(source, "scoreable")

    assert (result.path / "jpeg").is_symlink()
    assert (result.path / TRUTH_NAME).is_symlink()
    assert (result.path / "jpeg" / "0000" / "000000.jpg").exists()
    assert len(_events(result)) == 8
    assert (result.path / INDEX_NAME).read_text().count("\n") == 8

    assert main(["eval-session", str(result.path)]) == 0
    assert (result.path / REPORT_NAME).exists()
    report = json.loads((result.path / REPORT_NAME).read_text())
    assert report["events"]["scored"] == 8
    # The fake reads the start position on every frame, so the score is perfect —
    # what is under test is that scoring ran at all, on the replay's own events.
    assert report["streams"][0]["accuracy"] == 1.0
    # ...and it read the replay's events, not the source's (which has none).
    assert not (source.path / EVENTS_NAME).exists()
    after = {p.name: p.read_bytes() for p in source.path.iterdir() if p.is_file()}
    assert after == before
    assert sorted(p.name for p in source.path.iterdir() if p.is_dir()) == [
        "jpeg",
        "replays",
    ]


def test_manifest_records_the_provenance(source, monkeypatch):
    """A replay that cannot say what produced it is not evidence of anything."""
    monkeypatch.setenv("BR_MODEL", "models/run_e/model.onnx")
    result = _run(source, "provenance")
    manifest = json.loads((result.path / "manifest.json").read_text())
    block = manifest["replay"]
    assert block["source"] == str(source.path)
    assert block["board_id"] == "phone"
    assert block["env"]["BR_MODEL"]["value"] == "models/run_e/model.onnx"
    assert block["policy"] == "every-frame"
    assert block["estimator"] == "ScriptedEstimator"
    assert block["frames"]["frames_estimated"] == 8
    assert block["settle_seconds"] == SETTLE_SECONDS
    # Recording-shaped at the top, so `Recording` opens it like any other session.
    assert Recording(result.path).manifest["writer"] == "replay"


# --------------------------------------------------------------------------- #
# Multi-session recordings
# --------------------------------------------------------------------------- #


def test_seq_restart_splits_the_recording_into_sessions(tmp_path):
    """A reconnecting edge restarts ``seq``; truth is a step function over it."""
    root = tmp_path / "recordings"
    _write_recording(root, frames=3, session_id="multi")
    _write_recording(root, frames=6, session_id="multi", first_ts=1_700_000_100.0)
    recording = Recording(root / "multi")

    sessions = find_sessions(recording.frame_index())
    assert [s.frames for s in sessions] == [3, 6]

    result = _run(recording, "longest")
    assert result.replayed == [1]
    assert result.stats.frames_total == 6
    # The longest session's frames only: splicing the two would score the second
    # session's reads against the first session's truth segments.
    assert [e["frame_seq"] for e in _events(result)] == list(range(6))


def test_all_sessions_replays_everything(tmp_path):
    root = tmp_path / "recordings"
    _write_recording(root, frames=3, session_id="multi")
    _write_recording(root, frames=6, session_id="multi", first_ts=1_700_000_100.0)
    recording = Recording(root / "multi")

    result = _run(recording, "all", all_sessions=True)
    assert result.replayed == [0, 1]
    assert len(_events(result)) == 9


# --------------------------------------------------------------------------- #
# The frame-selection seam
# --------------------------------------------------------------------------- #


class EvenSeqs:
    """A stand-in for the move-triggered scheduler: half the frames, chosen by header."""

    name = "even-seqs"

    def __init__(self) -> None:
        self.asked: list[int] = []

    def should_estimate(self, entry) -> bool:
        self.asked.append(entry.seq)
        return entry.seq % 2 == 0


def test_the_policy_decides_which_frames_are_estimated(source):
    """Every frame is offered to the policy; only the ones it wants cost a read.

    The seam the move-triggered milestone plugs into. A skipped frame still
    appears in the thin ``frames.ndjson`` — the input rate is a fact about the
    recording, and a timing block that hid the frames the policy declined would
    make a sparse policy look like a slow camera.
    """
    estimator = ScriptedEstimator({})
    policy = EvenSeqs()
    result = _run(source, "policy", estimator, policy=policy)

    assert policy.asked == list(range(8))  # asked once each, in order
    assert estimator.seen == [0, 2, 4, 6]  # and the skipped frames cost nothing
    assert [e["frame_seq"] for e in _events(result)] == [0, 2, 4, 6]
    assert result.stats.frames_skipped_policy == 4
    assert result.path.joinpath(INDEX_NAME).read_text().count("\n") == 8


def test_limit_stops_reading_without_hiding_the_frames(source):
    """``--limit`` is a smoke-run brake, and says so separately from the policy.

    The frames past it are still real input, so they stay in the thin index and
    are counted apart — a truncated replay must not read like a sparse one.
    """
    estimator = ScriptedEstimator({})
    result = _run(source, "limited", estimator, limit=3)
    assert estimator.seen == [0, 1, 2]
    assert result.stats.frames_skipped_limit == 5
    assert result.stats.frames_skipped_policy == 0
    assert result.path.joinpath(INDEX_NAME).read_text().count("\n") == 8


def test_every_frame_is_the_default_policy(source):
    estimator = ScriptedEstimator({})
    _run(source, "default", estimator)
    assert estimator.seen == list(range(8))
    assert EveryFrame().name == "every-frame"


# --------------------------------------------------------------------------- #
# The CLI, end to end on the stub estimator
# --------------------------------------------------------------------------- #


def test_cli_replay_and_score(tmp_path, monkeypatch, capsys):
    """``replay --score`` on the walking-skeleton estimator: no model, no GPU.

    Also pins the exit codes, since this is a benchmarking command that will end
    up in scripts: 0 scored, 2 for a recording with no truth to score against.
    """
    for name in (
        "BR_MODEL", "BR_MODEL_B", "BR_MODEL_TUNED", "BR_CENTROIDS",
        "BR_SESSION_BANK", "BR_OCCUPANCY_BANK",
    ):
        monkeypatch.delenv(name, raising=False)
    # Cwd-relative writes have cost real user state before (BR_SESSION_BANK);
    # nothing here should write outside tmp_path, and this makes that true.
    monkeypatch.chdir(tmp_path)
    _write_recording(tmp_path / "recordings", frames=4)
    recording = tmp_path / "recordings" / "src"

    assert main(["replay", str(recording), "--name", "cli"]) == 0
    out = capsys.readouterr().out
    assert "4 estimated" in out

    replayed = recording / "replays" / "cli"
    # The environment names no model, so the stack the harness builds is the
    # walking skeleton — the same one `serve` would run with this environment.
    assert type(build_session().estimator).__name__ == "StubEstimator"
    assert (replayed / EVENTS_NAME).read_text().count("\n") == 4

    # No truth in the source: --score says so and exits 2 rather than pretending.
    assert main(["replay", str(recording), "--name", "cli", "--score"]) == 2

    (recording / TRUTH_NAME).write_text(
        json.dumps(
            {
                "format": 1,
                "board_id": "phone",
                "segments": [{"from_seq": 0, "placement": START_PLACEMENT}],
                "guard_seqs": 0,
            }
        ),
        encoding="utf-8",
    )
    assert main(["replay", str(recording), "--name", "cli", "--score"]) == 0
    report = capsys.readouterr().out
    assert "stream       per-square" in report
    assert (replayed / REPORT_NAME).exists()
