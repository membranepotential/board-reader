"""Tests for the replay harness (:mod:`board_reader_inference.measurement.replay`).

The load-bearing claims, in order: a replay is driven by the *recorded* clock and
not the wall one (so a 2.5 s settle window is 2.5 recorded seconds however fast
the GPU runs); the same recording through the same code writes byte-identical
events; the output is a recording directory ``eval-session`` scores unchanged and
the source recording is not written to; a multi-session directory replays one
session rather than splicing two; and the frame-selection policy is genuinely
asked, because the next milestone (move-triggered estimation) plugs in there.

No GPU and no model: the read is a scripted fake, which is also the only way
to assert on settle timing — a real model's answers are the thing under test
everywhere else, but here the *timing* is.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
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
from board_reader_inference.measurement import replay as replay_mod
from board_reader_inference.measurement.replay import (
    ReplayResult,
    check_name,
    default_name,
    find_sessions,
    render,
    replay,
)
from board_reader_inference.measurement.trial import REPORT_NAME, TRUTH_NAME
from board_reader_inference.recognition.recognizer import FrameReading
from board_reader_inference.stream.hub import PositionHub
from board_reader_inference.stream.ingest import FrameIngest, event_line
from board_reader_inference.stream.schedule import (
    ENTER_BURST,
    EXIT_BURST,
    RESTAMPED_KEY,
    BurstTriggered,
    EveryFrame,
    parse_policy,
)
from board_reader_inference.stream.stabilize import SETTLE_SECONDS, SettleConfig

E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"

#: Recorded seconds between frames. Five to a settle window, so the settle can be
#: counted in reads without the test knowing the window's value.
FRAME_DT = 0.5


# --------------------------------------------------------------------------- #
# Fixtures: a tiny recording and a scripted read
# --------------------------------------------------------------------------- #


class ScriptedRead:
    """A frame read that answers a placement per frame ``seq`` off a script, at no cost."""

    def __init__(self, script: dict[int, str], default: str = START_PLACEMENT) -> None:
        self.script = script
        self.default = default
        self.seen: list[int] = []

    def __call__(self, frame: Frame) -> FrameReading:
        self.seen.append(frame.seq)
        placement = self.script.get(frame.seq, self.default)
        event = PositionEvent.from_square_dist(
            board_id=frame.board_id,
            seq=frame.seq,
            utc=iso_utc(frame.capture_ts),
            square_dist=dist_from_placement(placement, 0.9),
            stability=frame.stability,
            clock=ClockState(frame.clock),
        )
        return FrameReading(event)


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


def _run(source: Recording, name: str, read=None, **kwargs) -> ReplayResult:
    return replay(
        source,
        source.path / "replays" / name,
        read=read or ScriptedRead({}),
        reader="ScriptedRead",
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
    read = ScriptedRead(dict.fromkeys(range(1, 8), E4))
    started = time.monotonic()
    result = _run(source, "clock", read)
    elapsed = time.monotonic() - started

    settled = [e["placement"] for e in _events(result)]
    assert settled[0] == START_PLACEMENT
    # seq 1 is the first e4 read (t0 + 0.5s); the window closes at t0 + 3.0s,
    # which is seq 6 — SETTLE_SECONDS after the challenger first appeared.
    first_e4 = 1
    window_frames = int(SETTLE_SECONDS / FRAME_DT)
    assert settled[first_e4 + window_frames - 1] == START_PLACEMENT
    assert settled[first_e4 + window_frames] == E4
    # The board changed when e4 was first read, not when the window closed: an
    # old recording replayed gets its changed_utc derived the same way.
    events = _events(result)
    assert events[first_e4 + window_frames]["changed_utc"] == events[first_e4]["utc"]
    assert events[first_e4 + window_frames - 1]["changed_utc"] == events[0]["utc"]
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
    a = _run(source, "a", ScriptedRead(script))
    b = _run(source, "b", ScriptedRead(script))
    assert (a.path / EVENTS_NAME).read_bytes() == (b.path / EVENTS_NAME).read_bytes()
    assert (a.path / INDEX_NAME).read_bytes() == (b.path / INDEX_NAME).read_bytes()


def test_default_name_tracks_the_model_environment(monkeypatch, tmp_path):
    """The default replay name separates model configurations, not runs.

    Re-running the same code against the same models must land in the same
    directory (it supersedes its own answer); pointing BR_CHESSQ somewhere else —
    or retraining the model that path names, in place — must land beside it, or
    one score silently overwrites another's.
    """
    monkeypatch.setattr(replay_mod, "git_describe", lambda: "g1a2b3c4")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"weights")
    monkeypatch.setenv("BR_CHESSQ", str(model))

    first = default_name()
    assert first.startswith("g1a2b3c4-")
    assert first == default_name()

    other = tmp_path / "other.onnx"
    other.write_bytes(b"weights")
    monkeypatch.setenv("BR_CHESSQ", str(other))
    assert default_name() != first  # a different model file

    # Retrained in place: same path, same size, new bytes. A model file is a path
    # and not a version, so the stat has to be part of the identity.
    monkeypatch.setenv("BR_CHESSQ", str(model))
    model.write_bytes(b"retrain")
    os.utime(model, ns=(1, 2))
    assert default_name() != first


def test_a_dirty_tree_has_no_default_name(monkeypatch):
    """Two sets of uncommitted changes describe identically — and would collide.

    The collision destroys the baseline a replay was meant to be diffed against,
    which is the one thing the harness exists to protect, so it is refused rather
    than resolved.
    """
    monkeypatch.setattr(replay_mod, "git_describe", lambda: "g1a2b3c4-dirty")
    with pytest.raises(ValueError, match="--name"):
        default_name()
    monkeypatch.setattr(replay_mod, "git_describe", lambda: "unversioned")
    with pytest.raises(ValueError, match="--name"):
        default_name()


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", "", "."])
def test_a_replay_name_must_be_one_path_component(name):
    """The jpeg/truth symlinks hop a fixed ../.., so the depth is a contract.

    A name with a separator in it would not merely escape the tree — it would
    produce dangling links and a replay that scores nothing, discovered ten
    minutes into a GPU run.
    """
    with pytest.raises(ValueError, match="single directory name"):
        check_name(name)


def test_a_replay_must_sit_two_levels_under_its_recording(source):
    with pytest.raises(ValueError, match="two levels"):
        replay(
            source,
            source.path / "shallow",
            read=ScriptedRead({}),
            reader="ScriptedRead",
            log=None,
        )


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
    monkeypatch.setenv("BR_CHESSQ", "models/chessq/cq.onnx")
    result = _run(source, "provenance")
    manifest = json.loads((result.path / "manifest.json").read_text())
    block = manifest["replay"]
    assert block["source"] == str(source.path)
    assert block["board_id"] == "phone"
    assert block["env"]["BR_CHESSQ"]["value"] == "models/chessq/cq.onnx"
    assert block["policy"] == "every-frame"
    assert block["estimator"] == "ScriptedRead"
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
    read = ScriptedRead({})
    policy = EvenSeqs()
    result = _run(source, "policy", read, policy=policy)

    assert policy.asked == list(range(8))  # asked once each, in order
    assert read.seen == [0, 2, 4, 6]  # and the skipped frames cost nothing
    assert [e["frame_seq"] for e in _events(result)] == [0, 2, 4, 6]
    assert result.stats.frames_skipped_policy == 4
    assert result.path.joinpath(INDEX_NAME).read_text().count("\n") == 8


def test_the_orientation_guard_drops_a_turned_read_as_the_live_worker_does(source):
    """Replay runs the same guard: a read a quarter turn off is counted, not written."""

    class Pointing(ScriptedRead):
        def __call__(self, frame: Frame) -> FrameReading:
            direction = 0.0 if frame.seq == 5 else 90.0
            return FrameReading(super().__call__(frame).event, direction)

    result = _run(source, "guarded", Pointing({}))
    assert [e["frame_seq"] for e in _events(result)] == [0, 1, 2, 3, 4, 6, 7]
    assert (result.stats.frames_estimated, result.stats.frames_misoriented) == (7, 1)
    assert "1 dropped by the orientation guard" in render(result)


def test_limit_truncates_the_take_rather_than_thinning_it(source):
    """``--limit`` stops the replay looking at the recording, and indexes nothing more.

    A truncated take is a shorter take, not a sparse reading of a long one: if the
    unread frames stayed in the index, ``eval-session`` would report a run that
    read one frame in 249 — a timing claim about a run that never happened. The
    policy is not consulted for frames nobody saw either.
    """
    read = ScriptedRead({})
    policy = EvenSeqs()
    result = _run(source, "limited", read, limit=2, policy=policy)

    assert read.seen == [0, 2]
    assert policy.asked == [0, 1, 2]  # asked up to the last frame it read, no further
    assert result.stats.truncated is True
    assert result.stats.frames_total == 3
    assert result.path.joinpath(INDEX_NAME).read_text().count("\n") == 3


def test_every_frame_is_the_default_policy(source):
    read = ScriptedRead({})
    _run(source, "default", read)
    assert read.seen == list(range(8))
    assert EveryFrame().name == "every-frame"


# --------------------------------------------------------------------------- #
# Parity with the live ingest worker
# --------------------------------------------------------------------------- #


def _ingest_event_lines(frames: list[Frame], read) -> list[dict]:
    """Drive the real :class:`FrameIngest` over ``frames`` and collect its events.

    One frame at a time, draining after each: the live inbox is latest-wins
    (maxsize=1) and would drop frames if they were pushed faster than the
    estimator ran, which is the one live behaviour a parity check must not be at
    the mercy of.
    """

    async def _drive() -> list[dict]:
        lines: list[dict] = []
        ingest = FrameIngest(
            read,
            PositionHub(),
            on_event=lambda reading, frame, rules: lines.append(
                event_line(reading.event, frame, published_ts=None, rules=rules)
            ),
        )
        for frame in frames:
            ingest.submit(frame)
            await ingest.drain()
        await ingest.close()
        return lines

    return asyncio.run(_drive())


def test_replay_publishes_exactly_what_the_live_worker_would(source):
    """The fidelity claim, pinned: same frames, same estimator, same event lines.

    Replay and :meth:`FrameIngest._run` are two hand-written orderings of the
    same two steps — route, estimate-repair-debounce —
    and the whole value of the harness is that they cannot drift. If someone adds
    a step to the live worker (another guard, a rewrite of the event before it is
    published) and not to the replay loop, this fails; without it, the replay
    quietly starts benchmarking a path nobody serves.

    Byte-equal modulo ``published_ts``, which is the gateway's wall clock at
    publish and the one thing an offline replay has no honest value for.
    """
    script = dict.fromkeys(range(3, 8), E4)
    replayed = _events(_run(source, "parity", ScriptedRead(script)))
    live = _ingest_event_lines(
        [recorded.frame() for recorded in source.frames()], ScriptedRead(script)
    )

    assert len(replayed) == 8
    assert [json.dumps(e, sort_keys=True) for e in replayed] == [
        json.dumps(e, sort_keys=True) for e in live
    ]


class _ReadAt:
    """A policy that reads exactly the frames captured at ``times``: the frames in
    between reach the node (and its ring) but are never read."""

    name = "read-at"
    keepalive_s = 1e9

    def __init__(self, times: set[float]) -> None:
        self.times = times

    def observe(self, capture_ts: float, stability: float) -> None:
        pass

    def due(self, capture_ts: float, stability: float) -> bool:
        return capture_ts in self.times

    def should_estimate(self, entry) -> bool:
        return self.due(entry.capture_ts, 1.0)


def _board_image(kind: str) -> np.ndarray:
    """A checkerboard with a dark piece on the left (old) or right (new) square, or a
    bright hand over both (the same scene as tests/stream/test_backdate.py)."""
    yy, xx = np.mgrid[0:96, 0:160]
    image = np.where(((yy // 16) + (xx // 16)) % 2 == 0, 90, 150).astype(np.uint8)
    if kind == "hand":
        image[20:80, 10:140] = 225
    else:
        x = 20 if kind == "old" else 100
        image[36:60, x : x + 24] = 30
    return np.repeat(image[:, :, None], 3, axis=2)


def test_replay_back_dates_changed_utc_over_the_unread_frames_as_the_live_worker_does(tmp_path):
    """Frames 0-2 show the old board, 3-4 a hand, 5-9 the new one; only 0, 2, 7, 8
    and 9 are read, and the reads see E4 from 7 on. E4 settles at 8 from the run that
    began at 7, and ``changed_utc`` goes back to frame 5, the first unread frame from
    which the changed pixels look like the new read. Live and replay agree line for
    line."""
    t0 = 1_700_000_000.0
    kinds = ["old"] * 3 + ["hand"] * 2 + ["new"] * 5
    recorder = FrameRecorder(tmp_path / "recordings", session_id="src", writer="test")
    for seq, kind in enumerate(kinds):
        frame = Frame(board_id="phone", seq=seq, capture_ts=t0 + seq * FRAME_DT, image=_board_image(kind))
        recorder.record(frame.header(), encode_jpeg(frame.image))
    recorder.stop()
    source = Recording(tmp_path / "recordings" / "src")
    read_at = {t0 + seq * FRAME_DT for seq in (0, 2, 7, 8, 9)}
    script = dict.fromkeys(range(7, 10), E4)

    replayed = _events(_run(source, "backdate", ScriptedRead(script), policy=_ReadAt(read_at)))
    assert [(e["frame_seq"], e["changed_utc"]) for e in replayed] == [
        (0, iso_utc(t0)),
        (2, iso_utc(t0)),
        (7, iso_utc(t0)),
        (8, iso_utc(t0 + 5 * FRAME_DT)),
        (9, iso_utc(t0 + 5 * FRAME_DT)),
    ]

    async def _live() -> list[dict]:
        lines: list[dict] = []
        ingest = FrameIngest(
            ScriptedRead(script),
            PositionHub(),
            on_event=lambda reading, frame, rules: lines.append(
                event_line(reading.event, frame, published_ts=None, rules=rules)
            ),
            policy=lambda: _ReadAt(read_at),
        )
        for recorded in source.frames():
            ingest.submit(recorded.frame())
            await ingest.drain()
        await ingest.close()
        return lines

    live = asyncio.run(_live())
    assert [json.dumps(e, sort_keys=True) for e in replayed] == [
        json.dumps(e, sort_keys=True) for e in live
    ]


def test_cli_replay_and_score(tmp_path, monkeypatch, capsys):
    """``replay --score`` on the walking-skeleton estimator: no model, no GPU.

    Also pins the exit codes, since this is a benchmarking command that will end
    up in scripts: 0 scored, 2 for a recording with no truth to score against.
    """
    monkeypatch.delenv("BR_CHESSQ", raising=False)
    # Cwd-relative writes have cost real user state before; nothing here should
    # write outside tmp_path, and this makes that true.
    monkeypatch.chdir(tmp_path)
    _write_recording(tmp_path / "recordings", frames=4)
    recording = tmp_path / "recordings" / "src"

    assert main(["replay", str(recording), "--name", "cli"]) == 0
    out = capsys.readouterr().out
    assert "4 estimated" in out

    replayed = recording / "replays" / "cli"
    # The environment names no model, so the stack the harness builds is the
    # walking skeleton — the same one `serve` would run with this environment.
    assert json.loads((replayed / "manifest.json").read_text())["replay"][
        "estimator"
    ] == "StubRecognizer"
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


# --------------------------------------------------------------------------- #
# Scheduling: the policy seam and --restamp-stability
# --------------------------------------------------------------------------- #


def _write_metered_recording(
    root, *, moving: range, frames: int = 80, stability: float = 0.29
) -> None:
    """A recording of a still scene interrupted by motion, stamped by a DEAD meter.

    The three older 08-27 takes look exactly like this: the field says "moving"
    on every frame, of a board that spends most of its time untouched.
    """
    rng = np.random.default_rng(7)
    scene = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    recorder = FrameRecorder(root, session_id="metered", writer="test")
    for i in range(frames):
        image = scene.astype(np.int16) + rng.integers(-2, 3, scene.shape)
        if i in moving:
            image[:, (i * 4) % 40 : (i * 4) % 40 + 8] = 0
        image = np.clip(image, 0, 255).astype(np.uint8)
        frame = Frame(
            board_id="phone",
            seq=i,
            capture_ts=1_700_000_000.0 + i * FRAME_DT,
            image=image,
            stability=stability,
        )
        recorder.record(frame.header(), encode_jpeg(image))
    recorder.stop()


@pytest.fixture
def metered(tmp_path):
    # The move sits deep into the take on purpose: the meter starts from the bare
    # sensor-noise floor and needs a second or two of frames to learn this rig's
    # real resting difference, so the opening of any restamped session reads as
    # one long burst. Real takes have the same shape; it costs a spurious
    # burst-end read at the top and nothing after.
    _write_metered_recording(tmp_path / "recordings", moving=range(40, 50))
    return Recording(tmp_path / "recordings" / "metered")


def _index(result: ReplayResult) -> list[dict]:
    text = (result.path / INDEX_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_restamp_overrides_the_recorded_stability_and_marks_it(metered):
    """The recorded field is a broken sensor's; the replayed index says so rather
    than quietly serving a number no camera ever sent."""
    result = _run(metered, "restamped", restamp=True)
    lines = _index(result)

    assert result.stats.frames_restamped == 80
    assert all(line[RESTAMPED_KEY] for line in lines)
    assert lines[0]["stability"] == 1.0  # the meter's first-frame invariant
    assert lines[35]["stability"] > EXIT_BURST, "a still scene reads as still"
    assert min(line["stability"] for line in lines[40:50]) < ENTER_BURST
    # ...and the source is untouched: its own index still carries the old values.
    recorded = [e.header["stability"] for e in metered.frame_index()]
    assert recorded == [pytest.approx(0.29)] * 80


def test_restamping_changes_the_schedule_and_nothing_else(metered):
    """The claim a restamped control run rests on.

    ``stability`` is a header field the estimators copy onto the published event
    and nothing else ever reads — it does not enter recognition or the
    debounce. So an every-frame replay of a restamped take reads the same pixels
    in the same order as an unrestamped one and must publish the same boards; only
    the advisory field differs. If that ever stops being true, a scheduling sweep
    stops being a comparison of schedulers.
    """
    plain = _events(_run(metered, "plain"))
    fresh = _events(_run(metered, "fresh", restamp=True))

    assert len(plain) == len(fresh) == 80
    assert [e["stability"] for e in fresh] != [e["stability"] for e in plain]
    for a, b in zip(plain, fresh, strict=True):
        assert {k: v for k, v in a.items() if k != "stability"} == {
            k: v for k, v in b.items() if k != "stability"
        }


def test_a_burst_policy_reads_the_move_and_skips_the_rest_of_the_stream(metered):
    """End to end on real pixels: re-meter, then schedule off what the meter says.

    The whole point in one number — sixty frames in, a handful of reads out, and
    the reads land after the motion rather than inside it.
    """
    result = _run(
        metered,
        "burst",
        # The floor is pushed out past the take so this test sees the burst
        # decision alone; that it also reads mid-move at 3 s is the D3 floor
        # working, and is pinned in test_schedule.py.
        policy=BurstTriggered(keepalive_s=30.0, burst_end_reads=2),
        restamp=True,
    )
    read = [e["frame_seq"] for e in _events(result)]

    assert result.stats.frames_total == 80  # every frame was offered
    assert len(_index(result)) == 80  # ...and indexed, declined or not
    assert result.stats.frames_estimated == len(read) < 10
    assert not set(read) & set(range(41, 50)), "nothing read mid-move"
    assert any(s > 50 for s in read), "the move-complete moment was read"


def test_the_replay_name_separates_policies_and_restamps(monkeypatch):
    """A sweep's rows must not land in one directory: the last one would be the
    only survivor, which is the exact failure ``default_name`` exists to prevent.
    """
    monkeypatch.setattr(replay_mod, "git_describe", lambda: "v1")
    monkeypatch.setattr(replay_mod, "config_digest", lambda env, settle: "dead")

    assert default_name({}) == "v1-dead"
    assert default_name({}, policy=EveryFrame()) == "v1-dead"
    assert default_name({}, restamp=True) == "v1-dead-restamped"
    assert (
        default_name({}, policy=BurstTriggered(keepalive_s=5.0), restamp=True)
        == "v1-dead-burst-end-keepalive_s5-restamped"
    )
    assert default_name({}, settle=SettleConfig()) == "v1-dead"
    assert default_name({}, settle=SettleConfig(seconds=1.0, min_reads=1)) == (
        "v1-dead-settle1s-1r-2w"
    )
    assert default_name({}, rules="standard") == "v1-dead"
    assert default_name({}, rules="free") == "v1-dead-rules-free"


def test_the_config_digest_moves_only_for_a_non_default_settle():
    """Default-settle digests are the ones every earlier replay carries (cq-v6)."""
    env = {"BR_CHESSQ": {"value": "models/chessq/cq.onnx"}}
    assert replay_mod.config_digest(env) == replay_mod.config_digest(env, SettleConfig())
    assert replay_mod.config_digest(env) != replay_mod.config_digest(
        env, SettleConfig(seconds=2.0)
    )


def test_a_replay_settles_with_the_config_it_is_given(source):
    """``settle`` reaches the stabilizer and the manifest: one read is enough at
    ``min_reads=1`` and a zero window, where the default needs 2.5 s."""
    read = ScriptedRead(dict.fromkeys(range(1, 8), E4))
    fast = SettleConfig(seconds=0.0, min_reads=1)
    result = _run(source, "fast", read, settle=fast)
    settled = [e["placement"] for e in _events(result)]
    assert settled[:2] == [START_PLACEMENT, E4]
    block = json.loads((result.path / "manifest.json").read_text())["replay"]
    assert block["settle_seconds"] == 0.0
    assert block["settle_min_reads"] == 1
    assert block["stale_windows"] == SettleConfig().stale_windows


def test_a_replay_settles_under_the_rules_it_is_given_and_records_them(source):
    """``free`` turns the guard off in replay as on the node; the manifest and
    every line say which setting ran, and a replay under another setting than
    the live run's says so."""
    phantom = "rnbqkbnr/pppppppp/1r6/8/8/8/PPPPPPPP/RNBQKBNR"  # 17 black pieces
    read = ScriptedRead(dict.fromkeys(range(1, 8), phantom))
    fast = SettleConfig(seconds=0.0, min_reads=1)
    standard = _run(source, "standard", read, settle=fast)
    assert {e["placement"] for e in _events(standard)} == {START_PLACEMENT}
    assert {e["rules"] for e in _events(standard)} == {"standard"}

    free = _run(source, "free", ScriptedRead(dict.fromkeys(range(1, 8), phantom)),
                settle=fast, rules="free")
    assert _events(free)[-1]["placement"] == phantom
    assert {e["rules"] for e in _events(free)} == {"free"}
    assert json.loads((free.path / "manifest.json").read_text())["replay"]["rules"] == "free"

    # A live run that settled under free, replayed under standard, warns.
    (source.path / EVENTS_NAME).write_text(json.dumps({"rules": "free"}) + "\n")
    assert replay_mod.recorded_rules(source) == {"free"}
    log = io.StringIO()
    replay(source, source.path / "replays" / "warned", read=ScriptedRead({}),
           reader="ScriptedRead", log=log)
    assert "settled under ['free']" in log.getvalue()


def test_the_manifest_records_the_policy_as_the_argument_that_rebuilds_it(metered):
    policy = BurstTriggered(keepalive_s=5.0, burst_end_reads=2)
    result = _run(metered, "provenance", policy=policy, restamp=True)
    block = json.loads((result.path / "manifest.json").read_text())["replay"]

    assert block["restamp_stability"] is True
    assert parse_policy(block["policy"]) == policy


def test_cli_policy_and_restamp_flags(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BR_CHESSQ", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_metered_recording(tmp_path / "recordings", moving=range(40, 50))
    recording = tmp_path / "recordings" / "metered"

    assert main(
        [
            "replay", str(recording), "--name", "cli-policy",
            "--policy", "burst-end:keepalive_s=5,burst_end_reads=2",
            "--restamp-stability",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "burst-end:keepalive_s=5,burst_end_reads=2" in out
    assert "re-metered on 80 frame(s)" in out
    block = json.loads(
        (recording / "replays" / "cli-policy" / "manifest.json").read_text()
    )["replay"]
    assert block["policy"] == "burst-end:keepalive_s=5,burst_end_reads=2"

    # A typo must cost a line of stderr, not a GPU sweep measuring the defaults.
    assert main(["replay", str(recording), "--name", "x", "--policy", "burst"]) == 2
    assert "unknown policy 'burst'" in capsys.readouterr().err
    assert not (recording / "replays" / "x").exists()


def test_cli_settle_flags_override_the_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BR_CHESSQ", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BR_SETTLE_SECONDS", "1.5")
    monkeypatch.setenv("BR_SETTLE_MIN_READS", "3")
    _write_metered_recording(tmp_path / "recordings", moving=range(40, 50))
    recording = tmp_path / "recordings" / "metered"

    assert main(["replay", str(recording), "--name", "s", "--settle-min-reads", "1"]) == 0
    block = json.loads((recording / "replays" / "s" / "manifest.json").read_text())["replay"]
    assert (block["settle_seconds"], block["settle_min_reads"]) == (1.5, 1)
    assert "settle:    1.5 s, >= 1 reads" in capsys.readouterr().out

    assert main(["replay", str(recording), "--name", "y", "--stale-windows", "0"]) == 2
    assert "stale_windows" in capsys.readouterr().err
    assert not (recording / "replays" / "y").exists()

# --------------------------------------------------------------------------- #
# The session seam: state must not cross it
# --------------------------------------------------------------------------- #


def _write_scripted_session(
    root, *, stabilities, first_ts: float, session_id: str = "multi"
) -> None:
    """One streaming session whose frames carry a scripted ``stability``.

    ``seq`` restarts at 0 for every call, which is what
    :func:`~board_reader_inference.measurement.replay.find_sessions` splits on.
    """
    rng = np.random.default_rng(11)
    recorder = FrameRecorder(root, session_id=session_id, writer="test")
    for i, stability in enumerate(stabilities):
        image = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
        frame = Frame(
            board_id="phone",
            seq=i,
            capture_ts=first_ts + i * FRAME_DT,
            image=image,
            stability=stability,
        )
        recorder.record(frame.header(), encode_jpeg(image))
    recorder.stop()


@pytest.fixture
def two_sessions(tmp_path):
    """Session 1 ends mid-move; session 2 is a board nobody touches."""
    root = tmp_path / "recordings"
    _write_scripted_session(root, stabilities=[0.2] * 6, first_ts=1_700_000_000.0)
    _write_scripted_session(root, stabilities=[0.99] * 20, first_ts=1_700_000_100.0)
    return Recording(root / "multi")


def test_each_replayed_session_gets_its_own_policy(two_sessions):
    """A seq restart is a different stream, so the scheduler starts over on it.

    Session 1 cuts out mid-burst. Carried across the seam, the policy would still
    believe a move was in progress, watch session 2's opening calm frames end
    "the" burst, and spend its whole ``burst_end_reads`` run on a move-complete
    moment for a move that happened in another session — three phantom reads at
    the top of a board nobody has touched. Fresh, session 2 reads its opening
    frame and then nothing until the keepalive, which here outlives the session.
    """
    policy = BurstTriggered(keepalive_s=30.0, burst_end_reads=3, spacing_s=1.0)
    result = _run(two_sessions, "sessions", policy=policy, all_sessions=True)

    second = [
        e["capture_ts"] for e in _events(result) if e["capture_ts"] >= 1_700_000_100.0
    ]
    assert second == [1_700_000_100.0], "session 2 read its opener and nothing else"
    assert result.replayed == [0, 1]


def test_a_policy_that_cannot_be_rebuilt_refuses_to_span_sessions(two_sessions):
    """Rather than silently leaking its state across the seam.

    A caller's own policy object has no spec string to rebuild it from, so there
    is no way to hand each session its own. One session with it is unchanged and
    fine; several is refused, loudly, naming why.
    """
    with pytest.raises(ValueError, match="burst state across every frame"):
        _run(two_sessions, "refused", policy=EvenSeqs(), all_sessions=True)
    # Refused before anything is written: no half-open replay to mistake for one.
    assert not (two_sessions.path / "replays" / "refused").exists()

    # ...and the single-session path still takes any object with the seam on it.
    ok = _run(two_sessions, "single", policy=EvenSeqs())
    assert ok.stats.frames_estimated == 10  # session 2's even seqs, 0..18


# --------------------------------------------------------------------------- #
# --restamp-stability is honest about gaps and about itself
# --------------------------------------------------------------------------- #


def test_a_restamp_refuses_a_recording_with_a_missing_jpeg(metered):
    """Loud beats subtly-wrong.

    The recorder writes the bytes before the line that names them, so an indexed
    frame without its JPEG means a damaged tree. Skipping it would leave that one
    frame carrying the OLD meter's absolute-scale value inside an otherwise
    re-metered stream — a phantom burst, then a fake move-complete read — and its
    index line would silently lack the restamped marker while the directory name
    claims otherwise.
    """
    (metered.path / "jpeg" / "0000" / "000012.jpg").unlink()

    with pytest.raises(ValueError, match="seq 12 has none"):
        _run(metered, "gap", restamp=True)

    # Without the flag the same gap is the ordinary counted case: one frame the
    # disk could not honour, charged to neither the policy nor the run.
    tolerated = _run(metered, "gap-ok")
    assert tolerated.stats.frames_missing == 1
    assert tolerated.stats.frames_estimated == 79


def test_the_manifest_records_the_restamp_FLAG_not_a_count(metered):
    """A run that re-metered nothing is still a restamp run.

    Derived from ``frames_restamped > 0`` this said "not restamped" for a replay
    sitting in a directory named ``-restamped`` — a manifest contradicting its own
    path, which is the one thing provenance may never do.
    """
    result = _run(metered, "flagged", restamp=True, limit=0)
    block = json.loads((result.path / "manifest.json").read_text())["replay"]

    assert result.stats.frames_restamped == 0
    assert block["restamp_stability"] is True
    assert block["frames"]["frames_restamped"] == 0


def test_cli_reports_a_damaged_recording_instead_of_replaying_it(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("BR_CHESSQ", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_metered_recording(tmp_path / "recordings", moving=range(40, 50))
    recording = tmp_path / "recordings" / "metered"
    (recording / "jpeg" / "0000" / "000005.jpg").unlink()

    assert main(["replay", str(recording), "--name", "gap", "--restamp-stability"]) == 2
    err = capsys.readouterr().err
    assert "seq 5 has none" in err
    assert "without --restamp-stability" in err
