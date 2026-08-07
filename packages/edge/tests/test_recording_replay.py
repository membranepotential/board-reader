"""Replaying a recorded session from the edge (DESIGN.md WP-H2).

The contract under test is *reproduction*: what a gateway once received is what it
receives again. Anything that would quietly alter the stream — a JPEG re-encode, a
motion gate, a rectifier — is either refused or absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.recording import FrameRecorder, Recording
from board_reader_core.wire import Frame
from board_reader_edge.adapter import EnvelopeSource, RecordingReplay
from board_reader_edge.cli import build_parser, cmd_feed


def _frame(seq: int, *, board_id: str = "A1", gap: float = 1.0) -> Frame:
    rng = np.random.default_rng(seq)
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=500.0 + seq * gap,
        image=(rng.random((32, 32, 3)) * 255).astype(np.uint8),
        stability=0.75,
    )


@pytest.fixture
def session(tmp_path):
    """A recorded session of three A1 frames. Yields (root, sent envelopes)."""
    rec = FrameRecorder(tmp_path / "rec", session_id="20260807-120000")
    sent = [f.encode() for f in (_frame(0), _frame(1), _frame(2))]
    for header, jpeg in sent:
        rec.record(header, jpeg)
    rec.close()
    return tmp_path / "rec", sent


def test_replay_re_sends_the_recorded_envelopes_unmodified(session):
    root, sent = session
    assert list(RecordingReplay(root).envelopes()) == sent


def test_replay_does_not_re_encode_the_jpeg(session):
    """A decode/re-encode round trip perturbs pixels — and a run fed perturbed
    pixels is not a reproduction of the one that was recorded."""
    root, sent = session
    replayed = [jpeg for _, jpeg in RecordingReplay(root).envelopes()]
    original = [jpeg for _, jpeg in sent]

    assert replayed == original
    # Guard the premise: re-encoding really would have changed the bytes.
    assert Frame.decode(*sent[0]).encode()[1] != original[0]


def test_replay_satisfies_the_envelope_source_protocol(session):
    root, _ = session
    assert isinstance(RecordingReplay(root), EnvelopeSource)


def test_replay_takes_the_newest_session_from_a_root_directory(tmp_path):
    for session_id, seq in (("20260807-090000", 0), ("20260807-120000", 9)):
        rec = FrameRecorder(tmp_path, session_id=session_id)
        rec.record(*_frame(seq).encode())
        rec.close()

    assert [h["seq"] for h, _ in RecordingReplay(tmp_path).envelopes()] == [9]


def test_replay_can_select_one_board(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    for f in (_frame(0, board_id="A1"), _frame(0, board_id="B2"), _frame(1)):
        rec.record(*f.encode())
    rec.close()

    replay = RecordingReplay(tmp_path / "s1", board_id="A1")
    assert [h["board_id"] for h, _ in replay.envelopes()] == ["A1", "A1"]


def test_frames_decodes_for_reprocessing(session):
    """The other direction: a recording as a frame source, when the point is to
    re-run the edge stages over it rather than to reproduce them."""
    root, _ = session
    frames = list(RecordingReplay(root).frames())

    assert [f.seq for f in frames] == [0, 1, 2]
    assert frames[0].image.shape == (32, 32, 3)
    assert frames[0].stability == 0.75


def test_throttle_paces_by_the_recorded_capture_times(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    for f in (_frame(0), _frame(1, gap=2.0), _frame(2, gap=2.0)):
        rec.record(*f.encode())
    rec.close()

    slept: list[float] = []
    clock = [0.0]
    replay = RecordingReplay(
        tmp_path / "s1",
        throttle=True,
        sleep=lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)),
        monotonic=lambda: clock[0],
    )
    list(replay.envelopes())

    # capture_ts 500, 502, 504 -> the gaps between them, not the absolute values.
    # The first frame goes out immediately: pacing is relative to it, not to epoch.
    assert slept == pytest.approx([2.0, 2.0], abs=1e-6)


def test_unthrottled_replay_never_sleeps(session):
    root, _ = session
    replay = RecordingReplay(
        root, sleep=lambda s: pytest.fail("unthrottled replay must run flat out")
    )
    assert len(list(replay.envelopes())) == 3


def test_replay_of_an_empty_directory_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        RecordingReplay(tmp_path)


@pytest.mark.parametrize(
    "flag",
    ["--adaptive", "--rectify", "--forward-original", "--loop"],
)
def test_cli_refuses_to_reprocess_a_recording(session, flag):
    """A recording already holds the edge's output. Re-running the edge stages over
    it would send something other than what was recorded — refuse, loudly."""
    root, _ = session
    args = build_parser().parse_args(["feed", "--recording", str(root), flag])

    with pytest.raises(SystemExit) as e:
        cmd_feed(args)
    assert flag in str(e.value)


def test_cli_refuses_a_recording_with_corners(session):
    root, _ = session
    args = build_parser().parse_args(
        ["feed", "--recording", str(root), "--corners", "0,0,1,0,1,1,0,1"]
    )
    with pytest.raises(SystemExit) as e:
        cmd_feed(args)
    assert "--corners" in str(e.value)


def test_cli_reports_a_missing_recording_without_a_traceback(tmp_path):
    args = build_parser().parse_args(["feed", "--recording", str(tmp_path / "nope")])
    with pytest.raises(SystemExit):
        cmd_feed(args)


def test_cli_replay_streams_the_recording(session, monkeypatch):
    """The whole path: `feed --recording` -> the same envelopes on the wire."""
    root, sent = session
    streamed: list[tuple[dict, bytes]] = []

    async def fake_stream(adapter, url, **kw):
        streamed.extend(adapter.envelopes())
        return len(streamed)

    monkeypatch.setattr("board_reader_edge.cli.stream_to_gateway", fake_stream)
    args = build_parser().parse_args(["feed", "--recording", str(root)])

    assert cmd_feed(args) == 0
    assert streamed == sent


def test_a_recording_round_trips_through_a_gateway_unchanged(tmp_path):
    """End to end: record a stream, replay it, and the second recording is the
    first. This is the property "reproduce a test run" actually rests on."""
    first = FrameRecorder(tmp_path / "one", session_id="s1")
    for f in (_frame(0), _frame(1), _frame(2)):
        first.record(*f.encode())
    first.close()

    second = FrameRecorder(tmp_path / "two", session_id="s1")
    for header, jpeg in RecordingReplay(tmp_path / "one" / "s1").envelopes():
        second.record(header, jpeg)
    second.close()

    def contents(root):
        return [(r.header, r.jpeg) for r in Recording.open(root)]

    assert contents(tmp_path / "two") == contents(tmp_path / "one")
