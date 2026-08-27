"""Tests for the frame recording format (DESIGN.md WP-H2).

The property that matters throughout: what comes out of a recording is *identical*
to what went in — same header, same bytes. A recording that merely resembles the
run it captured cannot reproduce it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from board_reader_core.recording import (
    FORMAT,
    INDEX_NAME,
    MANIFEST_NAME,
    FrameRecorder,
    Recording,
)
from board_reader_core.wire import Frame, encode_jpeg


def _img(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((32, 32, 3)) * 255).astype(np.uint8)


def _frame(
    seq: int, *, board_id: str = "A1", corners=None, ts: float | None = None
) -> Frame:
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=100.0 + seq if ts is None else ts,
        image=_img(seq),
        stability=0.5,
        corners=corners,
    )


def _record(rec: FrameRecorder, *frames: Frame) -> list[tuple[dict, bytes]]:
    """Record each frame, returning the exact envelopes handed to the recorder."""
    envelopes = [f.encode() for f in frames]
    for header, jpeg in envelopes:
        assert rec.record(header, jpeg)
    return envelopes


def test_records_and_replays_envelopes_byte_for_byte(tmp_path):
    """The round trip is exact: a replay re-sends the original bytes, not a re-encode."""
    rec = FrameRecorder(tmp_path, session_id="s1")
    written = _record(rec, _frame(0), _frame(1), _frame(2))
    rec.close()

    read = list(Recording.open(tmp_path / "s1"))
    assert [(r.header, r.jpeg) for r in read] == written


def test_replayed_header_survives_corners(tmp_path):
    """Corners are the one structured field on the wire — they must ride through."""
    quad = np.array([[1, 2], [30, 4], [31, 28], [3, 27]], dtype=np.float32)
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(rec, _frame(0, corners=quad))
    rec.close()

    (replayed,) = list(Recording.open(tmp_path / "s1"))
    assert replayed.header["corners"] == quad.reshape(-1).tolist()
    assert np.array_equal(replayed.frame().corners, quad)


def test_frame_decodes_back_to_a_wire_frame(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(rec, _frame(7, board_id="B2"))
    rec.close()

    (replayed,) = list(Recording.open(tmp_path / "s1"))
    frame = replayed.frame()
    assert (frame.board_id, frame.seq, frame.capture_ts) == ("B2", 7, 107.0)
    assert frame.stability == 0.5
    assert frame.image.shape == (32, 32, 3)


def test_nothing_is_written_until_the_first_frame(tmp_path):
    """A gateway that runs all day without an edge leaves no directories behind."""
    rec = FrameRecorder(tmp_path / "root", session_id="s1")
    assert not (tmp_path / "root").exists()
    _record(rec, _frame(0))
    assert (tmp_path / "root" / "s1" / INDEX_NAME).exists()


def test_manifest_names_the_format_and_writer(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1", writer="inference-gateway")
    _record(rec, _frame(0))
    rec.close()

    manifest = json.loads((tmp_path / "s1" / MANIFEST_NAME).read_text())
    assert manifest["format"] == FORMAT
    assert manifest["session_id"] == "s1"
    assert manifest["writer"] == "inference-gateway"
    assert Recording.open(tmp_path / "s1").manifest == manifest


def test_index_line_carries_the_header_plus_provenance(tmp_path):
    """The index is greppable on its own — `jq` over it should answer questions."""
    rec = FrameRecorder(tmp_path, session_id="s1", now=lambda: 42.0)
    _record(rec, _frame(3, board_id="B2"))
    rec.close()

    (line,) = (tmp_path / "s1" / INDEX_NAME).read_text().splitlines()
    entry = json.loads(line)
    assert entry["board_id"] == "B2"
    assert entry["seq"] == 3
    assert entry["capture_ts"] == 103.0
    assert entry["received_ts"] == 42.0
    assert entry["bytes"] == (tmp_path / "s1" / entry["jpeg"]).stat().st_size


def test_boards_are_interleaved_in_arrival_order_and_filterable(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(
        rec,
        _frame(0, board_id="A1"),
        _frame(0, board_id="B2"),
        _frame(1, board_id="A1"),
    )
    rec.close()

    recording = Recording.open(tmp_path / "s1")
    assert [r.board_id for r in recording.frames()] == ["A1", "B2", "A1"]
    assert [r.seq for r in recording.frames(board_id="A1")] == [0, 1]
    assert recording.board_ids() == ["A1", "B2"]


def test_open_picks_the_newest_session_in_a_root_directory(tmp_path):
    """`--recording recordings/` means "the run that just happened"."""
    for session in ("20260807-090000", "20260807-120000"):
        rec = FrameRecorder(tmp_path, session_id=session)
        _record(rec, _frame(0))
        rec.close()

    assert Recording.open(tmp_path).path == tmp_path / "20260807-120000"


def test_open_rejects_a_directory_with_no_recording(tmp_path):
    with pytest.raises(FileNotFoundError):
        Recording.open(tmp_path)


def test_a_truncated_session_still_replays_what_it_captured(tmp_path):
    """A killed process loses the frame in flight — never the ones before it."""
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(rec, _frame(0), _frame(1))
    rec.close()

    index = tmp_path / "s1" / INDEX_NAME
    index.write_text(index.read_text() + '{"board_id": "A1", "seq": 2, "cap')

    assert [r.seq for r in Recording.open(tmp_path / "s1")] == [0, 1]


def test_a_frame_whose_jpeg_vanished_is_skipped_not_fatal(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(rec, _frame(0), _frame(1))
    rec.close()

    entry = json.loads((tmp_path / "s1" / INDEX_NAME).read_text().splitlines()[0])
    (tmp_path / "s1" / entry["jpeg"]).unlink()

    assert [r.seq for r in Recording.open(tmp_path / "s1")] == [1]


def test_max_bytes_truncates_the_session_rather_than_filling_the_disk(tmp_path):
    jpeg = encode_jpeg(_img())
    rec = FrameRecorder(tmp_path, session_id="s1", max_bytes=len(jpeg))
    assert rec.record(_frame(0).header(), jpeg)
    assert not rec.record(_frame(1).header(), jpeg)

    assert not rec.active
    assert "max_bytes" in rec.stopped_reason
    assert [r.seq for r in Recording.open(tmp_path / "s1")] == [0]


def test_a_cap_smaller_than_one_frame_still_records_that_frame(tmp_path):
    """A limit must never turn into "record nothing" — the first frame always lands."""
    rec = FrameRecorder(tmp_path, session_id="s1", max_bytes=1)
    assert rec.record(*_frame(0).encode())

    assert not rec.active
    assert [r.seq for r in Recording.open(tmp_path / "s1")] == [0]


def test_a_write_failure_stops_recording_without_raising(tmp_path):
    """Recording is infrastructure: it must never take the stream down with it."""
    blocker = tmp_path / "root"
    blocker.write_text("not a directory")  # mkdir under this can only fail

    rec = FrameRecorder(blocker, session_id="s1")
    assert rec.record(_frame(0).header(), encode_jpeg(_img())) is False
    assert not rec.active
    assert rec.stopped_reason
    assert rec.record(_frame(1).header(), encode_jpeg(_img())) is False


def test_stats_report_what_landed(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(rec, _frame(0), _frame(1))

    stats = rec.stats()
    assert stats["frames"] == 2
    assert stats["bytes"] > 0
    assert stats["active"] is True
    assert stats["session_id"] == "s1"
    assert json.dumps(stats)  # a health endpoint has to be able to serialise it


def test_jpegs_are_sharded_so_a_long_session_stays_navigable(tmp_path):
    """A three-hour session is O(100k) frames; one flat directory is not usable."""
    rec = FrameRecorder(tmp_path, session_id="s1")
    # Advancing seqs: a repeated one would (correctly) be read as a reconnect.
    for seq in range(3):
        rec.record(*_frame(seq).encode())
    rec.close()

    index = (tmp_path / "s1" / INDEX_NAME).read_text().splitlines()
    paths = [json.loads(line)["jpeg"] for line in index]
    assert paths == ["jpeg/0000/000000.jpg", "jpeg/0000/000001.jpg", "jpeg/0000/000002.jpg"]


def test_events_land_beside_the_frames_they_came_from(tmp_path):
    """Published events are the OUTPUTS of a run; they belong in the session dir,
    one JSON line each, flushed like the index so a crash loses at most one."""
    from board_reader_core.recording import EVENTS_NAME

    rec = FrameRecorder(tmp_path, session_id="s", writer="test")
    _record(rec, _frame(0))
    assert rec.record_event({"board_id": "A1", "seq": 0, "placement": "8/8/8/8/8/8/8/8"})
    assert rec.record_event({"board_id": "A1", "seq": 1, "placement": "8/8/8/8/8/8/8/8"})

    lines = (tmp_path / "s" / EVENTS_NAME).read_text().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [0, 1]
    assert rec.stats()["events"] == 2
    rec.close()
    assert not rec.record_event({"seq": 2})  # a closed recorder refuses, quietly


def test_an_event_before_any_frame_still_opens_the_session(tmp_path):
    """A stream can publish its first event while frame 0 is still being written."""
    from board_reader_core.recording import EVENTS_NAME, MANIFEST_NAME

    rec = FrameRecorder(tmp_path, session_id="s", writer="test")
    assert rec.record_event({"seq": 0})
    assert (tmp_path / "s" / MANIFEST_NAME).exists()
    assert (tmp_path / "s" / EVENTS_NAME).exists()
    rec.close()


# --------------------------------------------------------------------------- #
# The index pass — everything a reader can learn without the pixels
# --------------------------------------------------------------------------- #


def test_the_index_reads_without_touching_a_single_jpeg(tmp_path):
    """A session's JPEGs are hundreds of MB; its index is a few hundred kB. A
    reader after timing or board ids must pay only the second."""
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(rec, _frame(0), _frame(1), _frame(2))
    rec.close()
    for jpeg in (tmp_path / "s1").rglob("*.jpg"):
        jpeg.unlink()

    entries = list(Recording.open(tmp_path / "s1").frame_index())

    assert [e.seq for e in entries] == [0, 1, 2]
    assert [e.capture_ts for e in entries] == [100.0, 101.0, 102.0]
    assert all(e.board_id == "A1" for e in entries)
    assert all(e.received_ts > 0 for e in entries)
    # The pixel pass over the same session now yields nothing, which is the point:
    # the two differ exactly by the bytes.
    assert list(Recording.open(tmp_path / "s1").frames()) == []


def test_the_index_filters_by_board_like_the_frames_do(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(
        rec, _frame(0, board_id="A1"), _frame(1, board_id="B2"), _frame(2, board_id="A1")
    )
    rec.close()
    recording = Recording.open(tmp_path / "s1")

    assert [e.seq for e in recording.frame_index(board_id="A1")] == [0, 2]
    assert recording.board_ids() == ["A1", "B2"]


def test_the_index_skips_a_junk_line_like_the_frames_do(tmp_path):
    rec = FrameRecorder(tmp_path, session_id="s1")
    _record(rec, _frame(0), _frame(1))
    rec.close()
    index = tmp_path / "s1" / INDEX_NAME
    index.write_text(index.read_text() + "{not json\n")

    assert [e.seq for e in Recording.open(tmp_path / "s1").frame_index()] == [0, 1]


# --------------------------------------------------------------------------- #
# Rolling — one session directory per streaming session
# --------------------------------------------------------------------------- #


class _Clock:
    """A fake wall clock: ``step`` seconds per read, frozen when it is 0."""

    def __init__(self, start: float = 1_785_000_000.0, step: float = 0.0) -> None:
        self.t = start
        self.step = step

    def __call__(self) -> float:
        t = self.t
        self.t += self.step
        return t


def _session_dirs(root) -> list:
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def test_a_seq_restart_rolls_onto_a_fresh_session_directory(tmp_path):
    """Two streaming sessions are two directories, each internally consistent.

    The wart this fixes: one directory holding both, with `seq` 0..n twice over,
    which no consumer can split (a trial had to be scored on a filtered copy).
    """
    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0), _frame(1), _frame(2))
    first = rec.path
    _record(rec, _frame(0, ts=200), _frame(1, ts=201))  # the edge reconnected
    second = rec.path
    rec.close()

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert [r.seq for r in Recording.open(first)] == [0, 1, 2]
    assert [r.seq for r in Recording.open(second)] == [0, 1]
    assert [r.capture_ts for r in Recording.open(second)] == [200.0, 201.0]
    assert rec.stats()["sessions"] == 2
    assert rec.stats()["frames"] == 5  # run totals, not per directory


def test_each_rolled_session_shards_its_jpegs_from_zero(tmp_path):
    """The new directory is a session in its own right, not a continuation."""
    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0), _frame(1))
    _record(rec, _frame(0, ts=200))
    rec.close()

    index = (rec.path / INDEX_NAME).read_text().splitlines()
    assert [json.loads(line)["jpeg"] for line in index] == ["jpeg/0000/000000.jpg"]
    assert (rec.path / "jpeg" / "0000" / "000000.jpg").exists()


def test_a_rolled_manifest_says_why_the_directory_ended(tmp_path):
    """A directory that stops mid-stream must not read as a crash."""
    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0))
    _record(rec, _frame(0, ts=200))
    rolled = rec.path
    rec.close()

    closed = Recording.open(tmp_path / "s1").manifest
    assert closed["closed_reason"] == "rolled: new streaming session"
    assert closed["closed_ts"] > 0
    opened = Recording.open(rolled).manifest
    assert opened["session_id"] == rolled.name
    assert opened["rolled_from"] == "s1"
    assert opened["closed_reason"] == "closed"  # stop() stamps its reason too


def test_rolled_sessions_in_the_same_second_get_distinct_names(tmp_path):
    """A frozen clock is the worst case: the ids must still be unique and sorted."""
    rec = FrameRecorder(tmp_path, now=_Clock())  # no explicit id, no clock movement
    _record(rec, _frame(0))
    _record(rec, _frame(0, ts=200))
    _record(rec, _frame(0, ts=300))
    rec.close()

    names = _session_dirs(tmp_path)
    assert len(names) == len(set(names)) == 3
    assert names == sorted(names)  # chronological order survives the suffix
    assert all(Recording.open(tmp_path / name) for name in names)


def test_a_monotonic_stream_never_rolls(tmp_path):
    """Gaps are dropped frames, not a new session — one take, one directory."""
    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0), _frame(1), _frame(5), _frame(9))
    rec.close()

    assert _session_dirs(tmp_path) == ["s1"]
    assert [r.seq for r in Recording.open(tmp_path / "s1")] == [0, 1, 5, 9]


def test_a_second_board_starting_at_zero_is_not_a_new_session(tmp_path):
    """`seq` is per board; two boards on one recorder both count from 0."""
    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(
        rec,
        _frame(0, board_id="A1"),
        _frame(1, board_id="A1"),
        _frame(0, board_id="B2"),
        _frame(2, board_id="A1"),
    )
    rec.close()

    assert _session_dirs(tmp_path) == ["s1"]
    assert Recording.open(tmp_path / "s1").board_ids() == ["A1", "B2"]


def test_a_late_event_lands_in_the_session_it_was_estimated_from(tmp_path):
    """THE race: an estimate that started before the roll publishes after it.

    `seq`/`frame_seq` cannot tell the sessions apart (both start at 0), so the
    line is routed by `capture_ts` against the roll boundary.
    """
    from board_reader_core.recording import EVENTS_NAME

    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0), _frame(1))
    _record(rec, _frame(0, ts=200))  # rolls: boundary is capture_ts 200
    rolled = rec.path

    assert rec.record_event({"frame_seq": 1, "capture_ts": 101.0, "tag": "late"})
    assert rec.record_event({"frame_seq": 0, "capture_ts": 200.0, "tag": "new"})
    rec.close()

    old_lines = (tmp_path / "s1" / EVENTS_NAME).read_text().splitlines()
    new_lines = (rolled / EVENTS_NAME).read_text().splitlines()
    assert [json.loads(line)["tag"] for line in old_lines] == ["late"]
    assert [json.loads(line)["tag"] for line in new_lines] == ["new"]
    assert rec.stats()["events"] == 2


def test_events_before_and_after_a_roll_split_by_capture_time(tmp_path):
    """The pre-roll events stay put; the recorder appends to that same file."""
    from board_reader_core.recording import EVENTS_NAME

    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0), _frame(1))
    assert rec.record_event({"frame_seq": 0, "capture_ts": 100.0})
    _record(rec, _frame(0, ts=200), _frame(1, ts=201))
    rolled = rec.path
    assert rec.record_event({"frame_seq": 1, "capture_ts": 101.0})  # late
    assert rec.record_event({"frame_seq": 1, "capture_ts": 201.0})
    rec.close()

    old = [json.loads(x) for x in (tmp_path / "s1" / EVENTS_NAME).read_text().splitlines()]
    new = [json.loads(x) for x in (rolled / EVENTS_NAME).read_text().splitlines()]
    assert [e["capture_ts"] for e in old] == [100.0, 101.0]
    assert [e["capture_ts"] for e in new] == [201.0]


def test_an_event_without_a_capture_ts_goes_to_the_current_session(tmp_path):
    """Nothing to route on — the current directory is the only honest guess."""
    from board_reader_core.recording import EVENTS_NAME

    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0))
    _record(rec, _frame(0, ts=200))
    rolled = rec.path
    assert rec.record_event({"frame_seq": 0})
    rec.close()

    assert not (tmp_path / "s1" / EVENTS_NAME).exists()
    assert len((rolled / EVENTS_NAME).read_text().splitlines()) == 1


def test_routing_reaches_exactly_one_session_back(tmp_path):
    """The documented limit: an event older than the current session goes to the
    previous directory, and nothing tracks further back than that.

    A read takes a second or two where sessions are minutes apart, so an event
    that outlives two rolls does not happen; this pins what would become of it
    rather than pretending the recorder can place it."""
    from board_reader_core.recording import EVENTS_NAME

    rec = FrameRecorder(tmp_path, session_id="s1", now=_Clock(step=1.0))
    _record(rec, _frame(0))
    _record(rec, _frame(0, ts=200))
    middle = rec.path
    _record(rec, _frame(0, ts=300))
    rolled = rec.path
    assert rec.record_event({"frame_seq": 0, "capture_ts": 100.0})  # from session 1
    rec.close()

    assert not (tmp_path / "s1" / EVENTS_NAME).exists()
    assert not (rolled / EVENTS_NAME).exists()
    assert len((middle / EVENTS_NAME).read_text().splitlines()) == 1


def test_max_bytes_is_the_budget_for_the_whole_run_not_for_each_session(tmp_path):
    """A rolling recorder must not turn a disk-fill cap into a cap per reconnect."""
    jpeg = encode_jpeg(_img())
    rec = FrameRecorder(
        tmp_path, session_id="s1", max_bytes=3 * len(jpeg), now=_Clock(step=1.0)
    )
    assert rec.record(_frame(0).header(), jpeg)
    assert rec.record(_frame(1).header(), jpeg)
    assert rec.record(_frame(0, ts=200).header(), jpeg)  # rolls, and spends the budget
    rolled = rec.path

    assert not rec.active
    assert "max_bytes" in rec.stopped_reason
    assert not rec.record(_frame(1, ts=201).header(), jpeg)
    assert [r.seq for r in Recording.open(tmp_path / "s1")] == [0, 1]
    assert [r.seq for r in Recording.open(rolled)] == [0]


def test_open_on_the_root_finds_the_newest_rolled_session(tmp_path):
    """`--recording recordings/` still means "the run that just happened"."""
    rec = FrameRecorder(tmp_path, now=_Clock(step=1.0))
    _record(rec, _frame(0))
    _record(rec, _frame(0, ts=200))
    rec.close()

    assert Recording.open(tmp_path).path == rec.path
