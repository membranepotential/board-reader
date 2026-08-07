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


def _frame(seq: int, *, board_id: str = "A1", corners=None) -> Frame:
    return Frame(
        board_id=board_id,
        seq=seq,
        capture_ts=100.0 + seq,
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
    header, jpeg = _frame(0).encode()
    for _ in range(3):
        rec.record(header, jpeg)
    rec.close()

    index = (tmp_path / "s1" / INDEX_NAME).read_text().splitlines()
    paths = [json.loads(line)["jpeg"] for line in index]
    assert paths == ["jpeg/0000/000000.jpg", "jpeg/0000/000001.jpg", "jpeg/0000/000002.jpg"]
