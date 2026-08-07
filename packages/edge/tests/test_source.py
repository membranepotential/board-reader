"""Tests for edge frame sources (DESIGN.md §6.1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import itertools

from board_reader_core.sim import render_board
from board_reader_edge.source import (
    FrameSource,
    ImageDirSource,
    RepeatSource,
    SyntheticSource,
    VideoFileSource,
)


def _imgs(n: int) -> list[np.ndarray]:
    return [render_board() for _ in range(n)]


def test_synthetic_source_stamps_monotonic_seq_and_even_timestamps():
    src = SyntheticSource(_imgs(4), board_id="A1", fps=2.0, base_ts=10.0)
    frames = list(src.frames())
    assert [f.seq for f in frames] == [0, 1, 2, 3]
    assert all(f.board_id == "A1" for f in frames)
    # fps=2 -> 0.5s spacing, starting at base_ts.
    assert [f.capture_ts for f in frames] == [10.0, 10.5, 11.0, 11.5]


def test_synthetic_source_honours_injected_clock():
    ticks = iter([100.0, 200.0])
    src = SyntheticSource(_imgs(2), clock=lambda: next(ticks))
    assert [f.capture_ts for f in src.frames()] == [100.0, 200.0]


def test_synthetic_source_satisfies_protocol():
    assert isinstance(SyntheticSource(_imgs(1)), FrameSource)


def test_repeat_source_is_finite_and_lazy():
    img = render_board()
    src = RepeatSource(img, board_id="R1", count=3, fps=2.0)
    frames = list(src.frames())
    assert [f.seq for f in frames] == [0, 1, 2]
    # the same image object is reused (no per-frame copy/render).
    assert all(f.image is img for f in frames)
    assert isinstance(src, FrameSource)


def test_repeat_source_streams_forever_when_count_none():
    src = RepeatSource(render_board(), count=None)
    first = list(itertools.islice(src.frames(), 5))
    assert [f.seq for f in first] == [0, 1, 2, 3, 4]  # would never end without islice


def test_image_dir_source_reads_sorted(tmp_path):
    for i in (2, 0, 1):  # write out of order
        cv2.imwrite(str(tmp_path / f"frame_{i}.png"), render_board())
    src = ImageDirSource(tmp_path, board_id="B1", fps=4.0)
    frames = list(src.frames())
    assert [f.seq for f in frames] == [0, 1, 2]  # sorted by filename
    assert frames[1].capture_ts == pytest.approx(0.25)


def test_image_dir_source_rejects_missing_dir(tmp_path):
    with pytest.raises(NotADirectoryError):
        ImageDirSource(tmp_path / "nope")


def test_video_file_source_replays_frames(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 96))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(render_board(square_px=12))
    writer.release()

    frames = list(VideoFileSource(path, board_id="V1").frames())
    assert len(frames) >= 4  # codecs can drop the odd frame; most must survive
    assert [f.seq for f in frames] == list(range(len(frames)))
    assert all(f.board_id == "V1" for f in frames)
    # capture_ts strictly increases.
    assert all(b.capture_ts > a.capture_ts for a, b in zip(frames, frames[1:]))


def test_video_file_source_missing_file():
    with pytest.raises(FileNotFoundError):
        VideoFileSource("/no/such/clip.mp4")


def test_game_source_streams_the_synthetic_game_lazily():
    from board_reader_core.sim import SAMPLE_GAME_PLACEMENTS
    from board_reader_edge.source import GameSource

    placements = SAMPLE_GAME_PLACEMENTS[:3]
    src = GameSource(
        placements, board_id="G1", fps=12.0,
        frames_per_position=4, transition_frames=2, square_px=12,
    )
    it = src.frames()
    assert iter(it) is it  # a generator: frames render as pulled, O(1) memory
    frames = list(it)
    # len(placements)*frames_per_position + (len-1)*transition_frames
    assert len(frames) == 3 * 4 + 2 * 2
    assert [f.seq for f in frames] == list(range(len(frames)))
    assert all(f.board_id == "G1" for f in frames)
    # capture_ts spaced by exactly 1/fps.
    gaps = [round(b.capture_ts - a.capture_ts, 9) for a, b in zip(frames, frames[1:])]
    assert all(abs(g - 1 / 12.0) < 1e-6 for g in gaps)


def test_loop_source_restamps_seq_and_capture_ts_monotonically():
    from itertools import islice

    from board_reader_edge.source import LoopSource

    inner = SyntheticSource([render_board(square_px=12)] * 2, board_id="L1", fps=2.0)
    frames = list(islice(LoopSource(inner).frames(), 5))
    assert [f.seq for f in frames] == [0, 1, 2, 3, 4]  # never resets at the seam
    ts = [f.capture_ts for f in frames]
    assert ts == sorted(ts) and len(set(ts)) == 5  # strictly increasing
    # The seam continues at the stream's own inter-frame gap (0.5 s at 2 fps).
    gaps = {round(b - a, 6) for a, b in zip(ts, ts[1:])}
    assert gaps == {0.5}


def test_loop_source_refuses_an_empty_source():
    from board_reader_edge.source import LoopSource

    empty = SyntheticSource([], board_id="E1")
    with pytest.raises(RuntimeError):
        next(LoopSource(empty).frames())
