"""Tests for the file-based edge adapter (DESIGN.md §6.1, WP-I1)."""

from __future__ import annotations

import json

import numpy as np

from board_reader_core.control import CropRect
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, decode_jpeg

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig
from board_reader_edge.source import SyntheticSource


def _adapter(n=3, **kw):
    src = SyntheticSource([render_board() for _ in range(n)], board_id="A1")
    return FileEdgeAdapter(src, **kw)


def test_envelopes_yield_json_header_and_jpeg_bytes():
    envs = list(_adapter(3).envelopes())
    assert len(envs) == 3
    for seq, (header, jpeg) in enumerate(envs):
        # header must be JSON-serialisable and carry the wire fields.
        json.dumps(header)
        assert header["type"] == "frame"
        assert header["seq"] == seq
        assert header["board_id"] == "A1"
        assert isinstance(jpeg, bytes) and len(jpeg) > 0
        # the JPEG decodes back to an image.
        assert decode_jpeg(jpeg).ndim == 3


def test_envelope_reconstructs_into_a_frame():
    header, jpeg = next(_adapter(1).envelopes())
    frame = Frame.decode(header, jpeg)
    assert frame.board_id == "A1"
    assert frame.seq == 0


def test_frames_passthrough_for_in_process_wiring():
    frames = list(_adapter(2).frames())
    assert [f.seq for f in frames] == [0, 1]


def test_throttle_paces_to_capture_time_using_injected_clock():
    # Fake monotonic clock + recorder so the throttle is deterministic (no real sleep).
    t = [0.0]
    slept: list[float] = []

    def monotonic():
        return t[0]

    def sleep(d):
        slept.append(d)
        t[0] += d  # advancing the clock as a real sleep would

    # SyntheticSource at 10 fps -> capture_ts gaps of 0.1 s drive the pacing.
    src = SyntheticSource([render_board() for _ in range(3)], board_id="A1", fps=10.0)
    adapter = FileEdgeAdapter(src, throttle=True, sleep=sleep, monotonic=monotonic)
    list(adapter.envelopes())
    # 3 frames spanning 0.2 s of capture time -> two ~0.1 s gaps were waited out.
    assert len(slept) == 2
    assert all(abs(s - 0.1) < 1e-6 for s in slept)


def test_throttle_replays_uneven_capture_gaps():
    # Adaptive streams have no fixed period: a keepalive gap then a burst gap.
    t = [0.0]
    slept: list[float] = []

    def monotonic():
        return t[0]

    def sleep(d):
        slept.append(d)
        t[0] += d

    board = render_board()
    src = SyntheticSource([board] * 3, board_id="A1")
    ts = iter([5.0, 6.0, 6.1])  # 1 s at rest, then a 12 fps-ish burst gap
    src.clock = lambda: next(ts)
    adapter = FileEdgeAdapter(src, throttle=True, sleep=sleep, monotonic=monotonic)
    list(adapter.envelopes())
    assert [round(s, 6) for s in slept] == [1.0, 0.1]


def test_forwarder_gates_frames_and_stamps_stability():
    # A static 30 fps second: the meter reads at-rest, so only the keepalive
    # cadence (1 fps) survives the forwarder.
    board = render_board(square_px=48)
    src = SyntheticSource([board] * 30, board_id="A1", fps=30.0)
    fwd = AdaptiveForwarder(ForwarderConfig(keepalive_fps=1.0))
    frames = list(FileEdgeAdapter(src, forwarder=fwd).frames())
    assert 1 <= len(frames) <= 3  # ~1 s of capture at 1 fps keepalive
    assert frames[0].seq == 0  # the first frame is always forwarded
    for f in frames:
        assert 0.0 <= f.stability <= 1.0
        assert f.burst_id is None  # nothing moved


# ------------------------------------------------------------------- static roi

ROI = CropRect(x=0.25, y=0.5, w=0.5, h=0.25)


def test_roi_crops_every_frame_to_its_rectangle():
    board = render_board(square_px=48)  # 384 x 384
    frames = list(_adapter(2, roi=ROI).frames())
    assert [f.seq for f in frames] == [0, 1]
    for f in frames:
        assert f.image.shape == (96, 192, 3)
        assert np.array_equal(f.image, board[192:288, 96:288])


def test_edge_frames_carry_no_corners():
    """The dumb camera never locates the board, so nothing it sends has a quad."""
    for adapter in (_adapter(1), _adapter(1, roi=ROI)):
        header, jpeg = next(adapter.envelopes())
        assert "corners" not in header
        assert Frame.decode(header, jpeg).corners is None


def _motion_outside_roi(n: int) -> SyntheticSource:
    """A still board with a hand-sized blob flickering in its top-left corner —
    i.e. outside :data:`ROI` — on every other frame."""
    still = render_board(square_px=48)
    moved = still.copy()
    moved[:150, :150] = 0
    return SyntheticSource([moved if i % 2 else still for i in range(n)],
                           board_id="A1", fps=30.0)


def test_motion_outside_the_roi_does_not_open_a_burst():
    """The crop comes before the forwarder, so it measures motion on the board's
    rectangle only. Without the roi the same stream does burst — proving the
    flicker is motion the meter would otherwise see."""
    def bursts(**kw) -> set[str | None]:
        fwd = AdaptiveForwarder(ForwarderConfig())
        return {f.burst_id for f in
                FileEdgeAdapter(_motion_outside_roi(30), forwarder=fwd, **kw).frames()}

    assert bursts(roi=ROI) == {None}
    assert bursts() - {None}
