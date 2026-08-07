"""Tests for the file-based edge adapter (DESIGN.md §6.1, WP-I1)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from board_reader_core.sim import render_board, simulate_angled_view
from board_reader_core.wire import Frame, decode_jpeg
from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.rectify import Rectifier
from board_reader_edge.source import SyntheticSource

# Deliberately NOT top-left-first: this is the operator's semantic a8,h8,h1,a1 order
# starting at a board whose a8 sits top-right in the image. order_quad() would
# reshuffle it; the forward-original path must not.
ROTATED_CORNERS = np.array(
    [[300.0, 20.0], [300.0, 300.0], [20.0, 300.0], [20.0, 20.0]], dtype=np.float32
)


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


def test_rectifier_rectifies_forwarded_frames_to_canonical_crops():
    board = render_board(square_px=48)
    angled, gt = simulate_angled_view(board)
    src = SyntheticSource([angled, angled], board_id="A1")
    rect = Rectifier(out_size=board.shape[0], manual_corners=gt)
    frames = list(FileEdgeAdapter(src, rectifier=rect).frames())
    assert len(frames) == 2
    # Forwarded image is the canonical crop (board's size), not the larger angled view.
    assert frames[0].image.shape == board.shape
    assert frames[0].image.shape != angled.shape


def test_rectifier_suppresses_frames_it_cannot_localise():
    # A blank frame between two good ones: auto rectifier drops the blank (needs_setup).
    # redetect_interval_s=0 forces per-frame detection — at the default cadence the
    # blank would ride the cached homography unexamined (see test_rectify.py).
    board = render_board(square_px=48)
    angled, _ = simulate_angled_view(board)
    blank = np.zeros_like(angled)
    src = SyntheticSource([angled, blank, angled], board_id="A1")
    rect = Rectifier(out_size=board.shape[0], redetect_interval_s=0)
    frames = list(FileEdgeAdapter(src, rectifier=rect).frames())
    # The blank is suppressed; only the two locatable frames are forwarded.
    assert len(frames) == 2


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
    from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig

    board = render_board(square_px=48)
    src = SyntheticSource([board] * 30, board_id="A1", fps=30.0)
    fwd = AdaptiveForwarder(ForwarderConfig(keepalive_fps=1.0))
    frames = list(FileEdgeAdapter(src, forwarder=fwd).frames())
    assert 1 <= len(frames) <= 3  # ~1 s of capture at 1 fps keepalive
    assert frames[0].seq == 0  # the first frame is always forwarded
    for f in frames:
        assert 0.0 <= f.stability <= 1.0
        assert f.burst_id is None  # nothing moved


def test_dropped_frames_never_pay_for_rectification():
    # Stage order is cost-driven: rectification runs only on forwarded frames.
    from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig

    board = render_board(square_px=48)
    angled, gt = simulate_angled_view(board)

    class CountingRectifier(Rectifier):
        calls = 0

        def process(self, image, ts=None):
            CountingRectifier.calls += 1
            return super().process(image, ts)

    src = SyntheticSource([angled] * 30, board_id="A1", fps=30.0)
    fwd = AdaptiveForwarder(ForwarderConfig(keepalive_fps=1.0))
    rect = CountingRectifier(out_size=board.shape[0], manual_corners=gt)
    frames = list(FileEdgeAdapter(src, forwarder=fwd, rectifier=rect).frames())
    assert CountingRectifier.calls == len(frames)  # dropped frames cost nothing
    assert CountingRectifier.calls < 30
    assert frames[0].image.shape == board.shape  # and forwarded ones are crops


def test_forward_original_keeps_the_image_and_attaches_corners_verbatim():
    # The photo path (§4.5): no warp at the edge, geometry rides the header instead.
    board = render_board(square_px=48)
    angled, _ = simulate_angled_view(board)
    src = SyntheticSource([angled, angled], board_id="A1")
    frames = list(FileEdgeAdapter(src, wire_corners=ROTATED_CORNERS).frames())
    assert len(frames) == 2
    for f in frames:
        assert f.image.shape == angled.shape  # original pixels, not a 384px crop
        # Verbatim: the semantic a8,h8,h1,a1 order is what declares the orientation,
        # so it must survive unreordered (order_quad would put [20,20] first).
        assert np.array_equal(f.corners, ROTATED_CORNERS)


def test_forward_original_corners_survive_caller_mutation():
    # The adapter must own its quad: a recalibration UI writing new corners into
    # the caller's array in place must not rewrite frames already handed out.
    board = render_board(square_px=48)
    src = SyntheticSource([board, board], board_id="A1")
    caller_quad = ROTATED_CORNERS.copy()
    adapter = FileEdgeAdapter(src, wire_corners=caller_quad)
    it = adapter.frames()
    first = next(it)
    caller_quad[:] = 999.0
    assert np.array_equal(first.corners, ROTATED_CORNERS)
    assert np.array_equal(next(it).corners, ROTATED_CORNERS)


def test_forward_original_puts_eight_corner_floats_on_the_wire():
    board = render_board(square_px=48)
    src = SyntheticSource([board], board_id="A1")
    header, jpeg = next(FileEdgeAdapter(src, wire_corners=ROTATED_CORNERS).envelopes())
    assert header["corners"] == ROTATED_CORNERS.reshape(-1).tolist()
    json.dumps(header)  # still JSON-safe with corners attached
    assert np.array_equal(Frame.decode(header, jpeg).corners, ROTATED_CORNERS)


def test_forward_original_still_obeys_the_forwarder():
    # The motion gate is upstream of the corner story: it scores raw frames either way.
    from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig

    board = render_board(square_px=48)
    src = SyntheticSource([board] * 30, board_id="A1", fps=30.0)
    fwd = AdaptiveForwarder(ForwarderConfig(keepalive_fps=1.0))
    frames = list(
        FileEdgeAdapter(src, forwarder=fwd, wire_corners=ROTATED_CORNERS).frames()
    )
    assert 1 <= len(frames) <= 3  # ~1 s of capture at 1 fps keepalive
    assert all(np.array_equal(f.corners, ROTATED_CORNERS) for f in frames)


def test_rectifier_and_wire_corners_are_rejected_together():
    src = SyntheticSource([render_board()], board_id="A1")
    with pytest.raises(ValueError, match="exclusive"):
        FileEdgeAdapter(src, rectifier=Rectifier(), wire_corners=ROTATED_CORNERS)


def test_wire_corners_must_be_four_points():
    src = SyntheticSource([render_board()], board_id="A1")
    with pytest.raises(ValueError, match="4"):
        FileEdgeAdapter(src, wire_corners=np.array([[1.0, 2.0], [3.0, 4.0]]))
