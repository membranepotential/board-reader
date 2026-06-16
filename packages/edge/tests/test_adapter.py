"""Tests for the file-based edge adapter (DESIGN.md §6.1, WP-I1)."""

from __future__ import annotations

import json

import numpy as np

from board_reader_core.sim import render_board, simulate_angled_view
from board_reader_core.wire import Frame, decode_jpeg
from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.rectify import Rectifier
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
    board = render_board(square_px=48)
    angled, _ = simulate_angled_view(board)
    blank = np.zeros_like(angled)
    src = SyntheticSource([angled, blank, angled], board_id="A1")
    frames = list(FileEdgeAdapter(src, rectifier=Rectifier(out_size=board.shape[0])).frames())
    # The blank is suppressed; only the two locatable frames are forwarded.
    assert len(frames) == 2


def test_throttle_sleeps_between_frames_using_injected_clock():
    # Fake monotonic clock + recorder so the throttle is deterministic (no real sleep).
    t = [0.0]
    slept: list[float] = []

    def monotonic():
        return t[0]

    def sleep(d):
        slept.append(d)
        t[0] += d  # advancing the clock as a real sleep would

    adapter = _adapter(3, throttle=True, fps=10.0, sleep=sleep, monotonic=monotonic)
    list(adapter.envelopes())
    # 3 frames at 10 fps -> two inter-frame gaps of ~0.1s were waited out.
    assert len(slept) >= 2
    assert all(s > 0 for s in slept)
