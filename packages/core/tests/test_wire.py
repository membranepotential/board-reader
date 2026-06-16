"""Tests for the L1->L2 wire envelope (DESIGN.md §4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.wire import FRAME, Frame, decode_jpeg, encode_jpeg


def _img(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((48, 48, 3)) * 255).astype(np.uint8)


def test_jpeg_roundtrip_preserves_shape():
    img = _img()
    out = decode_jpeg(encode_jpeg(img))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_decode_jpeg_rejects_garbage():
    with pytest.raises(ValueError):
        decode_jpeg(b"not a jpeg")


def test_frame_header_is_json_safe_and_complete():
    f = Frame(board_id="A1", seq=3, capture_ts=12.5, image=_img(), stability=0.4, clock="white")
    h = f.header()
    assert h["type"] == FRAME
    assert h["board_id"] == "A1"
    assert h["seq"] == 3
    assert h["capture_ts"] == 12.5
    assert h["stability"] == 0.4
    assert h["clock"] == "white"
    assert "image" not in h  # the image never travels in the header


def test_frame_encode_decode_roundtrip():
    f = Frame(board_id="C3", seq=9, capture_ts=1.0, image=_img(1), stability=0.2, clock="black")
    header, jpeg = f.encode()
    back = Frame.decode(header, jpeg)
    assert back.board_id == "C3"
    assert back.seq == 9
    assert back.capture_ts == 1.0
    assert back.stability == 0.2
    assert back.clock == "black"
    assert back.image.shape == f.image.shape
