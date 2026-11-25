"""Tests for the L1->L2 wire envelope (DESIGN.md §4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from board_reader_core.wire import FRAME, Frame, FrameHeader, decode_jpeg, encode_jpeg


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


def test_sharpness_rides_the_header_and_defaults_to_none():
    """The edge's focus score is optional: edges that measure it send a float, the
    synthetic feed sends nothing, and both survive the roundtrip unchanged."""
    f = Frame(board_id="A1", seq=1, capture_ts=1.0, image=_img(), sharpness=712.5)
    assert f.header()["sharpness"] == 712.5
    assert Frame.from_header(f.header(), _img()).sharpness == 712.5
    bare = Frame(board_id="A1", seq=2, capture_ts=2.0, image=_img())
    assert bare.sharpness is None
    assert bare.header()["sharpness"] is None
    assert Frame.from_header(bare.header(), _img()).sharpness is None
    # an edge that never heard of the field at all
    legacy = {k: v for k, v in bare.header().items() if k != "sharpness"}
    assert Frame.from_header(legacy, _img()).sharpness is None


def test_the_sending_camera_rides_the_header_only_when_there_is_one():
    """A phone names its device id; an edge without one sends a header with no such
    key at all, exactly as before the field existed."""
    phone = Frame(board_id="board-1", seq=0, capture_ts=1.0, image=_img(), device_id="kitchen-phone-id")
    assert Frame.from_header(phone.header(), _img()).device_id == "kitchen-phone-id"
    bare = Frame(board_id="A1", seq=0, capture_ts=1.0, image=_img())
    assert "device_id" not in bare.header()
    with pytest.raises(ValueError, match="device_id"):
        Frame.from_header({**bare.header(), "device_id": "no"}, _img())


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


QUAD = [[10.0, 20.0], [110.0, 22.0], [112.0, 121.0], [8.0, 119.0]]  # a8,h8,h1,a1


def test_corners_are_normalised_to_a_float32_quad():
    f = Frame(board_id="A1", seq=0, capture_ts=0.0, image=_img(), corners=QUAD)
    assert f.corners.shape == (4, 2)
    assert f.corners.dtype == np.float32


def test_corners_travel_as_eight_flat_floats():
    f = Frame(board_id="A1", seq=1, capture_ts=0.0, image=_img(), corners=QUAD)
    h = f.header()
    assert h["corners"] == [10.0, 20.0, 110.0, 22.0, 112.0, 121.0, 8.0, 119.0]
    assert all(isinstance(v, float) for v in h["corners"])  # JSON-safe, not np.float32


def test_corners_roundtrip_through_header_and_the_jpeg_envelope():
    f = Frame(board_id="A1", seq=2, capture_ts=3.0, image=_img(2), corners=QUAD)
    back = Frame.decode(*f.encode())
    assert np.allclose(back.corners, f.corners)
    assert back.header() == f.header()


def test_frame_without_corners_stays_uncornered():
    f = Frame(board_id="A1", seq=0, capture_ts=0.0, image=_img())
    assert f.corners is None
    assert "corners" not in f.header()
    assert Frame.decode(*f.encode()).corners is None
    # An explicit null is the same as absent — the edge may send either.
    assert Frame.from_header({**f.header(), "corners": None}, _img()).corners is None


@pytest.mark.parametrize("bad", [[1.0, 2.0], [0.0] * 7, [0.0] * 9, [[1.0, 2.0]] * 3])
def test_corners_of_the_wrong_length_are_rejected(bad):
    with pytest.raises(ValueError):
        Frame(board_id="A1", seq=0, capture_ts=0.0, image=_img(), corners=bad)
    with pytest.raises(ValueError):
        Frame.from_header(
            {"board_id": "A1", "seq": 0, "capture_ts": 0.0, "corners": bad}, _img()
        )


def test_the_header_an_edge_sends_is_validated_as_it_is_built():
    """`header()` goes through `FrameHeader`, so a Python edge cannot send what the
    contract refuses (the browser edge is held to the same schema)."""
    for bad in ({"stability": 1.5}, {"sharpness": -1.0}, {"clock": "sideways"}, {"board_id": ""}):
        frame = Frame(**{"board_id": "A1", "seq": 0, "capture_ts": 1.0, "image": _img(), **bad})
        with pytest.raises(ValueError):
            frame.header()


def test_the_header_model_refuses_what_the_wire_does_not_define():
    good = Frame(board_id="A1", seq=0, capture_ts=1.0, image=_img()).header()
    assert FrameHeader.from_dict(good).to_dict() == good
    for bad in ({**good, "stability_restamped": True}, {**good, "seq": "4"}, {**good, "seq": 1.5}):
        with pytest.raises(ValueError):
            FrameHeader.from_dict(bad)
