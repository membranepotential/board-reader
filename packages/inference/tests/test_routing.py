"""Tests for the one routing decision (:mod:`board_reader_inference.routing`).

The point of the module under test is that the snapshot ingress (``POST /estimate``)
and the live ingress (``WS /ingest``) cannot read a frame differently.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT, PositionEvent
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference.estimator import StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.routing import (
    PhotoRead,
    RectifiedRead,
    WholeFrameRead,
    route_for,
)
from board_reader_inference.server import create_app
from board_reader_inference.squares import BboxGeometry, full_frame_corners

MIDGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
FLAT = BboxGeometry(0.0, 0.0)


class Spy:
    """A rectified-mode estimator that records how it was entered."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, int]]] = []
        self._stub = StubEstimator(START_PLACEMENT)

    def estimate(self, frame: Frame) -> PositionEvent:
        self.calls.append(("estimate", frame.image.shape[:2]))
        return self._stub.estimate(frame)


class PhotoSpy(Spy):
    """...and one that also offers the original-photo path."""

    crop_mode = "photo"
    geometry = FLAT
    square_size = 64

    def estimate_photo(self, frame: Frame, corners: np.ndarray) -> PositionEvent:
        self.calls.append(("estimate_photo", tuple(np.asarray(corners).ravel().tolist())))
        return self._stub.estimate(frame)


def _frame(image: np.ndarray, corners: np.ndarray | None = None) -> Frame:
    return Frame(board_id="A1", seq=3, capture_ts=1.0, image=image, corners=corners)


# --- the branch table --------------------------------------------------------


def test_photo_mode_with_corners_reads_the_original_pixels():
    image = render_board(square_px=8)
    quad = np.array([[1, 2], [60, 3], [62, 61], [0, 59]], dtype=np.float32)
    route = route_for(PhotoSpy(), _frame(image, quad))

    assert isinstance(route, PhotoRead)
    assert np.allclose(route.corners, quad)


def test_photo_mode_without_corners_treats_the_whole_frame_as_the_board():
    image = render_board(square_px=8)
    route = route_for(PhotoSpy(), _frame(image))

    assert isinstance(route, PhotoRead)
    assert np.allclose(route.corners, full_frame_corners(image))


def test_corners_against_a_rectified_mode_estimator_are_warped_server_side():
    image = render_board(square_px=8)
    quad = np.array([[1, 2], [60, 3], [62, 61], [0, 59]], dtype=np.float32)
    route = route_for(Spy(), _frame(image, quad))

    assert isinstance(route, RectifiedRead)
    assert route.view().image.shape[:2] == (512, 512)  # the top-down warp


def test_no_corners_and_no_photo_path_leaves_the_frame_alone():
    image = render_board(square_px=8)
    route = route_for(Spy(), _frame(image))

    assert isinstance(route, WholeFrameRead)
    assert route.view().image is image


# --- the two ingresses route identically -------------------------------------


def _http_calls(app_state, image, corners) -> list:
    """What the estimator saw when the photo arrived over ``POST /estimate``."""
    spy, app = app_state()
    client = TestClient(app)
    query = "" if corners is None else "?corners=" + ",".join(
        f"{x:.1f},{y:.1f}" for x, y in corners
    )
    r = client.post(
        f"/estimate{query}",
        content=encode_jpeg(image),
        headers={"Content-Type": "image/jpeg"},
    )
    assert r.status_code == 200
    return spy.calls


def _stream_calls(app_state, image, corners, gateway, drain_ingest) -> list:
    """...and when the very same photo arrived over ``WS /ingest``."""
    spy, app = app_state()
    client = gateway(app)
    with client.websocket_connect("/ingest") as ws:
        header, jpeg = _frame(image, corners).encode()
        ws.send_text(json.dumps(header))
        ws.send_bytes(jpeg)
        ws.receive_json()  # ack — the estimate is still in flight
    drain_ingest(client)
    return spy.calls


def _photo_gateway():
    spy = PhotoSpy()
    return spy, create_app(spy, PositionHub())


def _rectified_gateway():
    spy = Spy()
    return spy, create_app(spy, PositionHub())


# The parameter is `build_gateway`, not `gateway`: `gateway` is the conftest
# fixture that builds the client, and a parametrize id would shadow it.
@pytest.mark.parametrize(
    ("build_gateway", "with_corners"),
    [
        (_photo_gateway, True),  # branch 1: photo mode + corners
        (_rectified_gateway, True),  # branch 2: corners -> server-side warp
        (_rectified_gateway, False),  # branch 3: no corners, the JPEG is the board
    ],
)
def test_the_posted_and_the_streamed_frame_take_the_same_branch(
    build_gateway, with_corners, gateway, drain_ingest
):
    """One decision, two ingresses. A branch that reaches ``POST /estimate`` and
    not the stream (or the reverse) is exactly the bug this parity pins down."""
    image = render_board(MIDGAME, square_px=24)
    corners = full_frame_corners(image) if with_corners else None

    assert _http_calls(build_gateway, image, corners) == _stream_calls(
        build_gateway, image, corners, gateway, drain_ingest
    )
