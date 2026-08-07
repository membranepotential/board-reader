"""Tests for the one routing decision (:mod:`board_reader_inference.routing`).

The point of the module under test is that the snapshot ingress (``POST /estimate``)
and the live ingress (``WS /ingest``) cannot read a frame differently. The parity
test below is the one that would have caught the incident that produced it: the
stream's private copy of the decision was written with three branches, missed the
§4.6 uncalibrated one, and published the M0 stub's fixed start placement at 0.95
confidence at stream rate — a fabricated position on the hub.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from board_reader_core.protocol import START_PLACEMENT, PositionEvent
from board_reader_core.sim import render_board
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_inference.centroid import build_exemplars
from board_reader_inference.embed import LiteEmbedder
from board_reader_inference.estimator import CentroidEstimator, StubEstimator
from board_reader_inference.hub import PositionHub
from board_reader_inference.routing import (
    OccupancyRead,
    PhotoRead,
    RectifiedRead,
    Uncalibrated,
    WholeFrameRead,
    route_for,
)
from board_reader_inference.server import create_app
from board_reader_inference.session import EstimatorSession
from board_reader_inference.squares import (
    BboxGeometry,
    crop_squares_photo,
    full_frame_corners,
)

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


def _occupancy_bank() -> CentroidEstimator:
    """Lite-backbone stand-in for the ChessReD occupancy bank, over flat renders."""
    photo = render_board(START_PLACEMENT, square_px=24)
    quad = full_frame_corners(photo)
    emb = LiteEmbedder()
    bank = build_exemplars(
        emb,
        crop_squares_photo(photo, quad, geometry=FLAT, size=64),
        START_PLACEMENT,
        backbone="lite",
        box_headroom=0.0,
        box_side_pad=0.0,
    )
    return CentroidEstimator(emb, bank, square_size=64, geometry=FLAT)


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


def test_the_uncalibrated_branch_outranks_the_rest_and_yields_no_position():
    """§4.6: with an occupancy bank loaded and no piece-type calibration, there is
    nothing that could honestly produce a position — whatever the corners say."""
    image = render_board(MIDGAME, square_px=24)
    session = EstimatorSession(
        StubEstimator(START_PLACEMENT), occupancy=_occupancy_bank()
    )

    for corners in (None, full_frame_corners(image)):
        route = session.route(_frame(image, corners))
        assert isinstance(route, Uncalibrated)
        assert isinstance(route.read(), OccupancyRead)  # occupancy, not a position


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


def _stream_calls(app_state, image, corners) -> list:
    """...and when the very same photo arrived over ``WS /ingest``."""
    spy, app = app_state()
    client = TestClient(app)
    with client.websocket_connect("/ingest") as ws:
        header, jpeg = _frame(image, corners).encode()
        ws.send_text(json.dumps(header))
        ws.send_bytes(jpeg)
        ws.receive_json()  # ack — the estimate is still in flight
        # Wait for the worker to FINISH the frame, not for the spy to be called.
        # On the uncalibrated branch the spy is never called by design, so waiting
        # on `spy.calls` would burn the whole deadline on every run of that case
        # and still assert the same thing. The ingest counters say "this frame is
        # done" for every branch, including the one whose answer is "nothing".
        ingest = client.app.state.ingest
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (
            ingest.frames_estimated + ingest.frames_skipped + ingest.frames_failed
        ):
            time.sleep(0.005)
    return spy.calls


def _photo_gateway():
    spy = PhotoSpy()
    return spy, create_app(spy, PositionHub())


def _rectified_gateway():
    spy = Spy()
    return spy, create_app(spy, PositionHub())


def _uncalibrated_gateway():
    # The style estimator is present but unusable for this piece set (§4.6): the
    # occupancy bank is what decides, and it must keep the spy out of it entirely.
    spy = Spy()
    return spy, create_app(spy, PositionHub(), occupancy_estimator=_occupancy_bank())


@pytest.mark.parametrize(
    ("gateway", "with_corners"),
    [
        (_photo_gateway, True),  # branch 1: photo mode + corners
        (_rectified_gateway, True),  # branch 2: corners -> server-side warp
        (_rectified_gateway, False),  # branch 3: no corners, the JPEG is the board
        (_uncalibrated_gateway, True),  # branch 4: §4.6 — no position, either way
    ],
)
def test_the_posted_and_the_streamed_frame_take_the_same_branch(gateway, with_corners):
    """One decision, two ingresses. A branch that reaches ``POST /estimate`` and
    not the stream (or the reverse) is exactly the bug this parity pins down."""
    image = render_board(MIDGAME, square_px=24)
    corners = full_frame_corners(image) if with_corners else None

    assert _http_calls(gateway, image, corners) == _stream_calls(gateway, image, corners)


def test_the_uncalibrated_stream_publishes_nothing_and_counts_the_frame():
    """A stream has no response to answer with, so it skips — it must never fall
    back to the stub's fixed start placement (the incident)."""
    image = render_board(MIDGAME, square_px=24)
    spy, app = _uncalibrated_gateway()
    client = TestClient(app)

    with client.websocket_connect("/ingest") as ws:
        header, jpeg = _frame(image, full_frame_corners(image)).encode()
        ws.send_text(json.dumps(header))
        ws.send_bytes(jpeg)
        ws.receive_json()

    ingest = client.app.state.ingest
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and ingest.frames_skipped == 0:
        time.sleep(0.01)
    assert ingest.frames_skipped == 1
    assert ingest.frames_estimated == 0
    assert spy.calls == []
    assert client.get("/boards/A1").status_code == 404  # nothing published
