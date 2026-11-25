"""Auto-frame through the gateway (DESIGN D21): the camera socket, the read queue,
the crop pushed back and the read published, with a recognizer that finds a board.

The registry's decisions are tested in ``control/camera/test_registry_autoframe.py`` and the
controller in ``control/autoframe/``; this file checks only the wiring between
them, ``locate.autoframe`` and the read worker.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
from starlette.testclient import TestClient, WebSocketTestSession

from board_reader_core.protocol import START_PLACEMENT, dist_from_placement
from board_reader_core.wire import encode_jpeg

from board_reader_inference.recognition.locate import Grid
from board_reader_inference.recognition.recognizer import Recognition, StubRecognizer
from board_reader_inference.server import create_app
from board_reader_inference.stream.hub import PositionHub

# Non-hex letters on purpose: test_commit_refs reads 8+ hex characters as a commit.
CAMERA = "/cameras/kitchen-phone-id"
CONTROL = "/boards/board-1/control"
FULL_FRAME = encode_jpeg(np.zeros((540, 960, 3), np.uint8))


class FramingRecognizer:
    """Reads the start position, and sees the board in the middle 60 % of whatever
    it is shown, with a perfect grid."""

    name = "framing"
    model_path = None

    def __init__(self) -> None:
        self.reads = 0
        self._dist = dist_from_placement(START_PLACEMENT, 0.95)

    def read(self, image: np.ndarray) -> Recognition:
        self.reads += 1
        corners = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], np.float32)
        return Recognition(self._dist, grid=Grid(corners, r2=0.99, inliers=64))


def _until(ws: WebSocketTestSession, kind: str, done: Callable[[dict], bool] = bool) -> dict:
    """The next message of ``kind`` for which ``done`` holds (the control page's
    health tick arrives every second, so this cannot hang)."""
    for _ in range(50):
        message = json.loads(ws.receive_text())
        if message["type"] == kind and done(message):
            return message
    raise AssertionError(f"no such {kind} message")


def _send_full_frame(phone: WebSocketTestSession) -> None:
    request = _until(phone, "send_full_frame")["request"]
    phone.send_text(
        json.dumps({"type": "full_frame", "request": request, "capture_ts": 1764000000.5})
    )
    phone.send_bytes(FULL_FRAME)


def test_a_new_camera_is_framed_by_the_node_and_its_read_published(gateway, drain_ingest):
    recognizer = FramingRecognizer()
    client: TestClient = gateway(create_app(recognizer, PositionHub()))
    with client.websocket_connect(CAMERA) as phone:
        _send_full_frame(phone)
        config = _until(phone, "config", lambda m: m["config"]["crop"] is not None)["config"]
        # The board's box (0.2 .. 0.8) and 10 % of it on every side.
        assert config["crop"] == {"x": 0.14, "y": 0.14, "w": 0.72, "h": 0.72}
        assert config["auto"] is True
        assert recognizer.reads == 2  # the whole frame, then the board's box grown

        # The last read saw the board at 60 % of its width: a position read.
        drain_ingest(client)
        event = client.get("/boards/board-1").json()
        assert event["placement"] == START_PLACEMENT

        with client.websocket_connect(CONTROL) as control:
            assert _until(control, "state")["autoframe"] == "framed"
            # A crop drawn by hand switches Auto-frame off.
            drawn = {**config, "crop": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}}
            control.send_text(json.dumps({"type": "configure", "config": drawn}))
            state = _until(control, "state", lambda m: m["autoframe"] == "manual")
            assert state["config"]["auto"] is False


def test_a_full_frame_without_a_board_says_so_on_control(gateway):
    client: TestClient = gateway(create_app(StubRecognizer(START_PLACEMENT), PositionHub()))
    with client.websocket_connect(CAMERA) as phone, client.websocket_connect(CONTROL) as control:
        _send_full_frame(phone)
        # The stub reports no grid: no board, a retry waits.
        _until(control, "state", lambda m: m["autoframe"] == "no-board")
        assert client.get("/boards/board-1").status_code == 404
