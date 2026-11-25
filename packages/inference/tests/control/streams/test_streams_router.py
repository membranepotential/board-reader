"""``WS /streams`` through the real app, with ``fake_edge.py`` as the edge command."""

from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient, WebSocketTestSession

from board_reader_inference.control.streams.store import state_file
from board_reader_inference.recognition.recognizer import StubRecognizer
from board_reader_inference.server import create_app

FAKE = Path(__file__).with_name("fake_edge.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("BR_CAMERA_CONFIG", str(tmp_path / "camera-config.json"))
    monkeypatch.setenv("BR_EDGE_COMMAND", shlex.join([sys.executable, str(FAKE)]))
    monkeypatch.setenv("BR_NODE_URL", "ws://127.0.0.1:8765")
    return create_app(recognizer=StubRecognizer())


def next_of(ws: WebSocketTestSession, kind: str) -> dict:
    while True:
        message = json.loads(ws.receive_text())
        if message["type"] == kind:
            return message


def test_the_control_ui_adds_lists_and_removes_a_stream(app, tmp_path):
    with TestClient(app) as client, client.websocket_connect("/streams") as ws:
        assert next_of(ws, "streams")["streams"] == []
        ws.send_text(json.dumps({"type": "add_stream", "url": "file:///etc/passwd"}))
        assert "url" in next_of(ws, "refused")["reason"]
        ws.send_text(json.dumps({"type": "add_stream", "url": "https://video.test/sleep", "start_s": 60}))
        listed = next_of(ws, "streams")["streams"]
        assert [(s["id"], s["url"], s["start_s"], s["state"]) for s in listed] == [
            ("stream-1", "https://video.test/sleep", 60.0, "starting"),
        ]
        runs = Path(f"{state_file(tmp_path, 'stream-1')}.runs")
        deadline = time.monotonic() + 5
        while not runs.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert "ws://127.0.0.1:8765" in runs.read_text()
        assert json.loads((tmp_path / "streams.json").read_text())["streams"][0]["id"] == "stream-1"
        ws.send_text(json.dumps({"type": "remove_stream", "stream_id": "stream-1"}))
        assert next_of(ws, "streams")["streams"] == []
        ws.send_text(json.dumps({"type": "remove_stream", "stream_id": "stream-1"}))
        assert next_of(ws, "refused")["reason"] == "no stream stream-1"
    assert json.loads((tmp_path / "streams.json").read_text()) == {"streams": []}
