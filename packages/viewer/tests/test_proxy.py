"""Tests for the viewer's same-origin relay to the inference node.

Both directions are exercised against a *real* server on an ephemeral port rather than
a mock: the thing under test is that bytes, framing, query strings and status codes
survive a genuine hop, which a stubbed transport cannot tell you.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from starlette.testclient import TestClient
from websockets.asyncio.server import serve

from board_reader_viewer import proxy
from board_reader_viewer.server import create_app


# --------------------------------------------------------------------------- HTTP
class _EchoHandler(BaseHTTPRequestHandler):
    """Reflects the request back as JSON, so the test can assert what arrived."""

    protocol_version = "HTTP/1.1"

    def _reflect(self, method: str) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        payload = json.dumps(
            {
                "path": self.path,
                "method": method,
                "content_type": self.headers.get("content-type"),
                "body": body.decode("latin-1"),
            }
        ).encode()
        # A path the viewer must relay a *refusal* for, unaltered.
        status = 422 if self.path.endswith("/camera/report") else 200
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        # The node states which frame it served here (``/boards/{id}/frame``).
        self.send_header("X-Seq", "42")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        self._reflect("POST")

    def do_GET(self) -> None:
        self._reflect("GET")

    def do_PUT(self) -> None:
        self._reflect("PUT")

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def inference_http(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("BR_INFERENCE_HTTP", f"http://127.0.0.1:{server.server_port}")
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_proxy_forwards_body_query_and_content_type(inference_http):
    client = TestClient(create_app())
    response = client.post(
        "/api/estimate?board_id=snapshot&note=10,20,30",
        content=b"\xff\xd8jpeg-ish",
        headers={"Content-Type": "image/jpeg"},
    )
    assert response.status_code == 200
    echoed = response.json()
    # The query string must survive character-for-character: a re-encoded comma would
    # reach the node as a different value.
    assert echoed["path"] == "/estimate?board_id=snapshot&note=10,20,30"
    assert echoed["content_type"] == "image/jpeg"
    assert echoed["body"] == "\xff\xd8jpeg-ish"


def test_http_proxy_returns_the_nodes_refusal_verbatim(inference_http):
    """The node refuses a malformed camera report in its own words; a relay that
    rewrote that into a generic error would strip the only diagnosis the operator has."""
    response = TestClient(create_app()).post("/api/boards/phone/camera/report", content=b"{}")
    assert response.status_code == 422
    assert response.json()["path"] == "/boards/phone/camera/report"
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    "path",
    [
        "/api/estimate?board_id=snapshot",
        "/api/labels/upload-7",
        "/api/boards/board-1/frame?seq=42",
        "/api/uploads/abc/photo?max=160",
    ],
)
def test_http_proxy_relays_any_node_path_under_the_prefix(inference_http, path):
    """The relay is a catch-all, not a list of blessed endpoints.

    The pages between them reach /estimate, /labels/{id}, /boards/{id}/frame,
    /boards/{id}/camera/* and /uploads/{id}/photo; enumerating those here and in proxy.install() would mean a
    page could call something the relay 404s on."""
    response = TestClient(create_app()).get(path)
    assert response.status_code == 200
    assert response.json()["path"] == path.removeprefix("/api")


def test_http_proxy_relays_get_as_get(inference_http):
    """Image and listing endpoints are GETs; forwarding them as POST would 501."""
    assert TestClient(create_app()).get("/api/uploads?limit=1").json()["method"] == "GET"


def test_http_proxy_relays_put_with_its_body(inference_http):
    """The control UI replaces a board's camera config with a PUT."""
    response = TestClient(create_app()).put(
        "/api/boards/phone/camera",
        content=b'{"streaming": true}',
        headers={"content-type": "application/json"},
    )
    reflected = response.json()
    assert (reflected["method"], reflected["path"], reflected["body"]) == (
        "PUT",
        "/boards/phone/camera",
        '{"streaming": true}',
    )


def test_http_proxy_forwards_response_headers_the_pages_act_on(inference_http):
    """The node states things in headers (X-Seq: which frame ``/boards/{id}/frame``
    served; X-Preview-Seq on the camera preview). A relay that forwarded only the body
    would drop them silently."""
    response = TestClient(create_app()).get("/api/boards/b/frame")
    assert response.headers["x-seq"] == "42"


def test_http_proxy_does_not_copy_the_upstream_framing_headers(inference_http):
    """content-length describes the upstream hop's body; copying it onto ours would
    contradict what Starlette actually sends."""
    response = TestClient(create_app()).get("/api/uploads")
    assert int(response.headers["content-length"]) == len(response.content)


def test_http_proxy_reports_an_unreachable_node_as_a_bad_gateway(monkeypatch):
    # Port 1 on loopback: nothing listens, and the connect fails fast.
    monkeypatch.setenv("BR_INFERENCE_HTTP", "http://127.0.0.1:1")
    response = TestClient(create_app()).post("/api/estimate", content=b"photo")
    assert response.status_code == 502
    assert "127.0.0.1:1" in response.json()["error"]


# ----------------------------------------------------------------------- WebSocket
async def _echo_handler(connection) -> None:
    """Echo every message back with its frame type preserved."""
    async for message in connection:
        if isinstance(message, str):
            await connection.send("echo:" + message)
        else:
            await connection.send(b"echo:" + message)


@pytest.fixture
def inference_ws(monkeypatch):
    port_ready = threading.Event()
    port: list[int] = []
    stop = threading.Event()

    async def run() -> None:
        async with serve(_echo_handler, "127.0.0.1", 0) as server:
            port.append(server.sockets[0].getsockname()[1])
            port_ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.02)

    thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    thread.start()
    assert port_ready.wait(timeout=5)
    monkeypatch.setenv("BR_INFERENCE_WS", f"ws://127.0.0.1:{port[0]}/viewer")
    try:
        yield port[0]
    finally:
        stop.set()
        thread.join(timeout=5)


@pytest.mark.parametrize("path", ["/ws/ingest", "/ws/viewer"])
def test_ws_proxy_relays_text_and_binary_both_ways(inference_ws, path):
    """The ingest protocol is a JSON header (text) then its JPEG (binary), so the relay
    has to preserve the frame type as well as the bytes — a pipe that stringified the
    JPEG would corrupt every frame."""
    with TestClient(create_app()).websocket_connect(path) as ws:
        ws.send_text('{"type":"frame","seq":1}')
        assert ws.receive_text() == 'echo:{"type":"frame","seq":1}'
        ws.send_bytes(b"\xff\xd8\xff\xe0jpeg")
        assert ws.receive_bytes() == b"echo:\xff\xd8\xff\xe0jpeg"


def test_ws_proxy_preserves_order_across_frame_types(inference_ws):
    """Header-then-JPEG only means anything if the pair stays a pair."""
    with TestClient(create_app()).websocket_connect("/ws/ingest") as ws:
        for seq in range(3):
            ws.send_text(f'{{"seq":{seq}}}')
            ws.send_bytes(bytes([seq]) * 4)
        for seq in range(3):
            assert ws.receive_text() == f'echo:{{"seq":{seq}}}'
            assert ws.receive_bytes() == b"echo:" + bytes([seq]) * 4


def test_ws_proxy_closes_with_the_upstream_reason_when_the_node_is_down(monkeypatch):
    """Accept-then-close, not a failed handshake: the page gets a close code and the
    upstream's own error to display instead of a bare "connection error"."""
    monkeypatch.setenv("BR_INFERENCE_WS", "ws://127.0.0.1:1/viewer")
    with TestClient(create_app()).websocket_connect("/ws/ingest") as ws:
        message = ws.receive()
    assert message["type"] == "websocket.close"
    assert message["code"] == 1011
    assert "127.0.0.1:1" in message["reason"]


# --------------------------------------------------------------------- target URLs
def test_ws_target_is_the_origin_of_the_viewer_socket_env(monkeypatch):
    """One node serves both sockets, so BR_INFERENCE_WS names /viewer and the ingest
    socket is found beside it — only the origin is carried over."""
    monkeypatch.setenv("BR_INFERENCE_WS", "wss://jetson.lan:8443/viewer")
    assert proxy.ws_origin() == "wss://jetson.lan:8443"


def test_http_target_is_the_origin_of_the_inference_http_env(monkeypatch):
    monkeypatch.setenv("BR_INFERENCE_HTTP", "http://jetson.lan:8000/estimate")
    assert proxy.http_origin() == "http://jetson.lan:8000"


def test_targets_default_to_the_local_node(monkeypatch):
    monkeypatch.delenv("BR_INFERENCE_WS", raising=False)
    monkeypatch.delenv("BR_INFERENCE_HTTP", raising=False)
    assert proxy.ws_origin() == "ws://localhost:8000"
    assert proxy.http_origin() == "http://localhost:8000"


def test_proxy_imports_no_node_package():
    """The relay is network glue, not a participant: it must stay ignorant of core,
    edge and inference (CLAUDE.md, the separation rule)."""
    from pathlib import Path

    text = Path(proxy.__file__).read_text(encoding="utf-8")
    for forbidden in ("board_reader_core", "board_reader_edge", "board_reader_inference"):
        assert forbidden not in text


def test_websockets_client_is_a_declared_dependency():
    """It arrives transitively via uvicorn[standard] today; the relay uses it directly,
    so dropping that extra must break the lock rather than the camera page."""
    from pathlib import Path

    manifest = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    for dependency in ("httpx", "websockets", "cryptography"):
        assert f'"{dependency}>=' in manifest, dependency
