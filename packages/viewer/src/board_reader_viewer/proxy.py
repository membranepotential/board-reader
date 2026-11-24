"""Same-origin relay from the viewer to the inference node.

The phone camera page (``/camera``) needs a *secure* context for ``getUserMedia``
and a *same-origin* socket to stream frames into. Those two pull in the same
direction: if the page fetched the inference node directly, the operator would
have to accept a second self-signed certificate on a second origin, from a phone,
before anything worked at all. So the viewer speaks for the inference node —

    browser ──► viewer ``/ws/ingest``  ──► inference ``/ingest``
    browser ◄── viewer ``/ws/viewer``  ◄── inference ``/viewer``
    browser ──► viewer ``/api/<any>``  ──► inference ``/<any>``

— and the phone only ever knows one origin, one certificate.

**All three pages** relay, not just ``/camera``. The live page and ``/snapshot``
used to be handed an absolute ``http://<lan-ip>:8000`` origin injected into their
HTML, which made them work on a LAN and nowhere else: served over https they were
blocked mixed content, and behind a tunnel the address named a host the browser
could not reach. One origin for everything costs one prefix here and deletes that
whole failure class — see ``.claude/skills/tunnel``.

This is **pure network glue**: nothing here parses a frame header, decodes a JPEG
or knows what a board is. The viewer still imports no ``board_reader_*`` package
(CLAUDE.md "the separation rule"); it forwards bytes and status codes verbatim and
has no opinion about them.

Targets come from the same environment the viewer already reads: the *origin* of
``BR_INFERENCE_WS`` for sockets, ``BR_INFERENCE_HTTP`` for requests.
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

DEFAULT_WS = "ws://localhost:8000/viewer"
DEFAULT_HTTP = "http://localhost:8000"

# Calibration on the inference node runs a backbone over 64 squares and rebuilds the
# exemplar bank; on a cold CUDA session that is comfortably tens of seconds. The relay
# must outlast it, or the phone sees a timeout for a calibration that in fact succeeded.
_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=60.0, pool=5.0)

# Close codes the browser sees when the *upstream* is the problem, so a page that fails
# to stream can say whose fault it is rather than "connection closed".
_UPSTREAM_UNREACHABLE = 1011


def ws_origin() -> str:
    """Scheme + authority of the inference socket, from ``BR_INFERENCE_WS``.

    The env var names the full ``/viewer`` URL (that is what the live page is handed);
    the ingest socket lives beside it on the same node, so only the origin is ours to
    keep.
    """
    parts = urlsplit(os.environ.get("BR_INFERENCE_WS", DEFAULT_WS))
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def http_origin() -> str:
    """Scheme + authority of the inference HTTP API, from ``BR_INFERENCE_HTTP``."""
    parts = urlsplit(os.environ.get("BR_INFERENCE_HTTP", DEFAULT_HTTP))
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


async def _client_to_upstream(
    ws: WebSocket, upstream: websockets.ClientConnection
) -> None:
    """Pump browser -> inference, preserving text/binary framing.

    The ingest protocol is a JSON header (text) followed by its JPEG (binary) per
    frame, so collapsing the two frame types — or reordering them — would corrupt
    the stream. Starlette hands us whichever arrived; we send back the same kind.
    """
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        text = message.get("text")
        if text is not None:
            await upstream.send(text)
            continue
        data = message.get("bytes")
        if data is not None:
            await upstream.send(data)


async def _upstream_to_client(
    upstream: websockets.ClientConnection, ws: WebSocket
) -> None:
    """Pump inference -> browser, preserving text/binary framing."""
    async for message in upstream:
        if isinstance(message, str):
            await ws.send_text(message)
        else:
            await ws.send_bytes(message)


async def relay_websocket(ws: WebSocket, path: str) -> None:
    """Bridge one accepted browser socket to ``path`` on the inference node.

    Accept first, then dial: a browser that is already ``open`` receives a close
    *code and reason* when the node is down, which is a diagnosable failure. Failing
    the handshake instead would surface as a bare "connection error" with nothing to
    read.
    """
    await ws.accept()
    target = ws_origin() + path
    try:
        # max_size=None: we are a pipe, not a parser — the node's own limits decide
        # what is too large, and a viewer-side cap would silently truncate frames.
        upstream = await websockets.connect(target, max_size=None, open_timeout=5.0)
    except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
        await ws.close(code=_UPSTREAM_UNREACHABLE, reason=f"{target}: {exc}"[:120])
        return

    async with upstream:
        both = [
            asyncio.create_task(_client_to_upstream(ws, upstream)),
            asyncio.create_task(_upstream_to_client(upstream, ws)),
        ]
        try:
            # Either end hanging up ends the relay; the survivor is cancelled rather
            # than left pumping into a closed socket.
            done, pending = await asyncio.wait(
                both, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
        except (WebSocketDisconnect, websockets.WebSocketException):
            pass
        finally:
            for task in both:
                task.cancel()


# Headers that describe *this* hop's framing rather than the payload. Copying them
# onto our own response would contradict the body we are about to send — Starlette
# recomputes content-length, and a content-encoding we already decoded would tell the
# browser to inflate plain bytes a second time.
_HOP_BY_HOP = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
)


async def relay_request(request: Request, path: str) -> Response:
    """Forward one request to ``path`` on the inference node, verbatim both ways.

    Method, query string, body and content type go up untouched; status, body and
    headers come back untouched, so the node's own refusal wording is what the operator
    reads.

    Response headers are forwarded rather than just the content type, because the node
    states things there (``X-Seq`` on ``/boards/{id}/frame``, ``X-Preview-Seq`` on the
    camera preview), and relaying only the body would drop them silently.
    """
    query = request.url.query
    target = http_origin() + path + (f"?{query}" if query else "")
    body = await request.body()
    headers = {}
    content_type = request.headers.get("content-type")
    if content_type is not None:
        headers["content-type"] = content_type

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.request(
                request.method, target, content=body, headers=headers
            )
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"error": f"inference node unreachable at {http_origin()}: {exc}"},
            status_code=502,
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
        headers={
            k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP
        },
    )


def install(app: FastAPI) -> None:
    """Mount the relay routes on the viewer app."""

    @app.websocket("/ws/ingest")
    async def ws_ingest(ws: WebSocket) -> None:
        """Camera page -> inference frame ingress."""
        await relay_websocket(ws, "/ingest")

    @app.websocket("/ws/viewer")
    async def ws_viewer(ws: WebSocket) -> None:
        """Inference -> browser position events."""
        await relay_websocket(ws, "/viewer")

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT"])
    async def api(request: Request, path: str) -> Response:
        """Everything the pages ask of the inference node, under one prefix.

        Deliberately a catch-all rather than a route per endpoint. The pages between
        them reach ``/estimate``, ``/uploads``, ``/uploads/{id}/photo``,
        ``/labels/{id}``, ``/boards/{id}/frame``, ``/boards/{id}/camera/*`` and
        ``/healthz``, and that set grows with the node's API. Enumerating it here would add a second place to
        edit for every new endpoint and a class of bug — a page calling something the
        relay does not know about — where this has none. The viewer still parses
        nothing; it is a pipe with a prefix.
        """
        return await relay_request(request, "/" + path)
