"""Inference gateway (DESIGN.md §4.5) — FastAPI WebSocket ingress + egress.

Runs on the inference node. Three surfaces:

* ``WS /ingest`` — the edge connects and streams frames. Each frame is two WS
  messages: a JSON header then the binary JPEG. The gateway decodes, runs the
  estimator, and publishes the resulting position event to the hub.
* ``WS /viewer`` — viewers connect and receive the position-event JSON stream
  (retained latest per board first, then live updates).
* ``GET /boards/{id}`` + ``GET /healthz`` — latest snapshot + liveness.

The gateway holds the shared estimator and the position hub. It never imports the
edge package; it only speaks the wire (:mod:`board_reader_core.wire`).
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from board_reader_core.wire import Frame, decode_jpeg

from board_reader_inference.estimator import Estimator, StubEstimator
from board_reader_inference.hub import PositionHub


def _default_estimator() -> Estimator:
    """Pick the estimator from the environment.

    ``BR_CENTROIDS=<path>`` -> the M1 :class:`CentroidEstimator` with calibrated
    centroids and the lightweight embedder (the DINOv2 embedder swaps in here later).
    Unset -> the M0 :class:`StubEstimator`. Resolved at import time so the uvicorn
    string-imported ``app`` honours it without a code change.
    """
    path = os.environ.get("BR_CENTROIDS")
    if not path:
        return StubEstimator()
    from board_reader_inference.centroid import CentroidClassifier
    from board_reader_inference.embed import make_embedder
    from board_reader_inference.estimator import CentroidEstimator

    classifier = CentroidClassifier.load(path)
    # Rebuild the same backbone the centroids were calibrated with (dims must match).
    return CentroidEstimator(make_embedder(classifier.backbone), classifier)


def create_app(
    estimator: Estimator | None = None,
    hub: PositionHub | None = None,
) -> FastAPI:
    """Build the gateway app. Estimator/hub are injectable for tests."""
    est: Estimator = estimator if estimator is not None else _default_estimator()
    hub_: PositionHub = hub if hub is not None else PositionHub()

    app = FastAPI(title="board-reader inference node")
    # The viewer is served from a different origin (its own container), and the
    # browser connects here over WS; allow it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("BR_ALLOW_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.estimator = est
    app.state.hub = hub_

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "subscribers": hub_.subscriber_count}

    @app.get("/boards/{board_id}")
    async def board_latest(board_id: str) -> JSONResponse:
        event = hub_.latest(board_id)
        if event is None:
            return JSONResponse({"error": "no such board"}, status_code=404)
        return JSONResponse(event.to_dict())

    @app.websocket("/ingest")
    async def ingest(ws: WebSocket) -> None:
        """Edge -> inference frame ingress. Header (text) then JPEG (binary) per frame."""
        await ws.accept()
        try:
            while True:
                header = json.loads(await ws.receive_text())
                jpeg = await ws.receive_bytes()
                frame = Frame.from_header(header, decode_jpeg(jpeg))
                event = est.estimate(frame)
                await hub_.publish(event)
                await ws.send_json({"ack": frame.seq, "board_id": frame.board_id})
        except WebSocketDisconnect:
            return

    @app.websocket("/viewer")
    async def viewer(ws: WebSocket) -> None:
        """Inference -> viewer position-event egress."""
        await ws.accept()
        try:
            async for event in hub_.subscribe():
                await ws.send_text(event.to_json())
        except WebSocketDisconnect:
            return

    return app


# Module-level app for ``uvicorn board_reader_inference.server:app``.
app = create_app()
