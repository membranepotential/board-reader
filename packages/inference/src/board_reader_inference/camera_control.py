"""Camera control on the inference node: per-board config, phone reports, previews.

The node holds, per ``board_id``, what the control UI chose for that board's camera
(:class:`~board_reader_core.control.CameraConfig`), the last thing the camera said
about itself (:class:`~board_reader_core.control.CameraReport`) and a low-rate
full-frame preview, so the control UI can show the camera's whole field of view
while the stream itself carries only the crop. Everything is in memory: a restart
forgets the config, and the control UI sets it again.

The edge **pulls**: it posts its report about twice a second and the reply is the
config. That one request is the whole channel — no socket to hold open while the
camera is idle, nothing for the node to track per connection, and a config change
reaches the phone within one poll.

Routes (all under ``/boards/{board_id}/camera``):

* ``GET``  — the control UI's view: config, last report and its age, preview and
  last streamed frame metadata, and the node's clock to measure ages against.
* ``PUT``  — the control UI replaces the config (the whole object, validated).
* ``POST /report`` — the edge's poll: body is its report, reply is the config.
* ``POST /preview`` — the edge's full-frame JPEG; ``GET /preview`` serves it back.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from board_reader_core.control import CameraConfig, CameraReport
from board_reader_core.wire import decode_jpeg

#: A preview is a small thumbnail; anything bigger is not one.
MAX_PREVIEW_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class Preview:
    jpeg: bytes
    seq: int
    at: float
    width: int
    height: int


@dataclass(slots=True)
class BoardCamera:
    config: CameraConfig = field(default_factory=CameraConfig)
    report: CameraReport | None = None
    seen_at: float | None = None
    preview: Preview | None = None


@dataclass(frozen=True, slots=True)
class StreamedFrame:
    """The last frame ``/ingest`` received for a board, as the control UI needs it."""

    seq: int
    received_at: float
    width: int
    height: int


class CameraRegistry:
    """Every board's camera state. A board exists from the first time anyone asks."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._boards: dict[str, BoardCamera] = {}
        self._clock = clock
        self._preview_seq = 0

    def board(self, board_id: str) -> BoardCamera:
        return self._boards.setdefault(board_id, BoardCamera())

    def configure(self, board_id: str, config: CameraConfig) -> None:
        self.board(board_id).config = config

    def report(self, board_id: str, report: CameraReport) -> CameraConfig:
        """Record the edge's report; answer with the config it should apply."""
        board = self.board(board_id)
        board.report = report
        board.seen_at = self._clock()
        return board.config

    def store_preview(self, board_id: str, jpeg: bytes, width: int, height: int) -> None:
        self._preview_seq += 1
        self.board(board_id).preview = Preview(
            jpeg=jpeg, seq=self._preview_seq, at=self._clock(), width=width, height=height
        )

    def view(self, board_id: str, streamed: StreamedFrame | None) -> dict[str, object]:
        board = self.board(board_id)
        preview = board.preview
        return {
            "board_id": board_id,
            "config": board.config.to_dict(),
            "report": None if board.report is None else board.report.to_dict(),
            "seen_at": board.seen_at,
            "preview": (
                None
                if preview is None
                else {
                    "seq": preview.seq,
                    "at": preview.at,
                    "width": preview.width,
                    "height": preview.height,
                }
            ),
            "frame": (
                None
                if streamed is None
                else {
                    "seq": streamed.seq,
                    "received_at": streamed.received_at,
                    "width": streamed.width,
                    "height": streamed.height,
                }
            ),
            # Ages are measured against the node's clock, never the browser's: the
            # phone, the laptop and the node need not agree on the time.
            "now": self._clock(),
        }


def _refusal(message: str, status: int = 422) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _json_body(request: Request) -> object:
    try:
        return json.loads(await request.body())
    except ValueError as e:
        raise ValueError(f"body is not JSON: {e}") from e


def router(
    registry: CameraRegistry,
    streamed: Callable[[str], StreamedFrame | None],
) -> APIRouter:
    """The control routes. ``streamed`` looks up the last ingested frame per board."""
    api = APIRouter()

    @api.get("/boards/{board_id}/camera")
    async def camera_state(board_id: str) -> JSONResponse:
        return JSONResponse(registry.view(board_id, streamed(board_id)))

    @api.put("/boards/{board_id}/camera")
    async def configure(board_id: str, request: Request) -> JSONResponse:
        try:
            config = CameraConfig.from_dict(await _json_body(request))
        except ValueError as e:
            return _refusal(str(e))
        registry.configure(board_id, config)
        return JSONResponse(registry.view(board_id, streamed(board_id)))

    @api.post("/boards/{board_id}/camera/report")
    async def report(board_id: str, request: Request) -> JSONResponse:
        try:
            parsed = CameraReport.from_dict(await _json_body(request))
        except ValueError as e:
            return _refusal(str(e))
        return JSONResponse(registry.report(board_id, parsed).to_dict())

    @api.post("/boards/{board_id}/camera/preview")
    async def put_preview(board_id: str, request: Request) -> Response:
        body = await request.body()
        if len(body) > MAX_PREVIEW_BYTES:
            return _refusal(f"preview larger than {MAX_PREVIEW_BYTES} bytes", 413)
        try:
            image = decode_jpeg(body)
        except ValueError:
            return _refusal("preview is not a JPEG", 400)
        height, width = image.shape[:2]
        registry.store_preview(board_id, body, width=width, height=height)
        return Response(status_code=204)

    @api.get("/boards/{board_id}/camera/preview")
    async def get_preview(board_id: str) -> Response:
        preview = registry.board(board_id).preview
        if preview is None:
            return _refusal("no preview for board", 404)
        return Response(
            content=preview.jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Preview-Seq": str(preview.seq)},
        )

    return api
