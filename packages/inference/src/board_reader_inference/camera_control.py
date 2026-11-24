"""Camera control on the inference node: per-board config, phone reports, previews.

The node holds, per ``board_id``, what the control UI chose for that board's camera
(:class:`~board_reader_core.control.CameraConfig`), the last thing the camera said
about itself (:class:`~board_reader_core.control.CameraReport`) and the latest
full-frame preview, so the control UI can show the camera's whole field of view
while the stream itself carries only the crop. Everything is in memory: a restart
forgets the config, and the control UI sets it again.

The registry is the single source of truth, and it **pushes**: every change is
posted at once to every socket that cares, over two WebSockets per board (the
message types and their directions are :mod:`board_reader_core.control`'s):

* ``WS /boards/{board_id}/camera`` — the phone. It hears its config and whether to
  send previews; it says what the camera is and sends previews.
* ``WS /boards/{board_id}/control`` — each open control page. It hears the state,
  the previews, the latest received frame and a health tick; it sends configs.

No polling, so no HTTP request per message and nothing to log per message: a
socket costs one access-log line when it opens.

Each socket gets a :class:`Mailbox` that keeps only the **latest message of each
type**. Every message here is a snapshot (a config, a state, a picture, a health
reading), so a slow client that misses an intermediate one loses nothing it needs,
and no client can make the node buffer without bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from board_reader_core.control import (
    CameraConfig,
    CameraReport,
    ConfigMessage,
    ControlDown,
    FrameMessage,
    HealthMessage,
    PhoneDown,
    PreviewMessage,
    PreviewsMessage,
    RefusedMessage,
    StateMessage,
    parse_control_message,
    parse_phone_message,
)
from board_reader_core.wire import decode_jpeg

log = logging.getLogger(__name__)

#: A preview is a small thumbnail; anything bigger is not one.
MAX_PREVIEW_BYTES = 1_000_000

#: How often a control page hears the health tick (and the latest received frame).
HEALTH_EVERY_S = 1.0


class _Message(Protocol):
    TYPE: ClassVar[str]

    def to_json(self) -> str: ...


class Mailbox[M: _Message]:
    """The messages one socket has yet to send: the latest of each type.

    Posting a type again replaces the pending one and moves it to the back, so
    messages leave in the order they were last posted.
    """

    def __init__(self) -> None:
        self._pending: dict[str, M] = {}
        self._ready = asyncio.Event()

    def post(self, message: M) -> None:
        self._pending.pop(message.TYPE, None)
        self._pending[message.TYPE] = message
        self._ready.set()

    def take_nowait(self) -> list[M]:
        taken = list(self._pending.values())
        self._pending.clear()
        self._ready.clear()
        return taken

    async def take(self) -> list[M]:
        await self._ready.wait()
        return self.take_nowait()


@dataclass(slots=True)
class BoardCamera:
    config: CameraConfig = field(default_factory=CameraConfig)
    report: CameraReport | None = None
    preview: PreviewMessage | None = None
    phones: set[Mailbox[PhoneDown]] = field(default_factory=set)
    watchers: set[Mailbox[ControlDown]] = field(default_factory=set)

    def state(self) -> StateMessage:
        return StateMessage(
            config=self.config, report=self.report, phone_connected=bool(self.phones)
        )


class CameraRegistry:
    """Every board's camera state and the sockets it is pushed to.

    A board exists from the first time anyone asks. Methods are synchronous and run
    on the event loop; they only post to mailboxes, and each socket's own task sends.
    """

    def __init__(self) -> None:
        self._boards: dict[str, BoardCamera] = {}

    def board(self, board_id: str) -> BoardCamera:
        return self._boards.setdefault(board_id, BoardCamera())

    def _tell_watchers(self, board: BoardCamera, message: ControlDown) -> None:
        for watcher in board.watchers:
            watcher.post(message)

    def _tell_phones(self, board: BoardCamera, message: PhoneDown) -> None:
        for phone in board.phones:
            phone.post(message)

    # ---- the phone ---------------------------------------------------------------

    def connect_phone(self, board_id: str) -> Mailbox[PhoneDown]:
        board = self.board(board_id)
        mailbox: Mailbox[PhoneDown] = Mailbox()
        board.phones.add(mailbox)
        mailbox.post(ConfigMessage(board.config))
        mailbox.post(PreviewsMessage(on=bool(board.watchers)))
        self._tell_watchers(board, board.state())
        return mailbox

    def disconnect_phone(self, board_id: str, mailbox: Mailbox[PhoneDown]) -> None:
        board = self.board(board_id)
        board.phones.discard(mailbox)
        self._tell_watchers(board, board.state())

    def report(self, board_id: str, report: CameraReport) -> None:
        board = self.board(board_id)
        board.report = report
        self._tell_watchers(board, board.state())

    def preview(self, board_id: str, preview: PreviewMessage) -> None:
        board = self.board(board_id)
        board.preview = preview
        self._tell_watchers(board, preview)

    # ---- the control pages -------------------------------------------------------

    def connect_watcher(self, board_id: str) -> Mailbox[ControlDown]:
        board = self.board(board_id)
        mailbox: Mailbox[ControlDown] = Mailbox()
        board.watchers.add(mailbox)
        mailbox.post(board.state())
        if board.preview is not None:
            mailbox.post(board.preview)
        if len(board.watchers) == 1:
            self._tell_phones(board, PreviewsMessage(on=True))
        return mailbox

    def disconnect_watcher(self, board_id: str, mailbox: Mailbox[ControlDown]) -> None:
        board = self.board(board_id)
        board.watchers.discard(mailbox)
        if not board.watchers:
            self._tell_phones(board, PreviewsMessage(on=False))

    def configure(self, board_id: str, config: CameraConfig) -> None:
        board = self.board(board_id)
        board.config = config
        self._tell_phones(board, ConfigMessage(config))
        self._tell_watchers(board, board.state())


# ---- what the node says about itself on the health tick ----------------------------


@dataclass(frozen=True, slots=True)
class StreamedFrame:
    """The last frame ``/ingest`` received for a board, as the control UI needs it."""

    seq: int
    #: Node clock, seconds.
    received_at: float
    width: int
    height: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class NodeStatus:
    """The node-side facts a health tick reports, for one board."""

    recognizer: str
    fallback: str | None
    received: int
    refused: int
    frame: StreamedFrame | None


@dataclass(frozen=True, slots=True)
class StreamSample:
    """One reading of the stream's position, for its rate between two ticks."""

    seq: int
    at: float


def frame_rate(previous: StreamSample | None, current: StreamSample | None) -> float | None:
    """Frames per second between two samples, from the frame sequence number.

    ``None`` when there is nothing to measure: no earlier sample, no time between
    them, or a sequence that went backwards (the phone reloaded and began at 0).
    """
    if previous is None or current is None:
        return None
    dt = current.at - previous.at
    frames = current.seq - previous.seq
    return frames / dt if dt > 0 and frames >= 0 else None


# ---- the sockets -------------------------------------------------------------------


async def _deliver[M: _Message](ws: WebSocket, mailbox: Mailbox[M]) -> None:
    while True:
        for message in await mailbox.take():
            await ws.send_text(message.to_json())


class _HungUp(Exception):
    """One of a socket's loops ended: the client hung up (or a send failed)."""


async def _ending(loop: Coroutine[Any, Any, None]) -> None:
    with contextlib.suppress(WebSocketDisconnect):
        await loop
    raise _HungUp


async def _until_first_ends(*loops: Coroutine[Any, Any, None]) -> None:
    """Run a socket's loops together; the first to end (a hang-up) ends them all.

    A ``TaskGroup``, not ``asyncio.wait`` plus a hand-rolled cancel: the server (or
    Starlette's test client) may cancel the handler from outside at any moment, and
    only the task group keeps that cancellation intact while it stops the loops.
    """
    try:
        async with asyncio.TaskGroup() as group:
            for loop in loops:
                group.create_task(_ending(loop))
    except* _HungUp:
        pass


class _Refusals:
    """Refuse one message on one socket: tell the client, log each reason once."""

    def __init__(self, socket: str, mailbox: Mailbox[Any]) -> None:
        self._socket = socket
        self._mailbox = mailbox
        self._logged: set[str] = set()

    def __call__(self, reason: str) -> None:
        self._mailbox.post(RefusedMessage(reason))
        if reason not in self._logged:
            self._logged.add(reason)
            log.warning("refused a message on %s: %s", self._socket, reason)


def _parse_preview(jpeg: bytes) -> PreviewMessage:
    if len(jpeg) > MAX_PREVIEW_BYTES:
        raise ValueError(f"preview larger than {MAX_PREVIEW_BYTES} bytes")
    try:
        image = decode_jpeg(jpeg)
    except ValueError as e:
        raise ValueError("preview is not a JPEG") from e
    height, width = image.shape[:2]
    return PreviewMessage(jpeg=jpeg, width=width, height=height)


def router(
    registry: CameraRegistry,
    status: Callable[[str], NodeStatus],
    clock: Callable[[], float] = time.time,
) -> APIRouter:
    """The two control sockets. ``status`` reads the node's own facts per board."""
    api = APIRouter()

    @api.websocket("/boards/{board_id}/camera")
    async def phone(ws: WebSocket, board_id: str) -> None:
        await ws.accept()
        mailbox = registry.connect_phone(board_id)
        refuse = _Refusals(f"/boards/{board_id}/camera", mailbox)

        async def receive() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                try:
                    jpeg = message.get("bytes")
                    if jpeg is not None:
                        registry.preview(board_id, _parse_preview(jpeg))
                    else:
                        report = parse_phone_message(message.get("text") or "").report
                        registry.report(board_id, report)
                except ValueError as e:
                    refuse(str(e))

        try:
            await _until_first_ends(receive(), _deliver(ws, mailbox))
        finally:
            registry.disconnect_phone(board_id, mailbox)

    @api.websocket("/boards/{board_id}/control")
    async def control(ws: WebSocket, board_id: str) -> None:
        await ws.accept()
        mailbox = registry.connect_watcher(board_id)
        refuse = _Refusals(f"/boards/{board_id}/control", mailbox)

        async def receive() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None:
                    refuse("the control socket takes JSON text messages only")
                    continue
                try:
                    registry.configure(board_id, parse_control_message(text).config)
                except ValueError as e:
                    refuse(str(e))

        async def tick() -> None:
            sample: StreamSample | None = None
            shown: float | None = None  # received_at of the last frame sent
            while True:
                node = status(board_id)
                now = clock()
                frame = node.frame
                current = None if frame is None else StreamSample(seq=frame.seq, at=now)
                mailbox.post(
                    HealthMessage(
                        recognizer=node.recognizer,
                        fallback=node.fallback,
                        received=node.received,
                        refused=node.refused,
                        frame_age_s=None if frame is None else max(0.0, now - frame.received_at),
                        fps=frame_rate(sample, current),
                    )
                )
                sample = current
                if frame is not None and frame.received_at != shown:
                    shown = frame.received_at
                    mailbox.post(
                        FrameMessage(
                            seq=frame.seq, jpeg=frame.jpeg, width=frame.width, height=frame.height
                        )
                    )
                await asyncio.sleep(HEALTH_EVERY_S)

        try:
            await _until_first_ends(receive(), _deliver(ws, mailbox), tick())
        finally:
            registry.disconnect_watcher(board_id, mailbox)

    return api
