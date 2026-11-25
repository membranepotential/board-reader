"""The control sockets: the phone's, the control UI's camera list, each board's control page.

The registry is the single source of truth, and it **pushes**: every change is posted
at once to every socket that cares (the message types and their directions are
:mod:`board_reader_core.control`'s):

* ``WS /cameras/{device_id}`` — the phone. It hears its board id, its config and
  whether to send previews; it says what the camera is and sends previews and the
  full frames Auto-frame asks for. One socket per device: a newer one replaces the
  older, which hears ``replaced``.
* ``WS /cameras`` — the control UI's list of every known camera; it sends renames
  and forgets.
* ``WS /boards/{board_id}/control`` — each open control page. It hears the state,
  the previews, the latest received frame and a health tick; it sends configs.

No polling, so no HTTP request per message and nothing to log per message: a
socket costs one access-log line when it opens.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable

from fastapi import APIRouter, WebSocket

from board_reader_core.control import (
    CamerasDown,
    ConfigureMessage,
    ForgetMessage,
    FrameMessage,
    FullFrameMessage,
    HealthMessage,
    ReframeMessage,
    RefusedMessage,
    ReportMessage,
    parse_cameras_message,
    parse_control_message,
    parse_phone_message,
)
from board_reader_core.control import (
    device_id as checked_device_id,
)

from board_reader_inference.control.camera.health import (
    NodeStatus,
    StreamSample,
    frame_rate,
)
from board_reader_inference.control.camera.mailbox import Mailbox
from board_reader_inference.control.camera.registry import CameraRegistry
from board_reader_inference.control.camera.sockets import (
    Refusals,
    deliver,
    parse_preview,
    until_first_ends,
)

#: How often a control page hears the health tick (and the latest received frame).
HEALTH_EVERY_S = 1.0


def router(
    registry: CameraRegistry,
    status: Callable[[str], NodeStatus],
    clock: Callable[[], float] = time.time,
) -> APIRouter:
    """The control sockets. ``status`` reads the node's own facts per board."""
    api = APIRouter()

    @api.websocket("/cameras/{device_id}")
    async def phone(ws: WebSocket, device_id: str) -> None:
        await ws.accept()
        try:
            checked_device_id(device_id)
        except ValueError as e:
            await ws.send_text(RefusedMessage(str(e)).to_json())
            await ws.close(code=1008)
            return
        link = registry.connect_phone(device_id)
        refuse = Refusals(f"/cameras/{device_id[:8]}", link.mailbox)

        async def receive() -> None:
            # A full frame's header, waiting for its JPEG (the next binary message).
            header: FullFrameMessage | None = None
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                try:
                    jpeg = message.get("bytes")
                    if jpeg is not None:
                        answered, header = header, None
                        if answered is None:
                            registry.preview(link, parse_preview(jpeg))
                        else:
                            registry.full_frame(link, answered, jpeg)
                    else:
                        header = None
                        match parse_phone_message(message.get("text") or ""):
                            case ReportMessage(report=report):
                                registry.report(link, report)
                            case FullFrameMessage() as answer:
                                header = answer
                except ValueError as e:
                    refuse(str(e))

        try:
            await until_first_ends(receive(), deliver(ws, link.mailbox))
        finally:
            registry.disconnect_phone(link)

    @api.websocket("/boards/{board_id}/camera")
    async def outdated_phone(ws: WebSocket, board_id: str) -> None:
        """A /camera page from before the phones had device ids: tell it to reload."""
        await ws.accept()
        await ws.send_text(RefusedMessage("this /camera page is out of date: reload it").to_json())
        await ws.close(code=1008)

    @api.websocket("/cameras")
    async def camera_list(ws: WebSocket) -> None:
        await ws.accept()
        changed = registry.connect_list()
        mailbox: Mailbox[CamerasDown] = Mailbox()
        refuse = Refusals("/cameras", mailbox)

        def frame_age(board_id: str) -> float | None:
            frame = status(board_id).frame
            return None if frame is None else max(0.0, clock() - frame.received_at)

        async def receive() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                try:
                    asked = parse_cameras_message(message.get("text") or "")
                    if isinstance(asked, ForgetMessage):
                        registry.forget(asked.device_id)
                    else:
                        registry.rename(asked.device_id, asked.name)
                except ValueError as e:
                    refuse(str(e))

        async def push() -> None:
            """The list on connect, on every change, and once a second for the ages."""
            while True:
                changed.clear()
                mailbox.post(registry.cameras(frame_age))
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(HEALTH_EVERY_S):
                        await changed.wait()

        try:
            await until_first_ends(receive(), deliver(ws, mailbox), push())
        finally:
            registry.disconnect_list(changed)

    @api.websocket("/boards/{board_id}/control")
    async def control(ws: WebSocket, board_id: str) -> None:
        await ws.accept()
        mailbox = registry.connect_watcher(board_id)
        refuse = Refusals(f"/boards/{board_id}/control", mailbox)

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
                    match parse_control_message(text):
                        case ConfigureMessage(config=config):
                            registry.configure(board_id, config)
                        case ReframeMessage():
                            registry.reframe(board_id)
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
                        direction=registry.direction(board_id),
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
            await until_first_ends(receive(), deliver(ws, mailbox), tick())
        finally:
            registry.disconnect_watcher(board_id, mailbox)

    return api
