"""The Python edge's end of the camera control channel (DESIGN D11, D18).

The same socket the phone speaks (``packages/viewer/src/lib/edge/cameralink.ts``):
``WS /cameras/{device_id}`` on the inference node. The node says which board id this
camera streams as (``assigned``: assigned once, it never changes; a rename on
``/control`` only gives it a name), pushes the config the control UI set (``config``)
and says whether a control page watches (``previews``). The edge sends its
:class:`~board_reader_core.control.CameraReport` on connect and whenever it changes,
and a whole-frame preview JPEG (binary) only while previews are wanted.

The **device id** is what makes the node give this camera the same board id after a
restart, and what lets ``/ingest`` take frames under that board id: every frame names
it (``device_id`` in the header), and the node refuses a camera's board id to any
other sender. The phone keeps it in ``localStorage``; the Python edge keeps it in a
small state file (:func:`load_device_id`), one file per camera.

A lost link changes nothing the camera is doing: it keeps the last config it heard,
and the node pushes it again when the link comes back. ``replaced`` (another process
with this device id took over) ends :meth:`CameraLink.run` for good, as on the phone,
because two edges that both reconnect would take the camera from each other forever.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Callable
from pathlib import Path

import websockets

from board_reader_core.control import (
    CameraConfig,
    CameraReport,
    PhoneDown,
    ReportMessage,
    device_id,
)
from board_reader_core.wiremodel import parse

log = logging.getLogger(__name__)

#: The state file's name under ``$XDG_STATE_HOME/board-reader/``.
DEVICE_ID_FILE = "edge-device-id"


def default_state_path() -> Path:
    """Where the device id is kept unless ``--state`` says otherwise."""
    root = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(root) / "board-reader" / DEVICE_ID_FILE


def load_device_id(path: Path, new_id: Callable[[], str] = lambda: str(uuid.uuid4())) -> str:
    """The device id kept at ``path``; a new one, kept from now on, if there is none.

    A file that holds no valid device id is replaced, as the phone replaces a bad
    ``localStorage`` entry: the camera then registers as a new one rather than not at
    all. The write is tmp + rename, so a crash never leaves half an id behind.
    """
    try:
        return device_id(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        pass
    made = device_id(new_id())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(made + "\n")
    os.replace(tmp, path)
    return made


def cameras_url(node: str, device: str) -> str:
    """The node's ``/cameras/{device_id}`` socket."""
    return f"{node.rstrip('/')}/cameras/{device}"


class CameraLink:
    """One camera's control socket, reconnecting until the node says ``replaced``.

    Callbacks run on the event loop. ``report`` and ``offer_preview`` may be called
    at any time; one writer per connection sends them, so a preview still being
    written is replaced by the next rather than queued behind it.
    """

    def __init__(
        self,
        url: str,
        *,
        on_assigned: Callable[[str], None],
        on_config: Callable[[CameraConfig], None],
        on_previews: Callable[[bool], None],
        retry_s: float = 1.0,
        max_retry_s: float = 15.0,
    ) -> None:
        self.url = url
        self._on_assigned = on_assigned
        self._on_config = on_config
        self._on_previews = on_previews
        self._retry_s = retry_s
        self._max_retry_s = max_retry_s
        self._report: CameraReport | None = None
        #: The report text last sent on the CURRENT socket; reset on every open.
        self._sent: str | None = None
        self._preview: bytes | None = None
        self._previews = False
        self._wake = asyncio.Event()

    @property
    def previews_wanted(self) -> bool:
        """Whether a control page is watching, so a preview is worth encoding."""
        return self._previews

    def report(self, report: CameraReport) -> None:
        """What the camera is now; sent at once if it changed, else on the next open."""
        self._report = report
        self._wake.set()

    def offer_preview(self, jpeg: bytes) -> None:
        """A whole-frame preview, sent while wanted; a newer one replaces an unsent one."""
        if self._previews:
            self._preview = jpeg
            self._wake.set()

    async def run(self) -> None:
        """Hold the link; return only when another socket of this device took over."""
        delay = self._retry_s
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    log.info("control link up: %s", self.url)
                    delay = self._retry_s
                    self._sent = None
                    self._wake.set()  # the report goes out on every new socket
                    if await self._serve(ws):
                        log.error("another edge with this device id took over; stopping")
                        return
                    reason = "closed by the node"
            except (OSError, websockets.WebSocketException) as e:
                reason = str(e) or type(e).__name__
            finally:
                self._set_previews(False)  # a fresh socket tells again
            log.warning("control link lost (%s); retrying in %.0f s", reason, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_retry_s)

    async def _serve(self, ws: websockets.ClientConnection) -> bool:
        """Read and write one socket until it closes; ``True`` when replaced."""
        writer = asyncio.create_task(self._write(ws))
        try:
            async for message in ws:
                if isinstance(message, str) and self._handle(message):
                    return True
            return False
        finally:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)

    def _handle(self, text: str) -> bool:
        """Apply one message from the node; ``True`` for ``replaced``."""
        try:
            message = parse(PhoneDown, text)
        except ValueError:
            return False  # a newer node's message; not ours to mind
        match message.type:
            case "assigned":
                self._on_assigned(message.board_id)
            case "config":
                self._on_config(message.config)
            case "previews":
                self._set_previews(message.on)
            case "replaced":
                return True
            case "refused":
                log.warning("the node refused a message: %s", message.reason)
        return False

    async def _write(self, ws: websockets.ClientConnection) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            if self._report is not None:
                text = ReportMessage(self._report).to_json()
                if text != self._sent:
                    await ws.send(text)
                    self._sent = text
            preview, self._preview = self._preview, None
            if preview is not None and self._previews:
                await ws.send(preview)

    def _set_previews(self, on: bool) -> None:
        if on == self._previews:
            return
        self._previews = on
        if not on:
            self._preview = None
        self._on_previews(on)
