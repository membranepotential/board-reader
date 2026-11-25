"""The ``camera`` command's run loop: a stream on the control channel (D11, D18).

The edge then behaves as the phone's ``/camera`` page does, in two halves:

* **the camera thread** — samples the stream (:class:`~.stream.StreamSource`), cuts
  each frame to the configured crop, feeds the motion meter EVERY sampled frame and,
  while streaming, lets the adaptive forwarder (D4) pick the frames to send, which it
  shrinks to the phone's send size and encodes. It never blocks on the network.
* **the event loop** — the control socket (:class:`~.link.CameraLink`), the ``/ingest``
  socket (:class:`~board_reader_edge.client.IngestLink`, started and stopped with
  ``streaming``) and the previews.

They share :class:`Settings`, which the loop replaces whole on every ``assigned`` or
``config`` and the thread reads once per frame, so a frame is always processed under
one consistent config. The thread hands frames to the loop with
``call_soon_threadsafe``.

What a non-phone camera does with the config: ``crop`` and ``streaming`` apply;
``device`` and ``resolution`` do not (a stream has one picture at one size), so the
report lists the stream itself as its only lens and states the frame size it delivers.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from board_reader_core.control import (
    CameraConfig,
    CameraDevice,
    CameraReport,
    CropRect,
    FrameSize,
    IngestState,
)
from board_reader_core.wire import Frame, encode_jpeg

from board_reader_edge.adapter import DEFAULT_JPEG_QUALITY, crop_to
from board_reader_edge.camera.link import CameraLink, cameras_url
from board_reader_edge.camera.stream import Captured, CaptureSource
from board_reader_edge.client import Envelope, IngestLink
from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig

log = logging.getLogger(__name__)

#: The phone's send and preview sizes (``lib/components/camera/state.ts``), so both
#: edges cost the node and the uplink alike.
SEND_LONG_SIDE = 960
PREVIEW_LONG_SIDE = 480
PREVIEW_QUALITY = 60
PREVIEW_S = 1.5

#: The report's one "lens" id: a stream has no lenses to choose from.
STREAM_DEVICE_ID = "stream"

#: How a run ends: the video played to its end, or another edge took the device id.
Outcome = Literal["ended", "replaced"]


@dataclass(frozen=True)
class Settings:
    """What the camera thread works under; replaced whole, never mutated."""

    board_id: str | None = None
    config: CameraConfig = dataclasses.field(default_factory=CameraConfig)

    @property
    def streaming(self) -> bool:
        return self.config.streaming and self.board_id is not None


def fit_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    """``image`` shrunk so its longer side is at most ``long_side``; never enlarged."""
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    if scale >= 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class CameraPipeline:
    """Crop, meter and forward one sampled frame at a time (the phone's frame loop).

    Every frame names ``device_id``: the node takes a camera's board id only from
    that camera. ``seq`` counts every sampled frame and is never reset, so it is monotonic on the
    wire (gaps are the frames not forwarded) and the burst ids the forwarder makes
    from it never repeat.
    """

    def __init__(
        self,
        device_id: str,
        # Frozen, so a shared default cannot leak state between pipelines.
        config: ForwarderConfig = ForwarderConfig(),  # noqa: B008
        *,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> None:
        self.device_id = device_id
        self.config = config
        self.jpeg_quality = jpeg_quality
        self.forwarder = AdaptiveForwarder(config)
        self._seq = 0
        self._metered: tuple[CropRect | None, tuple[int, ...]] | None = None
        self._streaming = False

    def process(self, captured: Captured, settings: Settings) -> Frame | None:
        """The frame to send for ``captured``, or ``None`` (not streaming, or dropped)."""
        crop = settings.config.crop
        image = captured.image if crop is None else crop_to(captured.image, crop)
        if settings.streaming and not self._streaming:
            self.forwarder = AdaptiveForwarder(self.config)  # a fresh stream, as the phone
        self._streaming = settings.streaming
        # A new crop or a new frame size is a new picture, not motion in the old one.
        metered = (crop, captured.image.shape)
        if metered != self._metered:
            self.forwarder.meter.rebase()
            self._metered = metered
        seq = self._seq
        self._seq += 1
        if not settings.streaming or settings.board_id is None:
            self.forwarder.meter.update(image)  # the meter sees every frame regardless
            return None
        return self.forwarder.decide(
            Frame(
                board_id=settings.board_id,
                seq=seq,
                capture_ts=captured.capture_ts,
                image=image,
                device_id=self.device_id,
            )
        )

    def envelope(self, frame: Frame) -> Envelope:
        """``frame`` shrunk to the send size and encoded for ``/ingest``."""
        small = dataclasses.replace(frame, image=fit_long_side(frame.image, SEND_LONG_SIDE))
        return small.encode(self.jpeg_quality)


def preview_jpeg(image: np.ndarray) -> bytes:
    """The whole frame as the small preview ``/control`` draws the crop on."""
    return encode_jpeg(fit_long_side(image, PREVIEW_LONG_SIDE), PREVIEW_QUALITY)


class StreamCamera:
    """One stream as a camera on the node: the control link, the thread, the sockets."""

    def __init__(
        self,
        source: CaptureSource,
        *,
        node: str,
        device_id: str,
        title: str,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        preview_s: float = PREVIEW_S,
    ) -> None:
        self.source = source
        self.node = node.rstrip("/")
        self.title = title
        self.pipeline = CameraPipeline(device_id, jpeg_quality=jpeg_quality)
        self.preview_s = preview_s
        self.settings = Settings()
        #: The latest sampled whole frame, for the previews (set by the thread).
        self.latest: np.ndarray | None = None
        self._frame: FrameSize | None = None
        self._ingest_state: IngestState = "idle"
        self._ingest: IngestLink | None = None
        self._ingest_task: asyncio.Task[None] | None = None
        self._previews_on = asyncio.Event()
        self.link = CameraLink(
            cameras_url(self.node, device_id),
            on_assigned=self._assigned,
            on_config=self._configured,
            on_previews=self._previews,
        )

    # ---- the control channel (event loop) ------------------------------------

    def report(self) -> CameraReport:
        lens = CameraDevice(id=STREAM_DEVICE_ID, label=self.title)
        return CameraReport(
            devices=(lens,),
            device=self.title,
            frame=self._frame,
            ingest=self._ingest_state,
        )

    def _assigned(self, board_id: str) -> None:
        # Assigned once for good (D18); a rename only names it, so nothing here
        # needs to follow a change beyond taking the id the node says.
        if board_id != self.settings.board_id:
            log.info("streaming as board %s", board_id)
        self.settings = dataclasses.replace(self.settings, board_id=board_id)
        self._follow_streaming()

    def _configured(self, config: CameraConfig) -> None:
        if config != self.settings.config:
            log.info(
                "config: crop %s, streaming %s",
                config.crop.to_dict() if config.crop is not None else "whole frame",
                "on" if config.streaming else "off",
            )
        self.settings = dataclasses.replace(self.settings, config=config)
        self._follow_streaming()

    def _previews(self, on: bool) -> None:
        if on:
            self._previews_on.set()

    def _set_ingest_state(self, state: IngestState) -> None:
        self._ingest_state = state
        self.link.report(self.report())

    def _follow_streaming(self) -> None:
        """Open ``/ingest`` while the settings say stream, close it otherwise."""
        if self.settings.streaming and self._ingest_task is None:
            self._ingest = IngestLink(f"{self.node}/ingest", on_state=self._set_ingest_state)
            self._ingest_task = asyncio.get_running_loop().create_task(self._ingest.run())
        elif not self.settings.streaming and self._ingest_task is not None:
            self._ingest_task.cancel()
            self._ingest_task = None
            self._ingest = None

    def _send(self, envelope: Envelope) -> None:
        if self._ingest is not None:
            self._ingest.offer(envelope)

    def _frame_size(self, size: FrameSize) -> None:
        self._frame = size
        self.link.report(self.report())

    async def _post_previews(self) -> None:
        while True:
            latest = self.latest
            if self.link.previews_wanted and latest is not None:
                self.link.offer_preview(preview_jpeg(latest))
            try:
                async with asyncio.timeout(self.preview_s):
                    await self._previews_on.wait()
            except TimeoutError:
                pass
            self._previews_on.clear()

    # ---- the camera thread ----------------------------------------------------

    def _capture(
        self, loop: asyncio.AbstractEventLoop, ended: Callable[[BaseException | None], None]
    ) -> None:
        error: BaseException | None = None
        size: FrameSize | None = None
        try:
            for captured in self.source.captures():
                height, width = captured.image.shape[:2]
                if size is None or (size.width, size.height) != (width, height):
                    size = FrameSize(width=width, height=height)
                    loop.call_soon_threadsafe(self._frame_size, size)
                self.latest = captured.image
                frame = self.pipeline.process(captured, self.settings)
                if frame is not None:
                    loop.call_soon_threadsafe(self._send, self.pipeline.envelope(frame))
        except BaseException as e:  # handed to the loop, which re-raises it
            error = e
        finally:
            if not loop.is_closed():
                loop.call_soon_threadsafe(ended, error)

    # ---- the whole run ----------------------------------------------------------

    async def run(self) -> Outcome:
        """Run until the video ends or the node says another edge took over."""
        loop = asyncio.get_running_loop()
        ended: asyncio.Future[BaseException | None] = loop.create_future()

        def end(error: BaseException | None) -> None:
            if not ended.done():
                ended.set_result(error)

        self.link.report(self.report())
        thread = threading.Thread(
            target=self._capture, args=(loop, end), name="camera", daemon=True
        )
        thread.start()
        link = loop.create_task(self.link.run())
        previews = loop.create_task(self._post_previews())
        try:
            await asyncio.wait({link, ended}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.source.stop()
            for task in (link, previews, self._ingest_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(t for t in (link, previews, self._ingest_task) if t is not None),
                return_exceptions=True,
            )
        if ended.done() and (error := ended.result()) is not None:
            raise error
        return "ended" if ended.done() else "replaced"
