"""File-based edge adapter (DESIGN.md §6.1 step 1, WP-I1).

Pulls frames from a :class:`~board_reader_edge.source.FrameSource` and emits them as
wire envelopes ``(header, jpeg)`` for the inference gateway. This is the edge's
assembly point: a static crop and motion-driven forwarding (WP-C3) are both
optional stages here, and neither changes the wire shape.

**The edge is a dumb camera** (DESIGN D10, 2026-11-24): it never looks for the
board. What it sends is the part of the frame a static rectangle (``roi``, a
:class:`~board_reader_core.control.CropRect` in fractions of the frame) selects,
or the whole frame. Edge frames carry no ``corners``.

**Stage order:** the crop comes first, so the forwarder measures motion on the
board's rectangle rather than on the whole camera view — a hand moving outside
the rectangle must not open a burst.

:class:`RecordingReplay` is the second envelope source in this module: it re-sends
a recorded session instead of processing a live one. Both satisfy
:class:`EnvelopeSource`, which is all the WS client needs.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

from board_reader_core.control import CropRect
from board_reader_core.recording import Recording
from board_reader_core.wire import Frame

from board_reader_edge.forwarder import AdaptiveForwarder
from board_reader_edge.source import FrameSource

_T = TypeVar("_T")

# Default JPEG quality of the forwarded frames. Exported so the CLI's
# ``--jpeg-quality`` advertises the same number the adapter would have used.
DEFAULT_JPEG_QUALITY = 90


@runtime_checkable
class EnvelopeSource(Protocol):
    """Anything that can be streamed to the gateway's ``/ingest`` socket."""

    def envelopes(self) -> Iterator[tuple[dict, bytes]]:
        ...


def crop_to(image: np.ndarray, roi: CropRect) -> np.ndarray:
    """The part of ``image`` that ``roi`` selects, as its own contiguous array."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = roi.pixels(width, height)
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def pace_by_capture_ts(
    items: Iterable[tuple[float, _T]],
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Iterator[_T]:
    """Yield each item so the gaps between them match its own ``capture_ts`` gaps.

    Pacing replays capture-time *gaps* rather than a fixed period, because with
    adaptive forwarding there is no fixed period: the gap between two forwarded
    frames is 1 s at rest and 1/12 s inside a burst. Timestamps are taken relative
    to the first item, so a stream whose clock starts at 0 (a replayed clip) and
    one stamped with wall-clock epochs pace identically.
    """
    started = monotonic()
    first_ts: float | None = None
    for capture_ts, item in items:
        if first_ts is None:
            first_ts = capture_ts
        behind = started + max(capture_ts - first_ts, 0.0) - monotonic()
        if behind > 0:
            sleep(behind)
        yield item


class FileEdgeAdapter:
    """Adapt a frame source into a stream of wire envelopes.

    Parameters
    ----------
    source:
        The frame source to drain.
    roi:
        Optional static crop, in fractions of each source frame. When supplied,
        every frame is cut to it before anything else looks at the frame. ``None``
        forwards the whole frame.
    forwarder:
        Optional :class:`~board_reader_edge.forwarder.AdaptiveForwarder`. When
        supplied, every (cropped) frame is scored and only the keepalive/burst
        frames it selects go on (DESIGN D4); the forwarded frames carry
        ``stability`` and ``burst_id``. When ``None`` every frame is forwarded.
    jpeg_quality:
        JPEG quality of the encoded frame (1-100), which is what a real Pi's
        uplink pays for.
    throttle:
        If ``True``, pace the output to the stream's own **capture time** — a
        replayed clip then leaves the edge at the rate it was recorded at, and an
        adaptive feed keeps its keepalive/burst rhythm. Off by default so offline
        replay runs flat out.
    sleep / monotonic:
        Injectable for deterministic tests.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        roi: CropRect | None = None,
        forwarder: AdaptiveForwarder | None = None,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        throttle: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.roi = roi
        self.forwarder = forwarder
        self.jpeg_quality = jpeg_quality
        self.throttle = throttle
        self._sleep = sleep
        self._monotonic = monotonic

    def envelopes(self) -> Iterator[tuple[dict, bytes]]:
        """Yield ``(header, jpeg_bytes)`` per forwarded frame."""
        frames: Iterable[Frame] = self.frames()
        if self.throttle:
            frames = pace_by_capture_ts(
                ((f.capture_ts, f) for f in frames),
                sleep=self._sleep,
                monotonic=self._monotonic,
            )
        for frame in frames:
            yield frame.encode(self.jpeg_quality)

    def frames(self) -> Iterator[Frame]:
        """Yield the :class:`Frame` objects to forward, cropped to ``roi``.

        Frames the forwarder drops never appear, so consumers only ever see the
        rate D4 asks for.
        """
        for frame in self.source.frames():
            if self.roi is not None:
                frame = dataclasses.replace(frame, image=crop_to(frame.image, self.roi))
            if self.forwarder is not None:
                decided = self.forwarder.decide(frame)
                if decided is None:
                    continue  # below the current keepalive/burst rate
                frame = decided
            yield frame


class RecordingReplay:
    """Re-send a recorded session (WP-H2) to the gateway, byte for byte.

    This is the reproduce-a-run path: the frames a gateway once received are put
    back on the wire **verbatim** — the recorded header dict and the recorded JPEG
    bytes, untouched. Nothing is decoded and re-encoded on the way through, because
    a JPEG round trip perturbs pixels, and a recognition run fed perturbed pixels
    is not a reproduction of the original one.

    Deliberately *not* a :class:`~board_reader_edge.source.FrameSource` feeding
    :class:`FileEdgeAdapter`: a recording already holds the edge's output — cropping
    and motion gating have happened — so re-running those stages over it
    would produce a different stream and defeat the point. Use :meth:`frames` (or
    ``RecordingSource``) when reprocessing is what you actually want.

    Parameters
    ----------
    recording:
        The session to replay, or a path — a session directory, or a root
        directory whose newest session is taken.
    board_id:
        Replay only this board. Default replays every board, interleaved exactly
        as they arrived.
    throttle:
        Pace output to the recording's own capture time, i.e. re-run it at the
        speed it happened. Worth preferring for a faithful re-run: the node reads
        the same frames either way, but a flat-out replay can push its read queue
        into the overload rule, which drops keepalive reads the original run read.
    """

    def __init__(
        self,
        recording: Recording | str,
        *,
        board_id: str | None = None,
        throttle: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.recording = (
            recording if isinstance(recording, Recording) else Recording.open(recording)
        )
        self.board_id = board_id
        self.throttle = throttle
        self._sleep = sleep
        self._monotonic = monotonic

    def envelopes(self) -> Iterator[tuple[dict, bytes]]:
        """Yield the recorded ``(header, jpeg_bytes)`` pairs, unmodified."""
        pairs = (
            (rec.capture_ts, (rec.header, rec.jpeg))
            for rec in self.recording.frames(board_id=self.board_id)
        )
        if self.throttle:
            yield from pace_by_capture_ts(
                pairs, sleep=self._sleep, monotonic=self._monotonic
            )
            return
        for _, envelope in pairs:
            yield envelope

    def frames(self) -> Iterator[Frame]:
        """The same stream decoded to :class:`Frame` objects — for offline reruns.

        Lets a recording stand in as a :class:`~board_reader_edge.source.FrameSource`
        when the point *is* to reprocess it (a new motion threshold) rather than
        to reproduce it.
        """
        for rec in self.recording.frames(board_id=self.board_id):
            yield rec.frame()
