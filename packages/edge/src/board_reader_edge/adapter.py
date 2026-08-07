"""File-based edge adapter (DESIGN.md §6.1 step 1, WP-I1).

Pulls frames from a :class:`~board_reader_edge.source.FrameSource` and emits them as
wire envelopes ``(header, jpeg)`` for the inference gateway. This is the edge's
assembly point: motion-driven forwarding (WP-C3) and board rectification (WP-A)
are both optional stages here, and neither changes the wire shape.

**Stage order is cost-driven** (M2): the forwarder decides on the **raw** frame —
it is one 32x32 resize, and the motion meter needs the full camera view *and* the
undropped stream to difference against — and only the frames it forwards are then
rectified. Detection is the expensive step (~90-100 ms), so a dropped frame must
never pay for it.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator

import dataclasses

from board_reader_core.wire import Frame

from board_reader_edge.forwarder import AdaptiveForwarder
from board_reader_edge.rectify import Rectifier
from board_reader_edge.source import FrameSource


class FileEdgeAdapter:
    """Adapt a frame source into a stream of wire envelopes.

    Parameters
    ----------
    source:
        The frame source to drain.
    forwarder:
        Optional :class:`~board_reader_edge.forwarder.AdaptiveForwarder`. When
        supplied, every source frame is scored and only the keepalive/burst frames
        it selects go on (DESIGN D4); the forwarded frames carry ``stability`` and
        ``burst_id``. When ``None`` every frame is forwarded (the M0/M1 default).
    rectifier:
        Optional :class:`~board_reader_edge.rectify.Rectifier`. When supplied, each
        forwarded frame is rectified to a canonical top-down crop (WP-A); frames the
        rectifier suppresses (recalibrating / needs_setup, DESIGN §4.4.1) are
        **dropped**, not forwarded. The frame's ``capture_ts`` drives the
        rectifier's re-detection cadence.
    jpeg_quality:
        JPEG quality for the encoded crop.
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
        forwarder: AdaptiveForwarder | None = None,
        rectifier: Rectifier | None = None,
        jpeg_quality: int = 90,
        throttle: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.forwarder = forwarder
        self.rectifier = rectifier
        self.jpeg_quality = jpeg_quality
        self.throttle = throttle
        self._sleep = sleep
        self._monotonic = monotonic

    def envelopes(self) -> Iterator[tuple[dict, bytes]]:
        """Yield ``(header, jpeg_bytes)`` per forwarded frame.

        Throttling replays capture-time gaps rather than a fixed period, because
        with adaptive forwarding there *is* no fixed period: the gap between two
        forwarded frames is 1 s at rest and 1/12 s inside a burst.
        """
        started = self._monotonic()
        first_ts: float | None = None
        for frame in self.frames():
            if self.throttle:
                if first_ts is None:
                    first_ts = frame.capture_ts
                due = started + max(frame.capture_ts - first_ts, 0.0)
                behind = due - self._monotonic()
                if behind > 0:
                    self._sleep(behind)
            yield frame.encode(self.jpeg_quality)

    def frames(self) -> Iterator[Frame]:
        """Yield the :class:`Frame` objects to forward (rectified if configured).

        Frames the forwarder drops and frames the rectifier suppresses never
        appear, so consumers only ever see crops the edge is confident in, at the
        rate D4 asks for.
        """
        for frame in self.source.frames():
            if self.forwarder is not None:
                decided = self.forwarder.decide(frame)
                if decided is None:
                    continue  # below the current keepalive/burst rate
                frame = decided
            if self.rectifier is None:
                yield frame
                continue
            result = self.rectifier.process(frame.image, frame.capture_ts)
            if not result.ok:
                continue  # suppressed during a bump / awaiting setup
            yield dataclasses.replace(frame, image=result.crop)
