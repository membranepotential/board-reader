"""File-based edge adapter (DESIGN.md §6.1 step 1, WP-I1).

Pulls frames from a :class:`~board_reader_edge.source.FrameSource` and emits them as
wire envelopes ``(header, jpeg)`` for the inference gateway. This is the edge's
assembly point: motion-driven forwarding (WP-C3) and the board's geometry are both
optional stages here, and neither changes the wire shape.

**Two ways to hand the board's geometry across the wire** (DESIGN §4.5), mutually
exclusive:

* ``rectifier=`` — warp at the edge and forward a canonical top-down crop (WP-A).
  The corners are consumed here and never leave the node.
* ``wire_corners=`` — forward the **original** camera frame and attach the
  operator's corners to every wire frame, so the inference node cuts its boxes on
  the un-warped image (the photo path). At low camera angles a piece is much
  better represented by a tall box on the original view than by anything visible
  in the top-down crop, which is why this mode exists.

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

import numpy as np

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
        rectifier's re-detection cadence. Mutually exclusive with ``wire_corners``.
    wire_corners:
        Optional ``(4, 2)`` playing-surface quad in **source-image** pixels, in the
        operator's semantic order ``a8, h8, h1, a1``. When supplied, frames are
        forwarded **unwarped** with these corners attached to every wire frame
        (DESIGN §4.5). The order is passed through *verbatim* — it is what tells the
        inference node which physical corner is a1, so it must never be run through
        :func:`~board_reader_edge.rectify.order_quad`, which normalises to TL-first
        for the warp and throws that orientation away.
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
        wire_corners: np.ndarray | None = None,
        jpeg_quality: int = 90,
        throttle: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if rectifier is not None and wire_corners is not None:
            raise ValueError(
                "rectifier and wire_corners are exclusive: either the edge warps the "
                "board away, or it forwards the original frame with the corners on it"
            )
        if wire_corners is not None:
            quad = np.asarray(wire_corners, dtype=np.float32)
            if quad.size != 8:
                raise ValueError("wire_corners must be 4 (x, y) pairs: a8,h8,h1,a1")
            # Own copy, frozen: the same array is stamped on every yielded frame,
            # so a caller mutating its quad in place must not rewrite the corners
            # of frames already handed out (or of frames yet to come).
            wire_corners = quad.reshape(4, 2).copy()
            wire_corners.setflags(write=False)
        self.source = source
        self.forwarder = forwarder
        self.rectifier = rectifier
        self.wire_corners = wire_corners
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
        """Yield the :class:`Frame` objects to forward.

        Frames the forwarder drops and frames the rectifier suppresses never
        appear, so consumers only ever see crops the edge is confident in, at the
        rate D4 asks for. Under ``wire_corners`` there is nothing to suppress: the
        original frame goes out with the operator's quad attached.
        """
        for frame in self.source.frames():
            if self.forwarder is not None:
                decided = self.forwarder.decide(frame)
                if decided is None:
                    continue  # below the current keepalive/burst rate
                frame = decided
            if self.wire_corners is not None:
                yield dataclasses.replace(frame, corners=self.wire_corners)
                continue
            if self.rectifier is None:
                yield frame
                continue
            result = self.rectifier.process(frame.image, frame.capture_ts)
            if not result.ok:
                continue  # suppressed during a bump / awaiting setup
            yield dataclasses.replace(frame, image=result.crop)
