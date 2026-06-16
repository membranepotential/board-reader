"""File-based edge adapter (DESIGN.md §6.1 step 1, WP-I1).

Pulls frames from a :class:`~board_reader_edge.source.FrameSource` and emits them as
wire envelopes ``(header, jpeg)`` for the inference gateway. In M0 this is a plain
pass-through at a fixed rate; board detection/rectification (WP-A) and adaptive
burst forwarding (WP-C3) slot in here in later milestones without changing the
wire shape.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator

from board_reader_core.wire import Frame

from board_reader_edge.source import FrameSource


class FileEdgeAdapter:
    """Adapt a frame source into a stream of wire envelopes.

    Parameters
    ----------
    source:
        The frame source to drain.
    jpeg_quality:
        JPEG quality for the encoded crop.
    throttle:
        If ``True``, sleep between frames to honour ``fps`` (best-effort wall-clock).
        Off by default so offline replay runs flat out.
    sleep / monotonic:
        Injectable for deterministic tests.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        jpeg_quality: int = 90,
        throttle: bool = False,
        fps: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.jpeg_quality = jpeg_quality
        self.throttle = throttle
        self.period = 1.0 / fps if fps > 0 else 0.0
        self._sleep = sleep
        self._monotonic = monotonic

    def envelopes(self) -> Iterator[tuple[dict, bytes]]:
        """Yield ``(header, jpeg_bytes)`` per frame."""
        next_at = self._monotonic()
        for frame in self.source.frames():
            if self.throttle and self.period:
                now = self._monotonic()
                if now < next_at:
                    self._sleep(next_at - now)
                next_at = max(next_at + self.period, self._monotonic())
            yield frame.encode(self.jpeg_quality)

    def frames(self) -> Iterator[Frame]:
        """Yield the raw :class:`Frame` objects (for in-process wiring/tests)."""
        yield from self.source.frames()
