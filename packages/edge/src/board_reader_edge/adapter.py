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

:class:`RecordingReplay` is the second envelope source in this module: it re-sends
a recorded session instead of processing a live one. Both satisfy
:class:`EnvelopeSource`, which is all the WS client needs.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Iterator, Protocol, TypeVar, runtime_checkable

import dataclasses

import numpy as np

from board_reader_core.recording import Recording
from board_reader_core.wire import Frame

from board_reader_edge.forwarder import AdaptiveForwarder
from board_reader_edge.rectify import Rectifier
from board_reader_edge.source import FrameSource

_T = TypeVar("_T")


@runtime_checkable
class EnvelopeSource(Protocol):
    """Anything that can be streamed to the gateway's ``/ingest`` socket."""

    def envelopes(self) -> Iterator[tuple[dict, bytes]]:
        ...


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


class RecordingReplay:
    """Re-send a recorded session (WP-H2) to the gateway, byte for byte.

    This is the reproduce-a-run path: the frames a gateway once received are put
    back on the wire **verbatim** — the recorded header dict and the recorded JPEG
    bytes, untouched. Nothing is decoded and re-encoded on the way through, because
    a JPEG round trip perturbs pixels, and a recognition run fed perturbed pixels
    is not a reproduction of the original one.

    Deliberately *not* a :class:`~board_reader_edge.source.FrameSource` feeding
    :class:`FileEdgeAdapter`: a recording already holds the edge's output — motion
    gating and rectification have happened — so re-running those stages over it
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
        speed it happened. Worth preferring for a faithful re-run: the gateway's
        latest-wins pre-filter drops frames as a function of arrival rate, so a
        flat-out replay makes it drop more than the original run did.
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
        when the point *is* to reprocess it (a different rectifier, a new motion
        threshold) rather than to reproduce it.
        """
        for rec in self.recording.frames(board_id=self.board_id):
            yield rec.frame()
