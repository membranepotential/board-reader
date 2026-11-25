"""What the node says about itself on the health tick of a control page."""

from __future__ import annotations

from dataclasses import dataclass


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
