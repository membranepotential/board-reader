"""What a trial cost in time: input and read rates, and capture-to-publish latency."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from board_reader_core.recording import Recording

from board_reader_inference.measurement.trial.events import RecordedEvent
from board_reader_inference.measurement.trial.stats import quantile


@dataclass
class Timing:
    """What the session cost in time — the rates a placement score cannot show.

    Two clocks, deliberately kept apart: frames *arriving* (what the phone sent)
    and events *published* (what the estimator kept up with). Their ratio is the
    read-rate story, and ``latencies`` is the capture-to-publish lag a live
    display actually feels.
    """

    frames: int
    frames_span: float
    events: int
    events_span: float
    latencies: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.frames_span if self.frames_span > 0 else 0.0

    @property
    def reads_per_s(self) -> float:
        return self.events / self.events_span if self.events_span > 0 else 0.0

    @property
    def frames_per_read(self) -> float:
        return self.frames / self.events if self.events else 0.0

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "frames_span": round(self.frames_span, 3),
            "fps": round(self.fps, 3),
            "events": self.events,
            "events_span": round(self.events_span, 3),
            "reads_per_s": round(self.reads_per_s, 3),
            "frames_per_read": round(self.frames_per_read, 2),
            "latency": {
                "n": len(self.latencies),
                "median": round(median(self.latencies), 3) if self.latencies else None,
                "p90": round(quantile(self.latencies, 0.9), 3) if self.latencies else None,
                "max": round(max(self.latencies), 3) if self.latencies else None,
            },
        }


def measure_timing(
    recording: Recording, board_id: str, events: list[RecordedEvent]
) -> Timing:
    """Rates and latencies for one board, from the index and the event stamps.

    Reads the frame *index* only — a session's JPEGs are hundreds of MB and none
    of them are needed to count arrivals.
    """
    arrivals = [e.received_ts for e in recording.frame_index(board_id=board_id)]
    captures = [e.capture_ts for e in events if e.capture_ts > 0.0]
    latencies = [lat for lat in (e.latency for e in events) if lat is not None]
    return Timing(
        frames=len(arrivals),
        frames_span=(arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0,
        events=len(events),
        events_span=(captures[-1] - captures[0]) if len(captures) > 1 else 0.0,
        latencies=latencies,
    )
