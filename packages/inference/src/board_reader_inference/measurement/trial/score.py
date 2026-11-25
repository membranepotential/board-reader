"""Score the raw and settled streams of a trial against its truth, each on its own.

**Two streams, scored apart.** A published event carries two boards and they fail
differently: the read's own board (the argmax of ``square_dist``, recorded as the
top class of each square's ``evidence``) and the temporal debounce (``placement``,
:mod:`~board_reader_inference.stream.settle.stabilizer`). Scoring them side by side is the
whole point — it is what says whether the debounce earns its keep on a real
stream, which no single-photo eval can answer.

**The truth-frame check.** Truth is written in the chess frame (DESIGN D10), and a
recognizer reads in it, so on every scored raw read the board should agree with
its truth unturned. The report counts, per quarter-turn, the reads that agree with
the truth best when turned that far (:func:`~board_reader_inference.measurement.labels.best_rotation`)
and warns when the majority is not 0° — a truth file written in another frame, or
a recognizer misreading the orientation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from board_reader_inference.measurement.evaluate.errors import square_errors
from board_reader_inference.measurement.labels import best_rotation
from board_reader_inference.measurement.trial.events import SQUARES, RecordedEvent
from board_reader_inference.measurement.trial.truth import SessionTruth


class Stream(StrEnum):
    """The two boards a published event carries, scored side by side."""

    RAW = "raw"  # placement — the estimator's per-square argmax
    SETTLED = "settled"  # placement — the time-window debounce


@dataclass(frozen=True)
class ScoredRead:
    """One event joined to its truth segment, with both boards resolved."""

    event: RecordedEvent
    segment: int
    raw: str
    settled: str

    def board(self, stream: Stream) -> str:
        return self.raw if stream is Stream.RAW else self.settled


@dataclass
class SegmentScore:
    """One truth segment's slice of one stream."""

    index: int
    from_seq: int
    to_seq: int | None  # exclusive; None = runs to the end of the session
    placement: str
    events: int = 0
    correct: int = 0
    perfect: int = 0

    @property
    def squares(self) -> int:
        return self.events * SQUARES

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "from_seq": self.from_seq,
            "to_seq": self.to_seq,
            "placement": self.placement,
            "events": self.events,
            "correct": self.correct,
            "squares": self.squares,
            "perfect": self.perfect,
        }


@dataclass
class StreamScore:
    """How one of the two boards did over the whole scored session."""

    stream: Stream
    events: int
    correct: int
    perfect: int
    errors: Counter
    confusions: Counter
    segments: list[SegmentScore]
    #: True when this stream was reconstructed offline rather than read off the
    #: recording. Only ever the settled stream, and only for old recordings.
    derived: bool = False

    @property
    def squares(self) -> int:
        return self.events * SQUARES

    @property
    def accuracy(self) -> float:
        return self.correct / self.squares if self.squares else 0.0

    @property
    def perfect_rate(self) -> float:
        return self.perfect / self.events if self.events else 0.0

    def to_dict(self) -> dict:
        return {
            "stream": self.stream.value,
            "derived": self.derived,
            "events": self.events,
            "squares": self.squares,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "perfect": self.perfect,
            "perfect_rate": round(self.perfect_rate, 4),
            "errors_by_kind": dict(self.errors),
            "confusions": dict(self.confusions.most_common()),
            "per_segment": [s.to_dict() for s in self.segments],
        }


def score_stream(
    stream: Stream, reads: list[ScoredRead], truth: SessionTruth, *, derived: bool
) -> StreamScore:
    segments = [
        SegmentScore(
            index=i,
            from_seq=s.from_seq,
            to_seq=truth.starts[i + 1] if i + 1 < len(truth.segments) else None,
            placement=s.placement,
        )
        for i, s in enumerate(truth.segments)
    ]
    errors: Counter = Counter()
    confusions: Counter = Counter()
    correct = perfect = 0
    for read in reads:
        seg = segments[read.segment]
        misses = square_errors(truth.segments[read.segment].placement, read.board(stream))
        hit = SQUARES - len(misses)
        correct += hit
        seg.events += 1
        seg.correct += hit
        if not misses:
            perfect += 1
            seg.perfect += 1
        for m in misses:
            errors[m.kind] += 1
            confusions[f"{m.truth}->{m.predicted}"] += 1
    return StreamScore(
        stream=stream,
        events=len(reads),
        correct=correct,
        perfect=perfect,
        errors=errors,
        confusions=confusions,
        segments=segments,
        derived=derived,
    )


@dataclass(frozen=True)
class TruthFrame:
    """How many scored raw reads agree with their truth best at each quarter-turn."""

    by_rotation: tuple[int, int, int, int]

    @property
    def reads(self) -> int:
        return sum(self.by_rotation)

    @property
    def verdict(self) -> int:
        """The quarter-turn most reads agree under; 0 wins ties."""
        return max(range(4), key=lambda k: (self.by_rotation[k], k == 0))

    def to_dict(self) -> dict:
        return {
            "reads": self.reads,
            "by_rotation_deg": {str(90 * k): n for k, n in enumerate(self.by_rotation)},
            "verdict_deg": 90 * self.verdict,
        }


def score_truth_frame(reads: list[ScoredRead], truth: SessionTruth) -> TruthFrame:
    """The truth-frame check (module docstring) over the raw stream."""
    turns = Counter(best_rotation(truth.segments[r.segment].placement, r.raw) for r in reads)
    return TruthFrame(by_rotation=(turns[0], turns[1], turns[2], turns[3]))
