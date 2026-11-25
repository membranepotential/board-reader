"""What the estimator's own distribution says about the squares it got wrong.

The digest a recording keeps (the top-k classes per square) is the one thing a
placement alone cannot tell you: whether a wrong square was a coin flip or a
conviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from board_reader_core.protocol import GRID, square_name

from board_reader_inference.measurement.evaluate.errors import square_errors
from board_reader_inference.measurement.trial.score import ScoredRead
from board_reader_inference.measurement.trial.stats import quantile
from board_reader_inference.measurement.trial.truth import SessionTruth

#: Board-order index of every algebraic square, matching the evidence digest.
SQUARE_INDEX: dict[str, int] = {
    square_name(r, c): r * GRID + c for r in range(GRID) for c in range(GRID)
}


@dataclass
class ConfusionEvidence:
    """One truth->predicted pair, seen through the estimator's own distribution."""

    pair: str
    n: int = 0
    truth_ranked: int = 0  # truth appeared anywhere in the digest
    margins: list[float] = field(default_factory=list)

    @property
    def median_margin(self) -> float:
        return median(self.margins) if self.margins else 0.0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "n": self.n,
            "truth_ranked": self.truth_ranked,
            "median_margin": round(self.median_margin, 4),
        }


@dataclass
class EvidenceScore:
    """How close the estimator came on the squares it got wrong.

    Scored against the **raw** board on purpose: raw is the per-square argmax of
    the very distribution the digest summarises, so ``margin`` is exactly
    "how much more probability the wrong answer had than the right one". The
    settled board is composed *of* these reads, so the same evidence is what it
    had to work with.

    A small margin with truth as the runner-up is a fixable error — the crop
    carries the answer and the model nearly found it. Truth outside the top
    ``EVIDENCE_TOP_K`` is a different animal: the evidence is not there at all.
    """

    events: int  # scored events that carried a digest
    events_total: int  # scored events, digest or not
    wrong: int  # wrong squares over the events that carried one
    truth_runner_up: int
    truth_ranked: int
    margins: list[float] = field(default_factory=list)
    by_confusion: list[ConfusionEvidence] = field(default_factory=list)

    @property
    def recorded(self) -> bool:
        return self.events > 0

    def to_dict(self) -> dict:
        return {
            "events": self.events,
            "events_total": self.events_total,
            "wrong": self.wrong,
            "truth_runner_up": self.truth_runner_up,
            "truth_ranked": self.truth_ranked,
            "margin_median": round(median(self.margins), 4) if self.margins else None,
            "margin_p90": round(quantile(self.margins, 0.9), 4) if self.margins else None,
            "margin_max": round(max(self.margins), 4) if self.margins else None,
            "by_confusion": [c.to_dict() for c in self.by_confusion],
        }


def score_evidence(reads: list[ScoredRead], truth: SessionTruth) -> EvidenceScore:
    """Read every wrong raw square back against the digest that produced it."""
    with_digest = [r for r in reads if r.event.evidence is not None]
    score = EvidenceScore(
        events=len(with_digest), events_total=len(reads), wrong=0,
        truth_runner_up=0, truth_ranked=0,
    )
    per_pair: dict[str, ConfusionEvidence] = {}
    for read in with_digest:
        digest = read.event.evidence
        assert digest is not None  # filtered above; narrows for the type checker
        for miss in square_errors(truth.segments[read.segment].placement, read.raw):
            square = digest[SQUARE_INDEX[miss.square]]
            probs = dict(square)
            pair = per_pair.setdefault(
                f"{miss.truth}->{miss.predicted}", ConfusionEvidence(pair=f"{miss.truth}->{miss.predicted}")
            )
            score.wrong += 1
            pair.n += 1
            if miss.truth not in probs:
                continue
            score.truth_ranked += 1
            pair.truth_ranked += 1
            if len(square) > 1 and square[1][0] == miss.truth:
                score.truth_runner_up += 1
            margin = probs[miss.predicted] - probs[miss.truth]
            score.margins.append(margin)
            pair.margins.append(margin)
    score.by_confusion = sorted(per_pair.values(), key=lambda c: -c.n)
    return score
