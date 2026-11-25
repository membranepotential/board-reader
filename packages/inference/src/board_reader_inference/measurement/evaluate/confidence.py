"""Whether the recognizer's confidence means anything: reliability, ECE, temperature.

**Perfect boards, not per-square** is the headline of the evaluation; this module
measures the other half of the claim — that a confidence of 0.97 earns its 0.97 —
and what temperature scaling would change about it, leave-one-board-out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board_reader_inference.measurement.evaluate.board import BoardResult, round_opt

#: Reliability-diagram resolution. Ten bins over [0, 1] is the conventional
#: choice and keeps each row readable at the ~640-square scale we evaluate at.
CONF_BINS = 10


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Overconfidence in this bin: positive means it claims more than it earns."""
        return self.mean_confidence - self.accuracy


@dataclass(frozen=True)
class TemperatureReport:
    """What temperature scaling buys, measured leave-one-board-out.

    Every field is computed by scaling each board with a ``T`` fitted on the
    *other* boards, so none of it is self-graded. ``answers_changed`` exists to
    be asserted at zero: scaling is a monotone per-square transform, so if it
    ever moves an argmax something is wrong with the implementation, not with
    the calibration.
    """

    temperatures: list[float]  # per fold, one per board
    ece_before: float
    ece_after: float
    bins_after: list[Bin]
    conf_perfect_before: float | None
    conf_imperfect_before: float | None
    conf_perfect_after: float | None
    conf_imperfect_after: float | None
    answers_changed: int
    #: Probability that a randomly chosen perfect board outranks a randomly
    #: chosen imperfect one by board confidence (0.5 = no signal). This, not the
    #: raw gap, is what a downstream consumer thresholding on confidence
    #: actually depends on — scaling compresses the gap by construction, so the
    #: gap alone would make a harmless change look like a loss.
    auc_before: float | None
    auc_after: float | None

    @property
    def temperature(self) -> float:
        """The representative ``T`` (median across folds)."""
        return float(np.median(self.temperatures)) if self.temperatures else 1.0

    @property
    def separation_before(self) -> float | None:
        if self.conf_perfect_before is None or self.conf_imperfect_before is None:
            return None
        return self.conf_perfect_before - self.conf_imperfect_before

    @property
    def separation_after(self) -> float | None:
        if self.conf_perfect_after is None or self.conf_imperfect_after is None:
            return None
        return self.conf_perfect_after - self.conf_imperfect_after

    def to_dict(self) -> dict:
        return {
            "temperature_median": round(self.temperature, 4),
            "temperatures": [round(t, 4) for t in self.temperatures],
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "answers_changed": self.answers_changed,
            "board_ranking_auc": {
                "before": round_opt(self.auc_before),
                "after": round_opt(self.auc_after),
            },
            "board_confidence": {
                "perfect_before": round_opt(self.conf_perfect_before),
                "imperfect_before": round_opt(self.conf_imperfect_before),
                "perfect_after": round_opt(self.conf_perfect_after),
                "imperfect_after": round_opt(self.conf_imperfect_after),
            },
        }


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _ranking_auc(positive: list[float], negative: list[float]) -> float | None:
    """P(a random positive scores above a random negative), ties counted as half.

    Plain Mann-Whitney U over the two small samples — with tens of boards this is
    exact and needs no approximation.
    """
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative
    )
    return wins / (len(positive) * len(negative))


def score_temperature(
    results: list[BoardResult],
    dists: list[np.ndarray],
    truth_indices: list[np.ndarray],
) -> TemperatureReport:
    """Leave-one-board-out temperature scaling, scored against the same labels."""
    from board_reader_inference.measurement.calibration import leave_one_out_scaled

    scaled, temps = leave_one_out_scaled(dists, truth_indices)
    conf_after = [float(d.max(axis=-1).mean()) for d in scaled]
    changed = sum(
        int((np.asarray(a).argmax(axis=-1) != np.asarray(b).argmax(axis=-1)).sum())
        for a, b in zip(dists, scaled)
    )
    bins_after, ece_after = reliability(
        np.concatenate([d.max(axis=-1).ravel() for d in scaled]) if scaled else np.empty(0),
        np.concatenate(
            [(d.argmax(axis=-1) == t).ravel() for d, t in zip(scaled, truth_indices)]
        )
        if scaled
        else np.empty(0, dtype=bool),
    )
    _, ece_before = reliability(
        np.concatenate([np.asarray(d).max(axis=-1).ravel() for d in dists]) if dists else np.empty(0),
        np.concatenate(
            [(np.asarray(d).argmax(axis=-1) == t).ravel() for d, t in zip(dists, truth_indices)]
        )
        if dists
        else np.empty(0, dtype=bool),
    )
    perfect = [i for i, r in enumerate(results) if r.perfect]
    imperfect = [i for i, r in enumerate(results) if not r.perfect]
    return TemperatureReport(
        temperatures=temps,
        ece_before=ece_before,
        ece_after=ece_after,
        bins_after=bins_after,
        conf_perfect_before=_mean_or_none([results[i].confidence for i in perfect]),
        conf_imperfect_before=_mean_or_none([results[i].confidence for i in imperfect]),
        conf_perfect_after=_mean_or_none([conf_after[i] for i in perfect]),
        conf_imperfect_after=_mean_or_none([conf_after[i] for i in imperfect]),
        answers_changed=changed,
        auc_before=_ranking_auc(
            [results[i].confidence for i in perfect],
            [results[i].confidence for i in imperfect],
        ),
        auc_after=_ranking_auc(
            [conf_after[i] for i in perfect], [conf_after[i] for i in imperfect]
        ),
    )


def reliability(
    confidences: np.ndarray, hits: np.ndarray, bins: int = CONF_BINS
) -> tuple[list[Bin], float]:
    """Reliability diagram + expected calibration error over per-square decisions.

    ECE is the sample-weighted mean gap between claimed confidence and observed
    accuracy. It is the number that says whether 0.97 means anything — a board
    read with two white kings at 0.97 is not a rare accident if the gap is large.
    """
    confidences = np.asarray(confidences, dtype=np.float64).ravel()
    hits = np.asarray(hits, dtype=bool).ravel()
    if confidences.size == 0:
        return [], 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Rightmost bin is closed so confidence exactly 1.0 is counted, not dropped.
    which = np.clip(np.digitize(confidences, edges[1:-1], right=True), 0, bins - 1)
    out: list[Bin] = []
    ece = 0.0
    for b in range(bins):
        sel = which == b
        count = int(sel.sum())
        if not count:
            continue
        mean_conf = float(confidences[sel].mean())
        acc = float(hits[sel].mean())
        out.append(Bin(float(edges[b]), float(edges[b + 1]), count, mean_conf, acc))
        ece += count / confidences.size * abs(mean_conf - acc)
    return out, ece
