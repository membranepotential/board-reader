"""Whole-board evaluation of the **deployed** estimator stack.

Two things this measures that nothing else did:

**Perfect boards, not per-square.** 94.5% per-square sounds nearly solved; on a
64-square board it means ~3.5 wrong squares, and a position with 3 wrong squares
is not a position. Whole-board correctness is the number that decides whether a
photo can be read, so it is the headline here and per-square is a diagnostic.

**What actually serves.** ``tools/train/evaluate.py`` scores the trained model
alone, in the training venv. This runs the estimator :func:`recognizer.build_from_env`
builds from the environment — ensemble, bank blend and all — so the number
belongs to the thing the snapshot page talks to. The configuration is printed
above every report, because an accuracy figure without its estimator is a rumour.

Truth comes from :mod:`.labels` (operator-confirmed labels and the curated
``ground_truth.json``), always in the frame the photo's corners define. The
estimator is therefore scored on recognition given a labeled board, with
orientation held fixed; ``--autorotate`` folds the orientation guess back in for
an end-to-end figure.

``--prior`` adds a second, opt-in score: the same distributions run through the
hard-constraint decoder (:mod:`.prior`). It is reported *beside* the estimator's own
numbers, with squares fixed and squares broken kept apart, because a repair that
trades one right square for one wrong one is not an improvement.

    uv run board-reader-inference eval
    BR_MODEL=models/run_e/model.onnx uv run board-reader-inference eval data/uploads
    ... eval --prior data/uploads       # does the hard-constraint decode help?
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from board_reader_core.protocol import (
    CLASSES,
    GRID,
    NUM_CLASSES,
    parse_placement,
    square_name,
)
from board_reader_core.wire import Frame

from board_reader_inference.labels import LabeledBoard
from board_reader_inference.prior import PriorDecode, decode

#: Reliability-diagram resolution. Ten bins over [0, 1] is the conventional
#: choice and keeps each row readable at the ~640-square scale we evaluate at.
CONF_BINS = 10



def error_kind(truth: str, predicted: str) -> str:
    """How a square went wrong — the four failures that mean different things.

    A hallucination and a missed piece are occupancy failures (the geometry or
    the crop is wrong); a colour or type error is a recognition failure with the
    occupancy already right. They have different fixes, so they are counted apart.
    """
    if truth == "empty":
        return "hallucination"
    if predicted == "empty":
        return "missed"
    if truth.lower() == predicted.lower():
        return "colour"
    return "type"


@dataclass(frozen=True)
class SquareError:
    square: str
    truth: str
    predicted: str
    kind: str
    #: The estimator's confidence in the wrong answer, when it is known. A caller
    #: comparing two *placements* — a recorded session, whose ``square_dist`` was
    #: stripped before it hit disk — has none, and ``None`` says so rather than
    #: inventing a zero.
    confidence: float | None = None

    def __str__(self) -> str:
        return f"{self.square} {self.truth}->{self.predicted}"


def square_errors(
    truth: str, predicted: str, confidence: np.ndarray | None = None
) -> tuple[SquareError, ...]:
    """Every square where ``predicted`` disagrees with ``truth``, tagged by kind.

    Both arguments are FEN field 1, so this works equally on a distribution's
    argmax and on a placement read back off a recorded stream. ``confidence`` is
    the optional 8x8 grid of per-square max probability.
    """
    t, p = parse_placement(truth), parse_placement(predicted)
    return tuple(
        SquareError(
            square=square_name(r, c),
            truth=t[r][c],
            predicted=p[r][c],
            kind=error_kind(t[r][c], p[r][c]),
            confidence=None if confidence is None else float(confidence[r][c]),
        )
        for r in range(GRID)
        for c in range(GRID)
        if t[r][c] != p[r][c]
    )


@dataclass(frozen=True)
class BoardResult:
    board_id: str
    source: str
    truth: str
    predicted: str
    correct: int
    confidence: float  # mean per-square max probability
    errors: tuple[SquareError, ...]

    @property
    def perfect(self) -> bool:
        return self.correct == GRID * GRID

    def to_dict(self) -> dict:
        return {
            "board_id": self.board_id,
            "source": self.source,
            "truth": self.truth,
            "predicted": self.predicted,
            "correct": self.correct,
            "total": GRID * GRID,
            "perfect": self.perfect,
            "confidence": round(self.confidence, 4),
            "errors": [
                {
                    "square": e.square,
                    "truth": e.truth,
                    "predicted": e.predicted,
                    "kind": e.kind,
                    "confidence": _round_opt(e.confidence),
                }
                for e in self.errors
            ],
        }


class FlipVerdict(StrEnum):
    """What a decoder flip did to a square, judged against the truth label."""

    FIXED = "fixed"  # was wrong, now right
    BROKEN = "broken"  # was right, now wrong
    NEUTRAL = "neutral"  # wrong before and after — a different wrong answer


@dataclass(frozen=True)
class SquareChange:
    """One square the decoder relabelled, with the truth beside it."""

    square: str
    truth: str
    before: str
    after: str

    @property
    def verdict(self) -> FlipVerdict:
        if self.after == self.truth:
            return FlipVerdict.FIXED
        if self.before == self.truth:
            return FlipVerdict.BROKEN
        return FlipVerdict.NEUTRAL

    def __str__(self) -> str:
        return f"{self.square} {self.before}->{self.after} (truth {self.truth})"

    def to_dict(self) -> dict:
        return {
            "square": self.square,
            "truth": self.truth,
            "before": self.before,
            "after": self.after,
            "verdict": self.verdict.value,
        }


@dataclass(frozen=True)
class PriorBoardResult:
    """One board scored twice: as the estimator read it, and as the decoder repaired it.

    ``fixed`` and ``broken`` are kept apart on purpose. A decoder that fixes three
    squares and breaks two is not "+1"; it is a coin flip with extra steps, and the
    net number would hide that.
    """

    board_id: str
    decode: PriorDecode
    changes: tuple[SquareChange, ...]
    correct_before: int
    correct_after: int

    @property
    def fixed(self) -> tuple[SquareChange, ...]:
        return tuple(c for c in self.changes if c.verdict is FlipVerdict.FIXED)

    @property
    def broken(self) -> tuple[SquareChange, ...]:
        return tuple(c for c in self.changes if c.verdict is FlipVerdict.BROKEN)

    @property
    def perfect_before(self) -> bool:
        return self.correct_before == GRID * GRID

    @property
    def perfect_after(self) -> bool:
        return self.correct_after == GRID * GRID

    def to_dict(self) -> dict:
        return {
            "board_id": self.board_id,
            "placement": self.decode.placement,
            "flips": [f.to_dict() for f in self.decode.flips],
            "residual": self.decode.residual.to_dict(),
            "changes": [c.to_dict() for c in self.changes],
            "correct_before": self.correct_before,
            "correct_after": self.correct_after,
            "fixed": len(self.fixed),
            "broken": len(self.broken),
        }


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
                "before": _round_opt(self.auc_before),
                "after": _round_opt(self.auc_after),
            },
            "board_confidence": {
                "perfect_before": _round_opt(self.conf_perfect_before),
                "imperfect_before": _round_opt(self.conf_imperfect_before),
                "perfect_after": _round_opt(self.conf_perfect_after),
                "imperfect_after": _round_opt(self.conf_imperfect_after),
            },
        }


def _round_opt(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


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
    from board_reader_inference.calibration import leave_one_out_scaled

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


@dataclass
class Report:
    boards: list[BoardResult]
    estimator: str
    bins: list[Bin] = field(default_factory=list)
    ece: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: Populated only with ``--prior``: the same boards re-scored after the
    #: hard-constraint decoder. Empty means the decoder was not run at all, which
    #: is not the same as it having found nothing to do.
    prior: list[PriorBoardResult] = field(default_factory=list)
    #: Populated with ``--temperature``: what temperature scaling does to the
    #: CONFIDENCES (it cannot move a single answer — see :mod:`.calibration`).
    #: Every number here is leave-one-board-out, so no board is graded by a
    #: temperature its own labels helped fit.
    temperature: TemperatureReport | None = None

    @property
    def perfect_boards(self) -> int:
        return sum(b.perfect for b in self.boards)

    @property
    def prior_perfect_boards(self) -> int:
        return sum(p.perfect_after for p in self.prior)

    @property
    def prior_correct(self) -> int:
        return sum(p.correct_after for p in self.prior)

    @property
    def prior_fixed(self) -> int:
        return sum(len(p.fixed) for p in self.prior)

    @property
    def prior_broken(self) -> int:
        return sum(len(p.broken) for p in self.prior)

    @property
    def squares(self) -> int:
        return len(self.boards) * GRID * GRID

    @property
    def correct(self) -> int:
        return sum(b.correct for b in self.boards)

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "warnings": self.warnings,
            "boards": len(self.boards),
            "perfect_boards": self.perfect_boards,
            "squares": self.squares,
            "correct_squares": self.correct,
            "ece": round(self.ece, 4),
            "errors_by_kind": dict(self.errors_by_kind()),
            "confusions": dict(self.confusions()),
            "prior": {
                "perfect_boards": self.prior_perfect_boards,
                "correct_squares": self.prior_correct,
                "fixed": self.prior_fixed,
                "broken": self.prior_broken,
                "per_board": [p.to_dict() for p in self.prior],
            }
            if self.prior
            else None,
            "temperature": self.temperature.to_dict() if self.temperature else None,
            "reliability": [
                {
                    "lo": b.lo,
                    "hi": b.hi,
                    "count": b.count,
                    "mean_confidence": round(b.mean_confidence, 4),
                    "accuracy": round(b.accuracy, 4),
                }
                for b in self.bins
            ],
            "per_board": [b.to_dict() for b in self.boards],
        }

    def errors_by_kind(self) -> Counter:
        return Counter(e.kind for b in self.boards for e in b.errors)

    def confusions(self) -> Counter:
        return Counter(f"{e.truth}->{e.predicted}" for b in self.boards for e in b.errors)

    def _render_temperature(self) -> list[str]:
        """The temperature block: what honest confidences would look like.

        Reported leave-one-board-out, and always next to ``answers changed``, so
        the reader can see the claim being made is about confidence only.
        """
        t = self.temperature
        if t is None:
            return []
        lines = ["", f"temperature scaling (leave-one-board-out, T={t.temperature:.3f})"]
        spread = (min(t.temperatures), max(t.temperatures)) if t.temperatures else (1.0, 1.0)
        lines.append(f"  T across folds: {spread[0]:.3f}-{spread[1]:.3f}")
        lines.append(f"  ECE {t.ece_before:.3f} -> {t.ece_after:.3f}")
        if t.separation_before is not None and t.separation_after is not None:
            lines.append(
                f"  board confidence perfect-imperfect gap: "
                f"{t.separation_before:+.3f} -> {t.separation_after:+.3f}"
                f"  (perfect {t.conf_perfect_after:.3f} · "
                f"imperfect {t.conf_imperfect_after:.3f})"
            )
        if t.auc_before is not None and t.auc_after is not None:
            lines.append(
                f"  board ranking AUC: {t.auc_before:.3f} -> {t.auc_after:.3f}"
                "   (0.5 = confidence carries no board-level signal)"
            )
        # Must be zero: scaling is monotone per square. A nonzero here is a bug.
        lines.append(f"  answers changed: {t.answers_changed}")
        for b in t.bins_after:
            lines.append(
                f"  {b.lo:.1f}-{b.hi:.1f}{b.count:>7}"
                f"{b.mean_confidence:>10.3f}{b.accuracy:>9.3f}{b.gap:>+7.3f}"
            )
        return lines

    def _render_prior(self) -> list[str]:
        """The prior block: both scores side by side, then every flip.

        Fixed and broken are printed apart, and each flip is printed with its truth
        label, because the only way to trust this decoder is to read its edits.
        """
        if not self.prior:
            return []
        n = len(self.boards)
        touched = [p for p in self.prior if p.decode.flips]
        residual = [p for p in self.prior if not p.decode.residual.satisfied]
        lines = [
            "",
            "hard-constraint prior (decoded from square_dist; the wire is unchanged)",
            "  frame unknown here, so the back-rank pawn constraint sits out — counts only",
        ]
        lines.append(
            f"  PERFECT BOARDS   {self.perfect_boards}/{n} -> "
            f"{self.prior_perfect_boards}/{n}"
        )
        lines.append(
            f"  per-square       {self.correct}/{self.squares} -> "
            f"{self.prior_correct}/{self.squares}"
            f"   (fixed {self.prior_fixed} · broken {self.prior_broken})"
        )
        lines.append(
            f"  boards touched   {len(touched)}/{n}"
            + (f" · residual violation on {len(residual)}" if residual else "")
        )
        if not touched:
            lines.append("  no board violated a hard constraint — nothing to decode")
            return lines
        fired = Counter(f.constraint.value for p in touched for f in p.decode.flips)
        lines.append(
            "  constraints fired  "
            + " · ".join(f"{k} {v}" for k, v in fired.most_common())
        )
        lines.append("  flips")
        for p in touched:
            by_square = {c.square: c for c in p.changes}
            for flip in p.decode.flips:
                change = by_square.get(flip.square)
                verdict = change.verdict.value if change else "reverted"
                truth = change.truth if change else "?"
                lines.append(
                    f"    {p.board_id:<32}{flip!s:<16}cost {flip.cost:5.2f}"
                    f"  {flip.constraint.value:<17}truth {truth:<6}{verdict.upper()}"
                )
            if not p.decode.residual.satisfied:
                lines.append(f"    {p.board_id:<32}residual: {p.decode.residual}")
        return lines

    def render(self) -> str:
        if not self.boards:
            return "no labeled boards found — nothing to evaluate"
        n = len(self.boards)
        lines = [f"estimator: {self.estimator}"]
        lines += [f"warning:   {w}" for w in self.warnings]
        by_source = Counter(b.source for b in self.boards)
        lines.append(
            "boards:    "
            + f"{n} ({', '.join(f'{v} {k}' for k, v in sorted(by_source.items()))})"
        )
        lines.append("")
        lines.append(
            f"PERFECT BOARDS   {self.perfect_boards}/{n}"
            f"   {100.0 * self.perfect_boards / n:5.1f}%"
        )
        lines.append(
            f"per-square       {self.correct}/{self.squares}"
            f"   {100.0 * self.correct / self.squares:5.1f}%"
        )

        lines += self._render_prior()

        kinds = self.errors_by_kind()
        if kinds:
            lines.append("")
            lines.append("errors by kind")
            for kind, count in kinds.most_common():
                lines.append(f"  {kind:<14}{count:>4}")
            lines.append("top confusions")
            for pair, count in self.confusions().most_common(8):
                lines.append(f"  {pair:<14}{count:>4}")

        if self.bins:
            lines.append("")
            lines.append(f"confidence reliability (ECE {self.ece:.3f})")
            lines.append("  bin          n   claimed   actual    gap")
            for b in self.bins:
                lines.append(
                    f"  {b.lo:.1f}-{b.hi:.1f}{b.count:>7}"
                    f"{b.mean_confidence:>10.3f}{b.accuracy:>9.3f}{b.gap:>+7.3f}"
                )
            # The tell for uncalibrated confidence: if a wrong board is as
            # confident as a right one, confidence carries no signal at the level
            # the downstream smoothing plan (DESIGN §9) intends to consume it.
            good = [b.confidence for b in self.boards if b.perfect]
            bad = [b.confidence for b in self.boards if not b.perfect]
            parts = [
                f"{label} {np.mean(vals):.3f} ({len(vals)})"
                for label, vals in (("perfect", good), ("imperfect", bad))
                if vals
            ]
            lines.append("  board confidence: " + " · ".join(parts))
            lines.extend(self._render_temperature())

        lines.append("")
        lines.append("per board")
        for b in sorted(self.boards, key=lambda x: (x.correct, x.board_id)):
            errs = ", ".join(str(e) for e in b.errors[:6])
            if len(b.errors) > 6:
                errs += f", +{len(b.errors) - 6} more"
            mark = "ok " if b.perfect else "   "
            lines.append(
                f"  {mark}{b.board_id:<34}{b.correct:>3}/64  conf {b.confidence:.2f}"
                + (f"  {errs}" if errs else "")
            )
        return "\n".join(lines)


def describe_estimator(est: object) -> str:
    """A one-line identity for whatever stack is about to be measured."""
    name = type(est).__name__
    primary, booster = getattr(est, "primary", None), getattr(est, "booster", None)
    if primary is not None and booster is not None:
        w = float(getattr(est, "weight", 0.5))
        return (
            f"{describe_estimator(primary)} x{w:.2f}"
            f" + {describe_estimator(booster)} x{1 - w:.2f}"
        )
    bits: list[str] = []
    model_path = getattr(est, "model_path", None)
    if model_path:
        bits.append(str(model_path))
    classifier = getattr(est, "classifier", None)
    if classifier is not None:
        exemplars = getattr(classifier, "exemplars", None)
        if exemplars is not None:
            bits.append(f"{int(np.asarray(exemplars).shape[0])} exemplars")
        bits.append(str(getattr(classifier, "backbone", "?")))
    headroom = getattr(est, "headroom", None)
    if headroom is not None:
        bits.append(f"hr={float(headroom):g} sp={float(getattr(est, 'side_pad', 0)):g}")
    return f"{name}({', '.join(bits)})" if bits else name


def has_bank(est: object) -> bool:
    """Whether a per-set exemplar bank is part of the stack (contamination risk)."""
    if getattr(est, "classifier", None) is not None:
        return True
    return any(
        has_bank(m)
        for m in (getattr(est, "primary", None), getattr(est, "booster", None))
        if m is not None
    )


def read_board(estimator: object, board: LabeledBoard, *, autorotate: bool = False) -> np.ndarray:
    """Run the estimator over one labeled photo -> its (8, 8, 13) distribution."""
    import cv2

    photo_fn = getattr(estimator, "estimate_photo", None)
    if not callable(photo_fn):
        raise ValueError(
            f"{type(estimator).__name__} has no photo path; evaluation needs an "
            "estimator that reads original photos with corners (set BR_MODEL "
            "and/or BR_CENTROIDS)"
        )
    image = cv2.imread(str(board.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read {board.image}")
    event = photo_fn(
        Frame(board_id=board.board_id, seq=0, capture_ts=0.0, image=image),
        board.corners,
    )
    dist = np.asarray(event.square_dist, dtype=np.float32).reshape(GRID, GRID, NUM_CLASSES)
    if autorotate:
        from board_reader_inference.autocalib import resolve_rotation

        _, dist = resolve_rotation(dist)
    return dist


def score_board(board: LabeledBoard, dist: np.ndarray) -> BoardResult:
    """Compare one distribution against its truth, square by square."""
    from board_reader_core.protocol import encode_placement

    idx = dist.argmax(axis=-1)
    conf = dist.max(axis=-1)
    predicted = encode_placement(
        [[CLASSES[int(idx[r, c])] for c in range(GRID)] for r in range(GRID)]
    )
    errors = square_errors(board.placement, predicted, conf)
    return BoardResult(
        board_id=board.board_id,
        source=board.source,
        truth=board.placement,
        predicted=predicted,
        correct=GRID * GRID - len(errors),
        confidence=float(conf.mean()),
        errors=errors,
    )


def score_prior(board: LabeledBoard, dist: np.ndarray) -> PriorBoardResult:
    """Decode one distribution under the hard constraints and score the result.

    ``dist`` is only read — the decoder returns a placement beside the event, it
    never rewrites the distribution the estimator produced (DESIGN D1).

    Truth here lives in the frame the photo's *corners* define, which is not a
    chess frame — 16 of 21 confirmed boards were shot a quarter- or half-turn off —
    and nothing in the eval *establishes* the true one. Resolving it with
    :func:`.autocalib.resolve_rotation` would be circular, since that function
    picks the rotation minimising back-rank pawns, so the frame is reported as
    unknown and :attr:`Constraint.BACK_RANK_PAWNS` sits out. What is measured here
    is therefore the three rotation-invariant count constraints.
    """
    result = decode(dist, quarter_turns=None)
    truth = parse_placement(board.placement)
    before = [[CLASSES[int(i)] for i in row] for row in dist.argmax(axis=-1)]
    after = parse_placement(result.placement)
    changes = tuple(
        SquareChange(
            square=square_name(r, c),
            truth=truth[r][c],
            before=before[r][c],
            after=after[r][c],
        )
        for r in range(GRID)
        for c in range(GRID)
        if before[r][c] != after[r][c]
    )
    return PriorBoardResult(
        board_id=board.board_id,
        decode=result,
        changes=changes,
        correct_before=sum(
            before[r][c] == truth[r][c] for r in range(GRID) for c in range(GRID)
        ),
        correct_after=sum(
            after[r][c] == truth[r][c] for r in range(GRID) for c in range(GRID)
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


def evaluate(
    estimator: object,
    boards: list[LabeledBoard],
    *,
    autorotate: bool = False,
    prior: bool = True,
    temperature: bool = False,
) -> Report:
    """Read every labeled board and assemble the report.

    ``prior`` scores the hard-constraint decode of each distribution
    (:mod:`.prior`) alongside the estimator's own numbers. It defaults to **on**
    because the decoder now serves (``/estimate`` returns its repair), and this
    harness's whole contract is to score what actually serves — an off-by-default
    flag would quietly stop measuring the product. It still changes nothing about
    the primary numbers: both scores are reported side by side, never folded
    together. Pass ``prior=False`` (``--no-prior``) to isolate the raw estimator.
    """
    results: list[BoardResult] = []
    prior_results: list[PriorBoardResult] = []
    confidences: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    truth_indices: list[np.ndarray] = []
    for board in boards:
        dist = read_board(estimator, board, autorotate=autorotate)
        result = score_board(board, dist)
        results.append(result)
        if prior:
            prior_results.append(score_prior(board, dist))
        truth = parse_placement(board.placement)
        truth_idx = np.array(
            [[CLASSES.index(truth[r][c]) for c in range(GRID)] for r in range(GRID)]
        )
        dists.append(dist)
        truth_indices.append(truth_idx)
        confidences.append(dist.max(axis=-1).ravel())
        hits.append((dist.argmax(axis=-1) == truth_idx).ravel())
    bins, ece = reliability(
        np.concatenate(confidences) if confidences else np.empty(0),
        np.concatenate(hits) if hits else np.empty(0, dtype=bool),
    )
    warnings: list[str] = []
    if has_bank(estimator):
        warnings.append(
            "a calibration bank is active — boards it was built from are NOT held "
            "out, and score far above what this stack achieves on an unseen set"
        )
    return Report(
        boards=results,
        estimator=describe_estimator(estimator),
        bins=bins,
        ece=ece,
        warnings=warnings,
        prior=prior_results,
        temperature=(
            score_temperature(results, dists, truth_indices) if temperature else None
        ),
    )


#: Where labeled boards live by default: the curated set plus everything the
#: snapshot page has had confirmed. Both are git-ignored, so an empty result on
#: a fresh clone is expected, not an error.
DEFAULT_PATHS = ("data/eval", "data/uploads")


def run(
    paths: list[str] | None = None,
    *,
    estimator: object | None = None,
    autorotate: bool = False,
    prior: bool = True,
    temperature: bool = False,
    json_out: str | None = None,
) -> Report:
    """Load labels, build the deployed estimator if none was injected, report."""
    from board_reader_inference.labels import load_labels

    candidates = [p for p in (paths or DEFAULT_PATHS) if Path(p).exists()]
    boards = load_labels(candidates)
    if estimator is None:
        from board_reader_inference.recognizer import build_from_env

        estimator = build_from_env()
    report = evaluate(
        estimator, boards, autorotate=autorotate, prior=prior, temperature=temperature
    )
    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2))
    return report
