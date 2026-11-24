"""Whole-board evaluation of the **deployed** recognizer.

Two things this measures that nothing else did:

**Perfect boards, not per-square.** 94.5% per-square sounds nearly solved; on a
64-square board it means ~3.5 wrong squares, and a position with 3 wrong squares
is not a position. Whole-board correctness is the number that decides whether a
photo can be read, so it is the headline here and per-square is a diagnostic.

**What actually serves.** This runs the recognizer :func:`recognizer.build_from_env`
builds from the environment, so the number belongs to the thing the snapshot page
talks to. The recognizer is printed above every report, because an accuracy figure
without its model is a rumour.

Truth comes from :mod:`.labels` (operator-confirmed labels and the curated
``ground_truth.json``). A recognizer (DESIGN D10, :mod:`.recognizer`) reads the
whole photo, resolves the orientation itself and is scored only on chess-frame
labels; a corner-frame label cannot say which way its board really faces, so it
is skipped and counted.

**Frame diagnostic.** Every board also records the quarter-turn under which its
read agrees best with its truth (:func:`.labels.best_rotation`). A non-zero turn
means the read and the label disagree about which way the board faces — for a
recognizer, a misread orientation or a label in the wrong frame.

    uv run board-reader-inference eval
    uv run board-reader-inference eval data/uploads
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from board_reader_core.protocol import (
    CLASSES,
    GRID,
    parse_placement,
    square_name,
)

from board_reader_inference.measurement.labels import LabeledBoard, best_rotation
from board_reader_inference.recognition.recognizer import Recognizer

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
    #: Quarter-turns under which the read agrees best with the truth; 0 = same frame.
    rotation: int = 0

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
            "rotation": self.rotation,
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


@dataclass
class Report:
    boards: list[BoardResult]
    estimator: str
    bins: list[Bin] = field(default_factory=list)
    ece: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: Labels found but not scorable (corner-frame truth), by board id.
    skipped: list[str] = field(default_factory=list)
    #: Populated with ``--temperature``: what temperature scaling does to the
    #: CONFIDENCES (it cannot move a single answer — see :mod:`.calibration`).
    #: Every number here is leave-one-board-out, so no board is graded by a
    #: temperature its own labels helped fit.
    temperature: TemperatureReport | None = None

    @property
    def perfect_boards(self) -> int:
        return sum(b.perfect for b in self.boards)

    @property
    def rotated(self) -> list[BoardResult]:
        """Boards whose read agrees with the truth best when turned (the frame check)."""
        return [b for b in self.boards if b.rotation]

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
            "skipped": self.skipped,
            "rotated_boards": [
                {"board_id": b.board_id, "rotation": 90 * b.rotation} for b in self.rotated
            ],
            "perfect_boards": self.perfect_boards,
            "squares": self.squares,
            "correct_squares": self.correct,
            "ece": round(self.ece, 4),
            "errors_by_kind": dict(self.errors_by_kind()),
            "confusions": dict(self.confusions()),
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

    def render(self) -> str:
        if not self.boards:
            if self.skipped:
                return (
                    f"no scorable labeled boards — {len(self.skipped)} skipped, their "
                    f"truth is not in the frame {self.estimator} reads"
                )
            return "no labeled boards found — nothing to evaluate"
        n = len(self.boards)
        lines = [f"estimator: {self.estimator}"]
        lines += [f"warning:   {w}" for w in self.warnings]
        by_source = Counter(b.source for b in self.boards)
        lines.append(
            "boards:    "
            + f"{n} ({', '.join(f'{v} {k}' for k, v in sorted(by_source.items()))})"
            + (f" · {len(self.skipped)} skipped (frame)" if self.skipped else "")
        )
        rotated = self.rotated
        lines.append(
            f"frame:     {n - len(rotated)}/{n} read in the truth's frame"
            + (
                " · TURNED: "
                + ", ".join(f"{b.board_id} {90 * b.rotation}°" for b in rotated)
                if rotated
                else ""
            )
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


def describe(recognizer: Recognizer) -> str:
    """A one-line identity for the recognizer about to be measured."""
    name = type(recognizer).__name__
    return f"{name}({recognizer.model_path})" if recognizer.model_path else name


def read_board(recognizer: Recognizer, board: LabeledBoard) -> np.ndarray:
    """Run the recognizer over one labeled photo -> its (8, 8, 13) distribution."""
    import cv2

    image = cv2.imread(str(board.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read {board.image}")
    return np.asarray(recognizer.read(image).square_dist, dtype=np.float32)


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
        rotation=best_rotation(board.placement, predicted),
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
    recognizer: Recognizer,
    boards: list[LabeledBoard],
    *,
    temperature: bool = False,
) -> Report:
    """Read every labeled board and assemble the report."""
    results: list[BoardResult] = []
    confidences: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    truth_indices: list[np.ndarray] = []
    skipped = [b.board_id for b in boards if b.frame != "chess"]
    for board in (b for b in boards if b.frame == "chess"):
        dist = read_board(recognizer, board)
        result = score_board(board, dist)
        results.append(result)
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
    if skipped:
        warnings.append(
            f"{len(skipped)} corner-frame label(s) skipped: their truth does not say "
            "which way the board faces (tools/migrate_labels_to_chess_frame.py)"
        )
    return Report(
        boards=results,
        estimator=describe(recognizer),
        bins=bins,
        ece=ece,
        warnings=warnings,
        skipped=skipped,
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
    recognizer: Recognizer | None = None,
    temperature: bool = False,
    json_out: str | None = None,
) -> Report:
    """Load labels, build the deployed recognizer if none was injected, report."""
    from board_reader_inference.measurement.labels import load_labels

    candidates = [p for p in (paths or DEFAULT_PATHS) if Path(p).exists()]
    boards = load_labels(candidates)
    if recognizer is None:
        from board_reader_inference.recognition.recognizer import build_from_env

        recognizer = build_from_env()
    report = evaluate(recognizer, boards, temperature=temperature)
    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2))
    return report
