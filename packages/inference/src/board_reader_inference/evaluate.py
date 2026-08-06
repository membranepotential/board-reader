"""Whole-board evaluation of the **deployed** estimator stack.

Two things this measures that nothing else did:

**Perfect boards, not per-square.** 94.5% per-square sounds nearly solved; on a
64-square board it means ~3.5 wrong squares, and a position with 3 wrong squares
is not a position. Whole-board correctness is the number that decides whether a
photo can be read, so it is the headline here and per-square is a diagnostic.

**What actually serves.** ``tools/train/evaluate.py`` scores the trained model
alone, in the training venv. This runs the estimator :func:`server._default_estimator`
builds from the environment — ensemble, bank blend and all — so the number
belongs to the thing the snapshot page talks to. The configuration is printed
above every report, because an accuracy figure without its estimator is a rumour.

Truth comes from :mod:`.labels` (operator-confirmed labels and the curated
``ground_truth.json``), always in the frame the photo's corners define. The
estimator is therefore scored on recognition given a labeled board, with
orientation held fixed; ``--autorotate`` folds the orientation guess back in for
an end-to-end figure.

    uv run board-reader-inference eval
    BR_MODEL=models/run_e/model.onnx uv run board-reader-inference eval data/uploads
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from board_reader_core.protocol import CLASSES, GRID, NUM_CLASSES, parse_placement
from board_reader_core.wire import Frame

from board_reader_inference.labels import LabeledBoard

#: Reliability-diagram resolution. Ten bins over [0, 1] is the conventional
#: choice and keeps each row readable at the ~640-square scale we evaluate at.
CONF_BINS = 10


def _square_name(r: int, c: int) -> str:
    return f"{'abcdefgh'[c]}{GRID - r}"


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
    confidence: float

    def __str__(self) -> str:
        return f"{self.square} {self.truth}->{self.predicted}"


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
                    "confidence": round(e.confidence, 4),
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


@dataclass
class Report:
    boards: list[BoardResult]
    estimator: str
    bins: list[Bin] = field(default_factory=list)
    ece: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def perfect_boards(self) -> int:
        return sum(b.perfect for b in self.boards)

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

    truth = parse_placement(board.placement)
    idx = dist.argmax(axis=-1)
    conf = dist.max(axis=-1)
    predicted = [[CLASSES[int(idx[r, c])] for c in range(GRID)] for r in range(GRID)]
    errors = [
        SquareError(
            square=_square_name(r, c),
            truth=truth[r][c],
            predicted=predicted[r][c],
            kind=error_kind(truth[r][c], predicted[r][c]),
            confidence=float(conf[r, c]),
        )
        for r in range(GRID)
        for c in range(GRID)
        if truth[r][c] != predicted[r][c]
    ]
    return BoardResult(
        board_id=board.board_id,
        source=board.source,
        truth=board.placement,
        predicted=encode_placement(predicted),
        correct=GRID * GRID - len(errors),
        confidence=float(conf.mean()),
        errors=tuple(errors),
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
    estimator: object, boards: list[LabeledBoard], *, autorotate: bool = False
) -> Report:
    """Read every labeled board and assemble the report."""
    results: list[BoardResult] = []
    confidences: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    for board in boards:
        dist = read_board(estimator, board, autorotate=autorotate)
        result = score_board(board, dist)
        results.append(result)
        truth = parse_placement(board.placement)
        truth_idx = np.array(
            [[CLASSES.index(truth[r][c]) for c in range(GRID)] for r in range(GRID)]
        )
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
    json_out: str | None = None,
) -> Report:
    """Load labels, build the deployed estimator if none was injected, report."""
    from board_reader_inference.labels import load_labels

    candidates = [p for p in (paths or DEFAULT_PATHS) if Path(p).exists()]
    boards = load_labels(candidates)
    if estimator is None:
        from board_reader_inference.server import _default_estimator

        estimator = _default_estimator()
    report = evaluate(estimator, boards, autorotate=autorotate)
    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2))
    return report
