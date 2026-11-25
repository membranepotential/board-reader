"""How a square went wrong: the error taxonomy shared by every scorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board_reader_core.protocol import GRID, parse_placement, square_name


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
