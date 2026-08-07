"""Per-square accuracy split into the categories this project cares about.

Raw per-square accuracy hides *what kind* of error the model makes, and the design
(DESIGN §4.6) tracks three distinct failure modes with different downstream costs:

* ``empty->piece`` — hallucinating a piece on an empty square (the worst for move
  reconstruction; a phantom piece),
* ``piece->empty`` — missing a real piece,
* ``piece->piece`` — right occupancy, wrong type/colour (the style-transfer weak
  spot that motivated multi-style training in the first place).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import _bootstrap  # noqa: F401
from board_reader_core.protocol import EMPTY, NUM_CLASSES


@dataclass
class SquareStats:
    total: int = 0
    correct: int = 0
    empty_total: int = 0
    empty_correct: int = 0  # empty predicted empty
    empty_to_piece: int = 0  # hallucination
    piece_total: int = 0
    piece_to_empty: int = 0  # missed
    piece_correct: int = 0  # right type+colour
    piece_to_piece: int = 0  # wrong type/colour but occupied
    confusion: np.ndarray = field(  # (true, pred) counts over CLASSES order
        default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    )

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        true = np.asarray(true)
        pred = np.asarray(pred)
        np.add.at(self.confusion, (true, pred), 1)
        self.total += true.size
        self.correct += int((true == pred).sum())
        t_empty = true == EMPTY
        p_empty = pred == EMPTY
        self.empty_total += int(t_empty.sum())
        self.empty_correct += int((t_empty & p_empty).sum())
        self.empty_to_piece += int((t_empty & ~p_empty).sum())
        t_piece = ~t_empty
        self.piece_total += int(t_piece.sum())
        self.piece_to_empty += int((t_piece & p_empty).sum())
        self.piece_correct += int((t_piece & (true == pred)).sum())
        self.piece_to_piece += int((t_piece & ~p_empty & (true != pred)).sum())

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

    @property
    def occupancy_accuracy(self) -> float:
        """Occupied-vs-empty accuracy (ignores piece type) — the style-robust signal."""
        occ_correct = self.empty_correct + (self.piece_total - self.piece_to_empty)
        return occ_correct / max(1, self.total)

    def summary(self) -> str:
        return (
            f"acc {self.accuracy:6.3f} occ {self.occupancy_accuracy:6.3f} | "
            f"empty {self.empty_correct}/{self.empty_total} "
            f"(halluc {self.empty_to_piece}) | "
            f"piece ok {self.piece_correct}/{self.piece_total} "
            f"(missed {self.piece_to_empty}, wrongtype {self.piece_to_piece})"
        )

    def to_dict(self) -> dict:
        return {
            "n_squares": self.total,
            "accuracy": round(self.accuracy, 4),
            "occupancy_accuracy": round(self.occupancy_accuracy, 4),
            "empty_total": self.empty_total,
            "empty_correct": self.empty_correct,
            "empty_to_piece": self.empty_to_piece,
            "piece_total": self.piece_total,
            "piece_correct": self.piece_correct,
            "piece_to_empty": self.piece_to_empty,
            "piece_to_piece": self.piece_to_piece,
            "confusion": self.confusion.tolist(),
        }
