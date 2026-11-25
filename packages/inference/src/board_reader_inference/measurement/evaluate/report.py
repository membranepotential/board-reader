"""The evaluation's report: headline numbers, as JSON and as the text ``eval`` prints."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from board_reader_core.protocol import GRID

from board_reader_inference.measurement.evaluate.board import BoardResult
from board_reader_inference.measurement.evaluate.confidence import Bin, TemperatureReport


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
