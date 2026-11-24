"""Move corner-frame labels into the chess frame (DESIGN D10). One-off, reviewable.

Before D10 a label's ``placement`` was written in the frame its photo's corners
define (corner 0 = a8), and operators tapped those corners at whichever board corner
was convenient, so a label can be a quarter- or half-turn off the chess frame. A
recognizer reads the chess frame, so ``eval`` skips such labels until they are
migrated. Migration changes **only the rotation**, never a square: the chess-frame
truth is ``rotate_placement(truth, k)`` for one ``k`` in 0..3.

**How ``k`` is chosen, and when it is not trusted.** ChessQ reads the photo; ``k``
is the turn under which the label agrees with that read on the most squares. A label
is migrated only when every check passes; otherwise it is left untouched and listed
for a human to look at — this tool never resolves a disagreement itself:

* **margin** — ``k`` must beat the runner-up turn by :data:`MARGIN_SQUARES` squares;
* **no pawn on rank 1 or 8** under ``k`` (a fact about chess, not a model guess);
* **the truth's own plausibility** — among the turns with fewest back-rank pawns,
  the one with White's men nearest rank 1, when that is unique, must be ``k``;
* **the old autorotate** — a label whose read was autorotated was confirmed as shown,
  turned ``autorotated``; that turn must be ``k`` too.

Originals are copied to ``<dir>/.pre-chess-frame/`` before anything is rewritten.
Nothing is written without ``--apply``.

    uv run python tools/migrate_labels_to_chess_frame.py migrate data/uploads
    uv run python tools/migrate_labels_to_chess_frame.py migrate data/uploads --apply
    uv run python tools/migrate_labels_to_chess_frame.py mark data/eval/ground_truth.json \\
        data/eval/heldout-20260831 --apply

``mark`` is for truth that is already in the chess frame (the curated eval stills):
it only adds ``frame: "chess"``, and refuses a board whose read agrees best turned.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from board_reader_core.protocol import CLASSES, parse_placement

from board_reader_inference.labels import (
    LABEL_SUFFIX,
    parse_corners,
    parse_frame,
    rotate_placement,
    rotate_square,
)

#: How many more squares the chosen turn must agree on than the runner-up. A right
#: turn agrees on ~60/64 squares; a wrong turn of a real position on far fewer.
MARGIN_SQUARES = 8

#: Where originals go, inside the directory of the file being rewritten.
BACKUP_DIR = ".pre-chess-frame"

DEFAULT_MODEL = "models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx"

#: A whole BGR photo -> its (8, 8, 13) chess-frame distribution (a recognizer's read).
Read = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Verdict:
    """What the checks say about one label."""

    name: str
    agree: tuple[int, int, int, int]  # squares the read shares with rotate(truth, k)
    rule: int | None  # the truth's own plausible turn, None if not unique
    shown: int  # the old autorotate turn, 0 if none
    back_rank_pawns: int  # under the chosen turn

    @property
    def rotation(self) -> int:
        return max(range(4), key=lambda k: (self.agree[k], k == 0))

    @property
    def margin(self) -> int:
        rest = [a for k, a in enumerate(self.agree) if k != self.rotation]
        return self.agree[self.rotation] - max(rest)

    @property
    def problems(self) -> tuple[str, ...]:
        k = self.rotation
        found = []
        if self.margin < MARGIN_SQUARES:
            found.append(f"margin {self.margin} < {MARGIN_SQUARES}")
        if self.back_rank_pawns:
            found.append(f"{self.back_rank_pawns} pawn(s) on rank 1/8 turned {90 * k}°")
        if self.rule is not None and self.rule != k:
            found.append(f"truth plausibility says {90 * self.rule}°")
        if self.shown and self.shown != k:
            found.append(f"old autorotate showed it turned {90 * self.shown}°")
        return tuple(found)

    def line(self) -> str:
        state = "ok  " if not self.problems else "LEFT"
        agree = "/".join(str(a) for a in self.agree)
        rule = "—" if self.rule is None else f"{90 * self.rule}"
        return (
            f"{state} {self.name:<40} turn {90 * self.rotation:>3}°  agree(0/90/180/270) "
            f"{agree:<15} margin {self.margin:>2}  rule {rule:>3}  shown {90 * self.shown:>3}"
            + (f"  ! {'; '.join(self.problems)}" if self.problems else "")
        )


def _grid(placement: str) -> np.ndarray:
    return np.array(parse_placement(placement), dtype="<U5")


def agreement(read: np.ndarray, truth: str) -> tuple[int, int, int, int]:
    """Squares the read's argmax shares with ``rotate_placement(truth, k)``, per ``k``."""
    got = np.array(CLASSES)[np.asarray(read).argmax(-1)]
    want = _grid(truth)
    a0, a1, a2, a3 = (int((np.rot90(want, k) == got).sum()) for k in range(4))
    return a0, a1, a2, a3


def back_rank_pawns(placement: str) -> int:
    g = _grid(placement)
    return int(np.isin(g[[0, 7]], ["P", "p"]).sum())


def plausible_rotation(truth: str) -> int | None:
    """The turn that puts no pawn on rank 1/8 and White nearest rank 1, if unique."""

    def white_lower(k: int) -> float:
        g = np.rot90(_grid(truth), k)
        rows = np.indices(g.shape)[0]
        white = rows[np.char.isupper(g) & (g != "empty")]
        black = rows[np.char.islower(g) & (g != "empty")]
        if not white.size or not black.size:
            return 0.0
        return float(white.mean() - black.mean())  # > 0: White sits lower (rank 1)

    fewest = min(back_rank_pawns(rotate_placement(truth, k)) for k in range(4))
    candidates = [k for k in range(4) if back_rank_pawns(rotate_placement(truth, k)) == fewest]
    best = max(white_lower(k) for k in candidates)
    winners = [k for k in candidates if white_lower(k) == best]
    return winners[0] if len(winners) == 1 and best > 0 else None


def judge(name: str, read: np.ndarray, truth: str, *, autorotated: int) -> Verdict:
    agree = agreement(read, truth)
    k = max(range(4), key=lambda i: (agree[i], i == 0))
    return Verdict(
        name=name,
        agree=agree,
        rule=plausible_rotation(truth),
        shown=(autorotated // 90) % 4,
        back_rank_pawns=back_rank_pawns(rotate_placement(truth, k)),
    )


def rotate_corners(corners: np.ndarray, k: int) -> np.ndarray:
    """Relabel a8,h8,h1,a1 image corners for ``rotate_placement(_, k)``.

    One ``np.rot90`` turn is counter-clockwise, so the new a8 (top-left) is the old
    h8 (top-right): the corner list rolls back by ``k``.
    """
    return np.roll(np.asarray(corners), -k, axis=0)


def _corner_string(corners: np.ndarray) -> str:
    return ",".join(f"{v:g}" for v in np.asarray(corners, dtype=np.float64).reshape(-1))


def migrated(meta: dict, verdict: Verdict) -> dict:
    """``meta`` (a corner-frame ``.label.json``) rewritten in the chess frame."""
    k = verdict.rotation
    out = dict(meta)
    out["frame"] = "chess"
    out["placement"] = rotate_placement(meta["placement"], k)
    if meta.get("corners"):
        out["corners"] = _corner_string(rotate_corners(parse_corners(meta["corners"]), k))
    out["prior_flips"] = [rotate_square(s, k) for s in meta.get("prior_flips") or []]
    out["migrated"] = {
        "from_frame": "corners",
        "rotation_deg": 90 * k,
        "agree": list(verdict.agree),
        "margin": verdict.margin,
        "tool": "tools/migrate_labels_to_chess_frame.py",
    }
    return out


def _backup(path: Path) -> None:
    dest = path.parent / BACKUP_DIR / path.name
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():  # a re-run never overwrites the true original
        shutil.copy2(path, dest)


def _image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read {path}")
    return image


def migrate_dir(directory: Path, read: Read, *, apply: bool) -> list[Verdict]:
    """Judge every corner-frame label in ``directory``; rewrite the clean ones."""
    verdicts = []
    for path in sorted(directory.glob(f"*{LABEL_SUFFIX}")):
        meta = json.loads(path.read_text())
        if parse_frame(meta.get("frame")) == "chess":
            continue
        name = path.name[: -len(LABEL_SUFFIX)]
        verdict = judge(
            name,
            read(_image(path.parent / meta["photo"])),
            meta["placement"],
            autorotated=int(meta.get("autorotated") or 0),
        )
        verdicts.append(verdict)
        if apply and not verdict.problems:
            _backup(path)
            path.write_text(json.dumps(migrated(meta, verdict)))
    return verdicts


def _mark_entries(path: Path, data: dict) -> Iterator[tuple[str, dict, Path, str]]:
    """``(name, entry, image, placement)`` in a label file's or a ground_truth.json's
    parsed ``data``; each ``entry`` is the dict inside ``data`` to mark."""
    if path.name.endswith(LABEL_SUFFIX):
        yield path.name[: -len(LABEL_SUFFIX)], data, path.parent / data["photo"], data["placement"]
        return
    for board_id, entry in data.items():
        yield board_id, entry, Path(str(entry["source"])).expanduser(), entry["placement"]


def mark(paths: list[Path], read: Read, *, apply: bool) -> list[Verdict]:
    """Add ``frame: "chess"`` to truth that is already chess-frame, board by board."""
    files: list[Path] = []
    for p in paths:
        files += sorted(p.glob(f"*{LABEL_SUFFIX}")) if p.is_dir() else [p]
    verdicts = []
    for path in files:
        data = json.loads(path.read_text())
        changed = False
        for name, entry, image, placement in _mark_entries(path, data):
            verdict = judge(name, read(_image(image)), placement, autorotated=0)
            verdicts.append(verdict)
            if not verdict.problems and verdict.rotation == 0 and entry.get("frame") != "chess":
                entry["frame"] = "chess"
                changed = True
        if apply and changed:
            _backup(path)
            path.write_text(json.dumps(data, indent=1))
    return verdicts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("migrate", "mark"))
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="ChessQ ONNX (the recognizer)")
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    args = ap.parse_args(argv)

    from board_reader_inference.chessq import ChessQRecognizer

    read = ChessQRecognizer(args.model).read
    if args.mode == "migrate":
        verdicts = [v for d in args.paths for v in migrate_dir(d, read, apply=args.apply)]
    else:
        verdicts = mark(args.paths, read, apply=args.apply)
    for v in verdicts:
        print(v.line())
    if args.mode == "mark":
        left = [v for v in verdicts if v.problems or v.rotation]
    else:
        left = [v for v in verdicts if v.problems]
    done = "migrated" if args.mode == "migrate" else "marked chess"
    verb = done if args.apply else f"would be {done}"
    print(f"\n{len(verdicts) - len(left)}/{len(verdicts)} {verb}; {len(left)} left for a human")
    return 0


if __name__ == "__main__":
    sys.exit(main())
