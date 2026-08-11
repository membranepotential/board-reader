"""Repair rotated corner order in the ``data/uploads`` calibrate sidecars.

Corner order is **semantic**: the four points are ``a8, h8, h1, a1``, so naming the
wrong point ``a8`` rotates every label 90/180/270 degrees while the geometry still
grid-aligns perfectly. Twenty of the non-user boards in ``data/uploads`` were stored
that way, which floored every model's ``uploads_calibrate`` score at ~0.59 — a label
bug wearing a model bug's clothes. Same failure and same fix as
``data/external/chesscog_transfer/repair_orientation.py`` (commit ``f362f78``); this
script is that method applied to the uploads sidecars, plus a second, independent
opinion.

Every calibrate frame is a **start position**, which makes two unrelated channels
available, and a repair is written only where they agree:

1. **Photometric** (no model): a8 is a light square, so with the corners rolled
   correctly the ``(row + col)``-even cells the label calls *empty* must be lighter
   than the odd ones — that settles 90 vs 180/270. Light-vs-dark piece wood then
   settles 0 vs 180, each occupied cell read against the median of the *empty* cells
   sharing its square colour so the checkerboard cancels out.
2. **Model** (no photometry): score the known start placement under all four
   rotations with a trained classifier, on *side* (empty / white / black) rather than
   exact piece type. Side is what a rotation actually scrambles, and a model that
   cannot name a piece on an unfamiliar set can still see its colour, so this channel
   keeps working where the full 13-class score collapses to noise. A start position's
   occupancy is 180-symmetric, so the wrong rotations land near 22-37/64 while the
   right one clears 48.

Both must be conclusive *and* agree. Disagreements are reported and left alone, and
so is anything in :data:`PROTECTED` — the user's own board, whose sidecars are the
calibration/fine-tune split and were verified sound. Sound boards are the sanity
gate: they must come back as rotation 0 on both channels.

Known limit, worth reading before trusting a run: channel 1 assumes the board obeys
the a8-light convention. A physical board set up a quarter-turn off does not, and
then the parity test measures piece contamination instead of squares. Such a board
should fail the piece-wood gate and be reported inconclusive rather than repaired —
the per-rotation margins are printed so that case is recognisable.

    uv run python tools/repair_upload_orientation.py --dry-run
    uv run python tools/repair_upload_orientation.py            # writes + backs up
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
import onnxruntime as ort

from board_reader_core.protocol import (
    CLASS_INDEX,
    CLASSES,
    EMPTY,
    GRID,
    START_PLACEMENT,
    parse_placement,
)

from board_reader_inference.squares import (
    CropGeometry,
    crop_squares_photo,
    geometry_from_dict,
    rectify_photo,
)

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "data/uploads"
DEFAULT_MODEL = ROOT / "scratch/runs/run_e/model.onnx"
BACKUP_SUFFIX = ".pre-rotation-repair"

# Frames of the user's own physical board. They are the calibrate/fine-tune split
# (scratch/experiments/calib_finetune/split_manifest.json) and score 59-63/64 as
# stored, so their labels are known good: analyse them as a sanity gate, never write.
PROTECTED: frozenset[str] = frozenset(
    {
        "20260717-232704-0011-calibrate",
        "20260717-232806-0018-calibrate",
        "20260717-233102-0022-calibrate",
        "20260718-155204-0010-calibrate",
        "20260806-232827-0008-calibrate",
    }
)

Rotation: TypeAlias = Literal[0, 1, 2, 3]
ROTATIONS: tuple[Rotation, ...] = (0, 1, 2, 3)

Action: TypeAlias = Literal["repaired", "unchanged", "protected", "disagree", "inconclusive"]

WARP = 512
CELL = WARP // GRID
INSET = 16  # px trimmed off each cell edge before measuring (drops the grid lines)

# A real board separates light from dark squares, and light from dark piece wood, by
# far more than this; a wrong rotation lands at or below zero on the parity channel.
MIN_PARITY_MARGIN = 20.0
MIN_PIECE_MARGIN = 10.0

# The model channel only gets a vote when one rotation is unambiguously the position.
# Measured over these 35 boards, the right rotation scores 59-64 on side and the
# runner-up 32-37; the one board where the channel is genuinely blind peaks at 38.
MIN_MODEL_SIDE = 48
MIN_MODEL_LEAD = 15

SQUARE_SIZE = 224
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class Board:
    """One calibrate sidecar: the photo, its stored corner order, its placement."""

    stem: str
    sidecar: Path
    image_path: Path
    corners: np.ndarray  # (4, 2), a8/h8/h1/a1 as stored
    placement: str

    @property
    def protected(self) -> bool:
        return self.stem in PROTECTED

    def rotated(self, turns: Rotation) -> np.ndarray:
        """The stored quad rolled so that index 0 is the corner ``turns`` along."""
        return np.roll(self.corners, -turns, axis=0).astype(np.float32)


Quad = tuple[float, float, float, float]
QuadInt = tuple[int, int, int, int]


@dataclass(frozen=True)
class Photometric:
    """Rotation chosen from pixels alone, with the margins of all four rotations."""

    rotation: Rotation | None
    parity: Quad
    piece: Quad

    @property
    def margins(self) -> tuple[float, float]:
        """The two margins of the chosen rotation (zeros when nothing was chosen)."""
        k = self.rotation
        return (0.0, 0.0) if k is None else (self.parity[k], self.piece[k])


@dataclass(frozen=True)
class ModelVote:
    """Rotation chosen by scoring the known placement under all four rotations.

    ``side`` (empty/white/black) is what decides; ``exact`` is the full 13-class
    score, carried for the report because it is the number the eval harness quotes.
    """

    rotation: Rotation | None
    side: QuadInt
    exact: QuadInt


@dataclass(frozen=True)
class Verdict:
    board: Board
    photometric: Photometric
    model: ModelVote
    action: Action
    reason: str


def load_boards(uploads: Path = UPLOADS) -> list[Board]:
    """Every calibrate frame that ``tools/train/evaluate.py`` would load, same rules."""
    boards: list[Board] = []
    for sidecar in sorted(uploads.glob("*-calibrate.json")):
        meta = json.loads(sidecar.read_text())
        image_path = sidecar.with_suffix(".jpg")
        if not image_path.exists() or "corners" not in meta:
            continue
        nums = [float(v) for v in str(meta["corners"]).split(",")]
        if len(nums) != 8:
            continue
        boards.append(
            Board(
                stem=sidecar.stem,
                sidecar=sidecar,
                image_path=image_path,
                corners=np.asarray(nums, dtype=np.float32).reshape(4, 2),
                placement=str(meta.get("placement", START_PLACEMENT)).split()[0],
            )
        )
    return boards


# --- channel 1: photometry ---------------------------------------------------


def _cell_luminance(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """(8, 8) median grey level of each cell's interior, in protocol order."""
    warp = rectify_photo(image, corners, WARP)
    grey = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lum = np.empty((GRID, GRID), dtype=np.float32)
    for r in range(GRID):
        for c in range(GRID):
            patch = grey[
                r * CELL + INSET : (r + 1) * CELL - INSET,
                c * CELL + INSET : (c + 1) * CELL - INSET,
            ]
            lum[r, c] = float(np.median(patch))
    return lum


def parity_margin(lum: np.ndarray, grid: list[list[str]]) -> float:
    """Mean luminance of light-parity empty cells minus dark-parity ones.

    Positive iff a8 (row 0, col 0) really is the light corner square. Only cells the
    label calls empty are measured, so pieces cannot bias it.
    """
    sides: list[list[float]] = [[], []]
    for r in range(GRID):
        for c in range(GRID):
            if grid[r][c] == "empty":
                sides[(r + c) % 2].append(float(lum[r, c]))
    if not sides[0] or not sides[1]:
        return 0.0
    return float(np.mean(sides[0]) - np.mean(sides[1]))


def piece_margin(lum: np.ndarray, grid: list[list[str]]) -> float:
    """Mean brightness of white-labelled pieces minus black-labelled ones.

    Each occupied cell is measured against the median of the *empty* cells that share
    its square colour, so the checkerboard cancels and only the piece's own wood tone
    remains.
    """
    baseline: dict[int, float] = {}
    for par in (0, 1):
        empties = [
            float(lum[r, c])
            for r in range(GRID)
            for c in range(GRID)
            if (r + c) % 2 == par and grid[r][c] == "empty"
        ]
        baseline[par] = float(np.median(empties)) if empties else float(np.median(lum))
    white: list[float] = []
    black: list[float] = []
    for r in range(GRID):
        for c in range(GRID):
            piece = grid[r][c]
            if piece == "empty":
                continue
            dev = float(lum[r, c]) - baseline[(r + c) % 2]
            (white if piece.isupper() else black).append(dev)
    if not white or not black:
        return 0.0
    return float(np.mean(white) - np.mean(black))


def photometric_vote(image: np.ndarray, board: Board) -> Photometric:
    """Pick the rotation whose pixels agree with the label's colours."""
    grid = parse_placement(board.placement)
    margins = [
        (lambda lum: (parity_margin(lum, grid), piece_margin(lum, grid)))(
            _cell_luminance(image, board.rotated(turns))
        )
        for turns in ROTATIONS
    ]
    parity, piece = (tuple(m[0] for m in margins), tuple(m[1] for m in margins))
    # Parity gates first (it leaves only two rotations standing); wood then decides.
    best = max(ROTATIONS, key=lambda k: (parity[k] > MIN_PARITY_MARGIN, piece[k]))
    conclusive = parity[best] >= MIN_PARITY_MARGIN and piece[best] >= MIN_PIECE_MARGIN
    return Photometric(best if conclusive else None, parity, piece)  # type: ignore[arg-type]


# --- channel 2: the model ----------------------------------------------------


@dataclass(frozen=True)
class Classifier:
    """A trained square classifier plus the crop geometry it was trained on."""

    session: ort.InferenceSession
    input_name: str
    geometry: CropGeometry

    @classmethod
    def load(cls, onnx_path: Path, *, threads: int = 6) -> Classifier:
        sidecar = Path(str(onnx_path) + ".meta.json")
        if not sidecar.exists():  # geometry travels with the model, never with the caller
            raise SystemExit(f"no geometry sidecar at {sidecar}")
        meta = json.loads(sidecar.read_text())
        geometry = geometry_from_dict(meta.get("crop_geometry", meta))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        return cls(session, session.get_inputs()[0].name, geometry)

    def predict(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """(64,) class indices for the board under this corner order, crop order."""
        crops = crop_squares_photo(image, corners, geometry=self.geometry, size=SQUARE_SIZE)
        batch = np.stack([_prep(c) for c in crops]).astype(np.float32)
        return self.session.run(None, {self.input_name: batch})[0].argmax(1)


def _prep(crop: np.ndarray) -> np.ndarray:
    """Byte-identical to ``evaluate.OnnxPredictor._prep`` (colour path)."""
    rgb = cv2.resize(crop[:, :, ::-1], (SQUARE_SIZE, SQUARE_SIZE), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def labels_of(placement: str) -> np.ndarray:
    grid = parse_placement(placement)
    return np.array([CLASS_INDEX[c] for row in grid for c in row], dtype=np.int64)


# empty -> 0, white piece -> 1, black piece -> 2.
_SIDE_OF_CLASS = np.array(
    [0 if i == EMPTY else (1 if c.isupper() else 2) for i, c in enumerate(CLASSES)], dtype=np.int64
)


def model_vote(image: np.ndarray, board: Board, clf: Classifier) -> ModelVote:
    """Pick the rotation under which the model most nearly reads the known position."""
    labels = labels_of(board.placement)
    sides = _SIDE_OF_CLASS[labels]
    scores = [
        (lambda p: (int((_SIDE_OF_CLASS[p] == sides).sum()), int((p == labels).sum())))(
            clf.predict(image, board.rotated(turns))
        )
        for turns in ROTATIONS
    ]
    side, exact = tuple(s[0] for s in scores), tuple(s[1] for s in scores)
    best, runner_up = sorted(ROTATIONS, key=lambda k: side[k], reverse=True)[:2]
    conclusive = side[best] >= MIN_MODEL_SIDE and side[best] - side[runner_up] >= MIN_MODEL_LEAD
    return ModelVote(best if conclusive else None, side, exact)  # type: ignore[arg-type]


# --- adjudication ------------------------------------------------------------


def adjudicate(board: Board, photo: Photometric, model: ModelVote) -> Verdict:
    def verdict(action: Action, reason: str) -> Verdict:
        return Verdict(board, photo, model, action, reason)

    if photo.rotation is None or model.rotation is None:
        which = " and ".join(
            n for n, v in (("photometric", photo.rotation), ("model", model.rotation)) if v is None
        )
        return verdict("inconclusive", f"{which} inconclusive")
    if photo.rotation != model.rotation:
        return verdict("disagree", f"photometric {photo.rotation * 90}deg vs model {model.rotation * 90}deg")
    if photo.rotation == 0:
        return verdict("unchanged", "both channels say the stored order is already a8-first")
    if board.protected:
        return verdict("protected", f"user board — would be {photo.rotation * 90}deg, left untouched")
    return verdict("repaired", f"rolled {photo.rotation * 90}deg to put a8 first")


def apply_repair(verdict: Verdict, *, backup_suffix: str = BACKUP_SUFFIX) -> Path:
    """Back the sidecar up, then rewrite its ``corners`` in a8-first order."""
    rotation = verdict.photometric.rotation
    if verdict.action != "repaired" or rotation is None:
        raise ValueError(f"{verdict.board.stem}: not a repair ({verdict.action})")
    sidecar = verdict.board.sidecar
    backup = sidecar.with_name(sidecar.name + backup_suffix)
    if not backup.exists():  # never overwrite the pristine original on a second run
        shutil.copy2(sidecar, backup)
    meta = json.loads(sidecar.read_text())
    corners = verdict.board.rotated(rotation)
    parity, piece = verdict.photometric.margins
    meta["corners"] = ",".join(f"{v:.0f}" for v in corners.reshape(-1))
    meta["rotation_repair"] = {
        "turns": rotation,
        "original_corners": str(json.loads(backup.read_text())["corners"]),
        "photometric_margins": {"parity": round(parity, 1), "piece": round(piece, 1)},
        "model_side_of_64_per_rotation": list(verdict.model.side),
    }
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    return backup


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--uploads", type=Path, default=UPLOADS, help="directory of calibrate sidecars")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="ONNX square classifier")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--json", type=Path, help="also write the full per-board verdicts here")
    args = ap.parse_args()

    boards = load_boards(args.uploads)
    clf = Classifier.load(args.model)
    print(f"{len(boards)} calibrate boards | model {args.model} | geometry {clf.geometry}\n")

    verdicts: list[Verdict] = []
    for board in boards:
        image = cv2.imread(str(board.image_path))
        if image is None:
            print(f"  WARN unreadable {board.image_path}")
            continue
        photo = photometric_vote(image, board)
        model = model_vote(image, board, clf)
        v = adjudicate(board, photo, model)
        verdicts.append(v)
        print(
            f"  {board.stem:32s} parity {_row(photo.parity)} piece {_row(photo.piece)} "
            f"-> {_show(photo.rotation):>3s} | side {_row(model.side)} "
            f"-> {_show(model.rotation):>3s} | {v.action.upper()}: {v.reason}"
        )

    by_action: dict[Action, list[Verdict]] = {}
    for v in verdicts:
        by_action.setdefault(v.action, []).append(v)
    print("\n" + " | ".join(f"{a} {len(vs)}" for a, vs in sorted(by_action.items())))

    # Sanity gate: whatever we already believed was correctly labelled must survive
    # both channels as rotation 0. If a known-good board wants rotating, the method
    # is wrong and nothing should be written.
    bad = [v for v in verdicts if v.board.protected and v.action != "unchanged"]
    if bad:
        print("\nSANITY GATE FAILED — protected boards did not come back unrotated:")
        for v in bad:
            print(f"  {v.board.stem}: {v.action} ({v.reason})")
        return 1
    print("sanity gate: PASS — every protected board reads as rotation 0 on both channels")

    if args.json:
        args.json.write_text(json.dumps([_as_dict(v) for v in verdicts], indent=2) + "\n")
        print(f"wrote {args.json}")

    repairs = by_action.get("repaired", [])
    if args.dry_run:
        print(f"\n(dry run — {len(repairs)} sidecars would be rewritten, none touched)")
        return 0
    for v in repairs:
        backup = apply_repair(v)
        print(f"  repaired {v.board.sidecar.name} (backup {backup.name})")
    print(f"\n{len(repairs)} sidecars repaired")
    return 0


def _show(rotation: Rotation | None) -> str:
    return "??" if rotation is None else f"{rotation * 90}"


def _row(values: tuple[float, ...] | tuple[int, ...]) -> str:
    """The per-rotation series, printed so 0/90/180/270 line up down the log."""
    return "[" + " ".join(f"{v:5.0f}" for v in values) + "]"


def _as_dict(v: Verdict) -> dict[str, object]:
    return {
        "board": v.board.stem,
        "protected": v.board.protected,
        "action": v.action,
        "reason": v.reason,
        "photometric": {
            "rotation": v.photometric.rotation,
            "parity_margin_per_rotation": [round(x, 2) for x in v.photometric.parity],
            "piece_margin_per_rotation": [round(x, 2) for x in v.photometric.piece],
        },
        "model": {
            "rotation": v.model.rotation,
            "side_of_64_per_rotation": list(v.model.side),
            "exact_of_64_per_rotation": list(v.model.exact),
        },
        "corners_stored": ",".join(f"{x:.0f}" for x in v.board.corners.reshape(-1)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
