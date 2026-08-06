"""Build a DINOv2 kNN exemplar bank from the public ChessReD2K dataset.

This seeds the shipped recognition bank so square recognition works *out of the
box* — no per-user calibration — by drawing exemplars from a diverse public
dataset instead of only the operator's own labeled photos.

Data: ChessReD2K (Masouris & van Gemert, 4TU.ResearchData, CC BY-NC-SA 4.0), the
corner-annotated 20-game subset. Download ``annotations.json`` + ``chessred2k.zip``
into ``data/external/chessred/`` (see tools/README or the dataset page
https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f) and unzip.

Geometry, VALIDATED against the annotations (see module docstring of the eval
script): each corner record names the four board corners
``top_left / top_right / bottom_right / bottom_left`` in a **board-anchored** frame
(top_left is always the a8 corner, independent of camera angle). That maps to our
semantic ``corners_a8_h8_h1_a1`` order as ``[top_left, top_right, bottom_right,
bottom_left]`` — projecting every labeled piece through this homography lands 98.9%
of them in their annotated square (the residual is the base-point heuristic near
square edges, not a mapping error).

Boxes are cut with the LOO-validated photo geometry (headroom 0.15, side_pad 0.15),
matching ``tools/build_exemplar_bank.py`` and the gateway defaults, so a query frame
is cropped exactly as these exemplars were.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from board_reader_core.protocol import CLASS_INDEX, CLASSES, GRID
from board_reader_inference.centroid import photometric_variants
from board_reader_inference.embed import DinoV2Embedder, l2_normalize
from board_reader_inference.squares import crop_squares_photo

ROOT = Path(__file__).resolve().parent.parent
CHESSRED = ROOT / "data/external/chessred"
HEADROOM, SIDE_PAD, TEMPERATURE = 0.15, 0.15, 0.1

# ChessReD category id -> our CLASSES symbol. Their ids: white pawn..king 0-5,
# black pawn..king 6-11, empty 12 (empties are implicit — never listed as pieces).
CHESSRED_TO_CLASS: dict[int, str] = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
    12: "empty",
}

# ChessReD corner keys in our semantic a8,h8,h1,a1 order (validated, see docstring).
CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


class ChessReD:
    """Thin accessor over the ChessReD ``annotations.json`` for the 2K subset."""

    def __init__(self, root: Path = CHESSRED) -> None:
        self.root = root
        with open(root / "annotations.json") as f:
            ann = json.load(f)
        self.images = {im["id"]: im for im in ann["images"]}
        self.corners = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
        self._pieces: dict[int, list[dict]] = defaultdict(list)
        for p in ann["annotations"]["pieces"]:
            self._pieces[p["image_id"]].append(p)
        self.splits = ann["splits"]["chessred2k"]

    def subset_ids(self, splits: tuple[str, ...] = ("train", "val", "test")) -> list[int]:
        ids: list[int] = []
        for s in splits:
            ids.extend(self.splits[s]["image_ids"])
        return sorted(ids)

    def game_id(self, image_id: int) -> int:
        return self.images[image_id]["game_id"]

    def corner_quad(self, image_id: int) -> np.ndarray:
        """Corners in our semantic a8,h8,h1,a1 pixel order."""
        c = self.corners[image_id]
        return np.array([c[k] for k in CORNER_KEYS], dtype=np.float32)

    def labels(self, image_id: int) -> list[str]:
        """64 square labels in crop order (index r*8+c = rank 8-r, file a+c)."""
        grid = ["empty"] * (GRID * GRID)
        for p in self._pieces[image_id]:
            pos = p["chessboard_position"]  # e.g. "a8"
            file_c = ord(pos[0]) - ord("a")
            rank = int(pos[1])
            r = GRID - rank  # rank 8 -> row 0
            grid[r * GRID + file_c] = CHESSRED_TO_CLASS[p["category_id"]]
        return grid

    def image(self, image_id: int) -> np.ndarray:
        path = self.root / self.images[image_id]["path"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        return img


def sample_boards(
    ds: ChessReD, ids: list[int], per_game: int, *, seed: int = 0
) -> list[int]:
    """Pick ``per_game`` boards per game, spread evenly across move progression.

    Even spacing over the game's move order samples the full occupancy range
    (crowded openings -> sparse endgames), which is what gives class diversity;
    each ChessReD game is a single fixed camera, so angle diversity comes from
    spanning games, not from sampling more within one.
    """
    by_game: dict[int, list[int]] = defaultdict(list)
    for iid in ids:
        by_game[ds.game_id(iid)].append(iid)
    picked: list[int] = []
    for game, gids in sorted(by_game.items()):
        gids = sorted(gids, key=lambda i: ds.images[i]["move_id"])
        if per_game >= len(gids):
            picked.extend(gids)
            continue
        idx = np.linspace(0, len(gids) - 1, per_game).round().astype(int)
        picked.extend(gids[i] for i in sorted(set(idx.tolist())))
    return picked


def build_bank(
    ds: ChessReD,
    board_ids: list[int],
    embedder: DinoV2Embedder,
    *,
    augment: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Crop + embed every square of every board -> (exemplars, labels)."""
    crops: list[np.ndarray] = []
    labels: list[str] = []
    for n, iid in enumerate(board_ids, 1):
        img = ds.image(iid)
        sq = crop_squares_photo(img, ds.corner_quad(iid), headroom=HEADROOM, side_pad=SIDE_PAD)
        for crop, lab in zip(sq, ds.labels(iid)):
            variants = photometric_variants(crop) if augment else [crop]
            crops.extend(variants)
            labels.extend([lab] * len(variants))
        if n % 20 == 0 or n == len(board_ids):
            print(f"  cropped {n}/{len(board_ids)} boards -> {len(crops)} crops", flush=True)
    print(f"  embedding {len(crops)} crops on CPU ...", flush=True)
    exemplars = l2_normalize(embedder.embed(crops), axis=1)
    return exemplars.astype(np.float32), labels


def save_bank(path: Path, exemplars: np.ndarray, labels: list[str]) -> None:
    bad = sorted({lab for lab in labels if lab not in CLASS_INDEX})
    if bad:
        raise ValueError(f"unknown labels: {bad}")
    np.savez(
        path,
        kind=np.array("exemplar"),
        exemplars=exemplars,
        labels=np.array(labels),
        classes=np.array(CLASSES),
        temperature=np.float64(TEMPERATURE),
        backbone=np.array("dinov2"),
        crop_mode=np.array("photo"),
        box_headroom=np.float64(HEADROOM),
        box_side_pad=np.float64(SIDE_PAD),
    )
    mb = path.stat().st_size / 1e6
    print(f"wrote {path}: {exemplars.shape[0]} exemplars, dim {exemplars.shape[1]} ({mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-game", type=int, default=20, help="boards sampled per game")
    ap.add_argument("--augment", action="store_true", help="add photometric variants")
    ap.add_argument("--out", type=Path, default=ROOT / "models/exemplars_chessred.npz")
    ap.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="chessred2k splits to draw boards from",
    )
    args = ap.parse_args()

    ds = ChessReD()
    ids = ds.subset_ids(tuple(args.splits))
    boards = sample_boards(ds, ids, args.per_game)
    n_games = len({ds.game_id(i) for i in boards})
    print(f"sampling {len(boards)} boards across {n_games} games "
          f"(per_game={args.per_game}, augment={args.augment})")
    embedder = DinoV2Embedder(ROOT / "models/dinov2_small.onnx", providers=["CPUExecutionProvider"])
    exemplars, labels = build_bank(ds, boards, embedder, augment=args.augment)
    save_bank(args.out, exemplars, labels)


if __name__ == "__main__":
    main()
