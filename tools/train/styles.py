"""Style registry: every training/eval data source behind one uniform interface.

A **style** = one physical/rendered chess set (its own piece shapes, colours,
materials, board). Generalising to UNSEEN styles is the whole point of the harness
(frozen DINOv2 features proved style-entangled), so the unit of the leave-one-out
protocol is the style, and every source — real datasets, synthetic renders, the
user's own set — is normalised to a common :class:`StyleSample` stream here.

The load-bearing contract (produced by the data agents) is
``labels_normalized.json``: a list of ``{"image": relpath, "placement":
<FEN field 1>, "corners_a8_h8_h1_a1": [[x,y]x4]}``. :func:`discover_styles` finds
every such file under ``data/external`` and exposes it as a :class:`Style`.
ChessReD ships its own ``annotations.json`` instead, so it gets a small adapter
(:func:`chessred_style`) that emits the same :class:`StyleSample` shape.

Each sample also carries a ``group`` key (e.g. ChessReD game id) so a single-style
smoke run can still hold out unseen *boards/cameras* even when it can't hold out an
unseen style.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts board_reader_* on sys.path)
from board_reader_core.protocol import CLASS_INDEX, GRID, encode_placement

ROOT = _bootstrap.ROOT
EXTERNAL = ROOT / "data" / "external"

# Modes a Style can be in: "board" samples are full photos needing crop_squares_photo;
# "crop" samples are already-cut single-square images (a piece-classification dataset).
BOARD, CROP = "board", "crop"


@dataclass(frozen=True)
class CropSample:
    """One already-cropped square image + its class (crop-level datasets)."""

    image_path: Path
    label: str  # a CLASSES symbol
    group: str = ""


@dataclass(frozen=True)
class StyleSample:
    """One labelled board photo: the input to :func:`crop_squares_photo`.

    ``corners`` are the playing-surface corners in *original image* pixels, order
    a8, h8, h1, a1 (the deployed convention). ``placement`` is FEN field 1.
    ``group`` clusters samples that share a camera/game so a validation split can
    keep them together (avoids leaking the same viewpoint across train/val).
    """

    image_path: Path
    placement: str
    corners: np.ndarray  # (4, 2) float32, a8,h8,h1,a1
    group: str = ""

    def __post_init__(self) -> None:
        c = np.asarray(self.corners, dtype=np.float32).reshape(4, 2)
        object.__setattr__(self, "corners", c)


@dataclass
class Style:
    """A named collection of labelled data from one chess set.

    ``mode`` selects how :mod:`cropcache` turns ``_samples`` into 224x224 crops:
    ``board`` samples (:class:`StyleSample`) are photos cut with the deployed
    :func:`crop_squares_photo` geometry; ``crop`` samples (:class:`CropSample`) are
    already square-cut piece images used as-is.
    """

    name: str
    kind: str  # "real" | "synthetic"
    _samples: list  # list[StyleSample] (board) | list[CropSample] (crop)
    mode: str = BOARD

    def __len__(self) -> int:
        return len(self._samples)

    def samples(self) -> Iterator:
        return iter(self._samples)

    @property
    def n_groups(self) -> int:
        return len({s.group for s in self._samples})


# --------------------------------------------------------------------------- #
# Normalized-contract loader (chesscog renders, Blender styles, anything the
# data agents emit as labels_normalized.json).
# --------------------------------------------------------------------------- #
NORMALIZED_MANIFEST = "labels_normalized.json"

# Crop-level piece codes -> our CLASSES symbols. Datasets use FEN letters directly
# (upper=white, lower=black); an absent/None piece is an empty square.
_PIECE_TO_CLASS = {c: c for c in CLASS_INDEX}
_PIECE_TO_CLASS.update({None: "empty", "": "empty", "empty": "empty"})


def load_normalized_styles(manifest: Path) -> dict[str, Style]:
    """Load one ``labels_normalized.json`` into one or more styles.

    Two schemas are auto-detected from the first record:

    * **board-level** (``corners_a8_h8_h1_a1`` + ``placement``) -> a single
      :class:`Style` in ``board`` mode; images relative to the manifest dir.
    * **crop-level** (``piece`` + ``chess_set``; already-cut square images) -> one
      :class:`Style` per ``chess_set`` in ``crop`` mode. Empty-square crops (no
      ``chess_set``) form a shared pool copied into every style, since empties are
      style-agnostic (DESIGN §4.6).
    """
    manifest = Path(manifest)
    base = manifest.parent
    with open(manifest) as f:
        records = json.load(f)
    if not records:
        return {}
    if "corners_a8_h8_h1_a1" in records[0]:
        return {base.name: _load_board_manifest(records, base)}
    if "piece" in records[0]:
        return _load_crop_manifest(records, base)
    raise ValueError(f"unrecognised manifest schema in {manifest}: keys {list(records[0])}")


def _load_board_manifest(records: list[dict], base: Path) -> Style:
    style_name = base.name
    samples: list[StyleSample] = []
    for rec in records:
        corners = np.asarray(rec["corners_a8_h8_h1_a1"], dtype=np.float32).reshape(4, 2)
        samples.append(
            StyleSample(
                image_path=(base / rec["image"]).resolve(),
                placement=rec["placement"],
                corners=corners,
                group=str(rec.get("group", style_name)),
            )
        )
    kind = str(records[0].get("kind", "unknown"))
    return Style(name=style_name, kind=kind, _samples=samples, mode=BOARD)


def _load_crop_manifest(records: list[dict], base: Path) -> dict[str, Style]:
    prefix = base.name
    empties = [
        CropSample((base / r["image"]).resolve(), "empty", group=f"{prefix}/empty")
        for r in records
        if _PIECE_TO_CLASS.get(r.get("piece")) == "empty"
    ]
    by_set: dict[str, list[CropSample]] = defaultdict(list)
    for r in records:
        label = _PIECE_TO_CLASS.get(r.get("piece"))
        if label is None:
            raise ValueError(f"unknown piece code {r.get('piece')!r} in {prefix}")
        if label == "empty":
            continue
        setname = str(r.get("chess_set") or "default")
        by_set[setname].append(
            CropSample((base / r["image"]).resolve(), label, group=f"{prefix}/{setname}")
        )
    styles: dict[str, Style] = {}
    for setname, samples in sorted(by_set.items()):
        styles[f"{prefix}_{setname}"] = Style(
            name=f"{prefix}_{setname}", kind="real", _samples=samples + list(empties), mode=CROP
        )
    return styles


# --------------------------------------------------------------------------- #
# ChessReD adapter (real dataset, its own annotations.json schema).
# --------------------------------------------------------------------------- #
CHESSRED = EXTERNAL / "chessred"

# ChessReD category id -> our class symbol (white P..K 0-5, black p..k 6-11).
_CHESSRED_TO_CLASS = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
}
# Board-anchored corner keys in our semantic a8,h8,h1,a1 order (validated in
# tools/build_exemplar_bank_chessred.py: 98.9% of pieces project into their square).
_CHESSRED_CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


def chessred_style(
    root: Path = CHESSRED,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    max_boards: int | None = None,
) -> Style:
    """Adapt the corner-annotated ChessReD2K subset to a single :class:`Style`.

    ChessReD2K is one physical set filmed across 20 games (one fixed camera each),
    so it is a single *style* with 20 viewpoint ``group``s. ``max_boards`` caps the
    board count (evenly across games) for quick smoke runs.
    """
    with open(root / "annotations.json") as f:
        ann = json.load(f)
    images = {im["id"]: im for im in ann["images"]}
    corners = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
    pieces: dict[int, list[dict]] = defaultdict(list)
    for p in ann["annotations"]["pieces"]:
        pieces[p["image_id"]].append(p)
    split_ids = ann["splits"]["chessred2k"]

    ids: list[int] = []
    for s in splits:
        ids.extend(split_ids[s]["image_ids"])
    ids = sorted(set(ids) & set(corners))  # only boards with corner labels

    if max_boards is not None and max_boards < len(ids):
        by_game: dict[int, list[int]] = defaultdict(list)
        for iid in ids:
            by_game[images[iid]["game_id"]].append(iid)
        per_game = max(1, max_boards // max(1, len(by_game)))
        capped: list[int] = []
        for game, gids in sorted(by_game.items()):
            gids = sorted(gids, key=lambda i: images[i].get("move_id", i))
            idx = np.linspace(0, len(gids) - 1, min(per_game, len(gids)))
            capped.extend(gids[int(round(k))] for k in idx)
        ids = sorted(set(capped))[:max_boards]

    samples: list[StyleSample] = []
    for iid in ids:
        grid = [["empty"] * GRID for _ in range(GRID)]
        for p in pieces[iid]:
            pos = p["chessboard_position"]
            fc = ord(pos[0]) - ord("a")
            r = GRID - int(pos[1])
            grid[r][fc] = _CHESSRED_TO_CLASS[p["category_id"]]
        c = corners[iid]
        quad = np.array([c[k] for k in _CHESSRED_CORNER_KEYS], dtype=np.float32)
        samples.append(
            StyleSample(
                image_path=(root / images[iid]["path"]).resolve(),
                placement=encode_placement(grid),
                corners=quad,
                group=f"game{images[iid]['game_id']:03d}",
            )
        )
    return Style(name="chessred", kind="real", _samples=samples)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_styles(external: Path = EXTERNAL) -> dict[str, Style]:
    """Return every available style keyed by name (lazy sources loaded on demand).

    Finds ChessReD (if ``annotations.json`` present) plus every
    ``labels_normalized.json`` under ``data/external``. A style's name is its
    manifest directory; collisions get a ``_2`` suffix.
    """
    styles: dict[str, Style] = {}
    if (external / "chessred" / "annotations.json").exists():
        styles["chessred"] = chessred_style()
    for manifest in sorted(external.glob(f"**/{NORMALIZED_MANIFEST}")):
        for name, style in load_normalized_styles(manifest).items():
            uniq = name
            i = 2
            while uniq in styles:
                uniq = f"{name}_{i}"
                i += 1
            style.name = uniq
            styles[uniq] = style
    return styles


if __name__ == "__main__":
    found = discover_styles()
    if not found:
        print("no styles found under", EXTERNAL)
    for name, st in found.items():
        print(f"{name:24s} kind={st.kind:9s} boards={len(st):5d} groups={st.n_groups}")
