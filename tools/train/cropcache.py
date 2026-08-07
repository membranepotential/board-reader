"""Crop-cache builder: a style's boards -> (crop, label) samples on disk.

Each board is cut into 64 square crops with the **exact deployed geometry** —
:func:`board_reader_inference.squares.crop_squares_photo` with the shipped
``headroom=0.15, side_pad=0.15, size=224`` — so a model trained here sees pixels
identical to what the server feeds at inference time. Crops are JPEG-recompressed
into pickle-free shards (a concatenated byte blob + offsets) to keep a full style
in the low-GB range instead of tens of GB of raw uint8.

Layout::

    data/external/_cropcache/<style>/
        meta.json          # geometry, class balance, group map, shard index
        shard_0000.npz     # blob:uint8, offsets:int64[N+1], labels:int8[N], group_ids:int32[N]
        ...

``labels`` index into :data:`board_reader_core.protocol.CLASSES` (0 = empty).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Cap OpenCV's internal thread pool for quiet, low-parallelism caching runs
# (BR_CV_THREADS=2 nice -n 19 keeps the laptop fans down). Unset -> OpenCV default.
if os.environ.get("BR_CV_THREADS"):
    cv2.setNumThreads(int(os.environ["BR_CV_THREADS"]))

import _bootstrap
from styles import BOARD, Style, discover_styles

from board_reader_core.protocol import CLASS_INDEX, CLASSES, parse_placement

from board_reader_inference.squares import (
    BboxGeometry,
    CropGeometry,
    PoseGeometry,
    crop_squares_photo,
    geometry_from_dict,
)

CACHE_ROOT = _bootstrap.ROOT / "data" / "external" / "_cropcache"

# Deployed photo-crop geometry (must match the gateway + build_exemplar_bank*).
HEADROOM = 0.15
SIDE_PAD = 0.15
DEFAULT_CACHE_GEOMETRY: CropGeometry = BboxGeometry(HEADROOM, SIDE_PAD)
SIZE = 224
JPEG_QUALITY = 95
DEFAULT_SHARD = 4096  # crops per shard (~64 boards)


@dataclass
class CacheMeta:
    style: str
    kind: str
    n_boards: int
    n_crops: int
    size: int
    headroom: float
    side_pad: float
    shards: list[str]
    group_names: list[str]
    class_counts: dict[str, int]
    n_samples: int = 0  # source style sample count at build time (staleness check)
    # Full geometry record. `headroom`/`side_pad` above stay for the caches built
    # before pose geometry existed; for those this is None and they are the truth.
    crop_geometry: dict | None = None

    @property
    def geometry(self) -> CropGeometry:
        if self.crop_geometry is not None:
            return geometry_from_dict(self.crop_geometry)
        return BboxGeometry(headroom=self.headroom, side_pad=self.side_pad)

    def to_dict(self) -> dict:
        return self.__dict__


def _flatten_labels(placement: str) -> list[int]:
    grid = parse_placement(placement)
    return [CLASS_INDEX[cell] for row in grid for cell in row]


def _pad_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox a crop to square (edge-replicate) and resize — matches how
    crop_squares_photo squares its boxes, so crop-level and board-level caches
    are geometrically consistent enough to share a model input."""
    h, w = img.shape[:2]
    if h != w:
        d = abs(h - w)
        if h > w:
            img = cv2.copyMakeBorder(img, 0, 0, d // 2, d - d // 2, cv2.BORDER_REPLICATE)
        else:
            img = cv2.copyMakeBorder(img, d // 2, d - d // 2, 0, 0, cv2.BORDER_REPLICATE)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _board_crops(sample, size: int, geometry: CropGeometry):
    """(crop, label_idx) pairs for one board-mode sample."""
    img = cv2.imread(str(sample.image_path))
    if img is None:
        return None
    crops = crop_squares_photo(img, sample.corners, geometry=geometry, size=size)
    return list(zip(crops, _flatten_labels(sample.placement)))


def _crop_sample(sample, size: int):
    """(crop, label_idx) for one crop-mode sample (already a single square)."""
    img = cv2.imread(str(sample.image_path))
    if img is None:
        return None
    return [(_pad_resize(img, size), CLASS_INDEX[sample.label])]


class _ShardWriter:
    """Accumulates crops and flushes fixed-size pickle-free npz shards."""

    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.shards: list[str] = []
        self._jpegs: list[bytes] = []
        self._labels: list[int] = []
        self._groups: list[int] = []

    def add(self, jpeg: bytes, label: int, group_id: int) -> None:
        self._jpegs.append(jpeg)
        self._labels.append(label)
        self._groups.append(group_id)
        if len(self._jpegs) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._jpegs:
            return
        blob = np.frombuffer(b"".join(self._jpegs), dtype=np.uint8)
        offsets = np.zeros(len(self._jpegs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(j) for j in self._jpegs])
        name = f"shard_{len(self.shards):04d}.npz"
        np.savez(
            self.out_dir / name,
            blob=blob,
            offsets=offsets,
            labels=np.asarray(self._labels, dtype=np.int8),
            group_ids=np.asarray(self._groups, dtype=np.int32),
        )
        self.shards.append(name)
        self._jpegs.clear()
        self._labels.clear()
        self._groups.clear()


def build_cache(
    style: Style,
    *,
    out_root: Path = CACHE_ROOT,
    size: int = SIZE,
    geometry: CropGeometry = DEFAULT_CACHE_GEOMETRY,
    shard_size: int = DEFAULT_SHARD,
    force: bool = False,
) -> CacheMeta:
    """Crop every board of ``style`` and write shards + ``meta.json``.

    Idempotent: returns the existing meta if a complete cache is already present
    (unless ``force``).
    """
    out_dir = out_root / style.name
    meta_path = out_dir / "meta.json"
    n_samples = len(style)
    if meta_path.exists() and not force:
        with open(meta_path) as f:
            existing = CacheMeta(**json.load(f))
        # Reuse only if the source style is unchanged; a differing sample count
        # (e.g. a capped smoke cache vs a full run, or new images added) is stale.
        if existing.n_samples == n_samples and existing.size == size:
            return existing
        print(f"  {style.name}: cache stale (had {existing.n_samples} samples, now {n_samples}); rebuilding")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("shard_*.npz"):
        old.unlink()

    group_ids: dict[str, int] = {}
    writer = _ShardWriter(out_dir, shard_size)
    class_counts: Counter[str] = Counter()
    n_boards = 0
    samples = list(style.samples())
    unit = "boards" if style.mode == BOARD else "crops"
    for n, s in enumerate(samples, 1):
        pairs = (
            _board_crops(s, size, geometry)
            if style.mode == BOARD
            else _crop_sample(s, size)
        )
        if pairs is None:
            print(f"  WARN unreadable image, skipping: {s.image_path}")
            continue
        gid = group_ids.setdefault(s.group, len(group_ids))
        for crop, lab in pairs:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            writer.add(buf.tobytes(), lab, gid)
            class_counts[CLASSES[lab]] += 1
        n_boards += 1
        if n % 50 == 0 or n == len(samples):
            print(f"  {style.name}: {n}/{len(samples)} {unit}, {sum(class_counts.values())} crops", flush=True)
    writer.flush()

    ordered_groups = [g for g, _ in sorted(group_ids.items(), key=lambda kv: kv[1])]
    meta = CacheMeta(
        style=style.name,
        kind=style.kind,
        n_boards=n_boards,
        n_crops=sum(class_counts.values()),
        size=size,
        headroom=getattr(geometry, "headroom", 0.0),
        side_pad=getattr(geometry, "side_pad", 0.0),
        shards=writer.shards,
        group_names=ordered_groups,
        class_counts={c: class_counts.get(c, 0) for c in CLASSES},
        n_samples=len(samples),
        crop_geometry=geometry.to_dict(),
    )
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)
    return meta


def load_meta(style_name: str, out_root: Path = CACHE_ROOT) -> CacheMeta:
    with open(out_root / style_name / "meta.json") as f:
        return CacheMeta(**json.load(f))


def iter_shard(shard_path: Path):
    """Yield ``(jpeg_bytes, label, group_id)`` from one shard."""
    with np.load(shard_path) as z:
        blob = z["blob"].tobytes()
        offsets = z["offsets"]
        labels = z["labels"]
        groups = z["group_ids"]
    for i in range(len(labels)):
        yield blob[offsets[i] : offsets[i + 1]], int(labels[i]), int(groups[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("styles", nargs="*", help="style names (default: all discovered)")
    ap.add_argument("--max-boards", type=int, default=None, help="cap boards (chessred only)")
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD)
    ap.add_argument("--headroom", type=float, default=HEADROOM, help="bbox geometry")
    ap.add_argument("--side-pad", type=float, default=SIDE_PAD, help="bbox geometry")
    ap.add_argument(
        "--pose", nargs=2, type=float, metavar=("PIECE_HEIGHT", "PIECE_RADIUS"),
        default=None,
        help="pose geometry instead of bbox: piece prism size in board units "
             "(e.g. --pose 1.9 0.5). Overrides --headroom/--side-pad.",
    )
    ap.add_argument("--out-root", type=Path, default=CACHE_ROOT, help="cache root (geometry-keyed)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    geometry: CropGeometry = (
        PoseGeometry(piece_height=args.pose[0], piece_radius=args.pose[1])
        if args.pose
        else BboxGeometry(headroom=args.headroom, side_pad=args.side_pad)
    )
    available = discover_styles()
    names = args.styles or list(available)
    for name in names:
        if name not in available:
            print(f"unknown style {name!r}; available: {list(available)}")
            continue
        style = available[name]
        if args.max_boards is not None and name == "chessred":
            from styles import chessred_style

            style = chessred_style(max_boards=args.max_boards)
        print(f"caching {name} ({len(style)} boards) {geometry} -> {args.out_root}")
        meta = build_cache(style, out_root=args.out_root, shard_size=args.shard_size,
                           geometry=geometry, force=args.force)
        nz = {c: n for c, n in meta.class_counts.items() if n}
        print(f"  -> {meta.n_crops} crops in {len(meta.shards)} shards; classes: {nz}")


if __name__ == "__main__":
    main()
