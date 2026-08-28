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
from typing import TYPE_CHECKING

import cv2
import numpy as np

# Cap OpenCV's internal thread pool for quiet, low-parallelism caching runs
# (BR_CV_THREADS=2 nice -n 19 keeps the laptop fans down). Unset -> OpenCV default.
if os.environ.get("BR_CV_THREADS"):
    cv2.setNumThreads(int(os.environ["BR_CV_THREADS"]))

import _bootstrap
from styles import BOARD, Style, chessred_style, discover_styles

if TYPE_CHECKING:
    from config import Config

from board_reader_core.protocol import CLASS_INDEX, CLASSES, parse_placement

from board_reader_inference.squares import (
    BboxGeometry,
    CropGeometry,
    OcclusionFill,
    PoseGeometry,
    crop_squares_photo,
    geometry_from_dict,
    geometry_tag,
)

CACHE_ROOT = _bootstrap.ROOT / "data" / "external" / "_cropcache"

# The cache's legacy tight-crop default — NOT the deployed geometry. The gateway
# and models/*.meta.json cut at BboxGeometry(headroom=1.6, side_pad=0.15)
# (squares.DEFAULT_GEOMETRY); headroom 0.15 decapitates every tall piece. Runs
# that must match deployment pass crop_geometry explicitly (run_e/run_f do).
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
    # Occlusion fill applied to BOARD-mode crops at build time (E7), or None for
    # the pre-E7 unmasked caches. Missing in old meta.json -> None (default),
    # which is the correct read: those caches predate masking entirely.
    mask: str | None = None
    # Style mode at build time ("board" | "crop"). Lets a machine that holds
    # only the cache (a cloud pod — no raw datasets) resolve styles from it.
    # Metas older than this field default to "board", which is right for every
    # cache we ship to such machines; the one crop-mode cache (zenodo) lives
    # only where the raw registry also lives and resolves from there.
    mode: str = BOARD

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


def _board_crops(
    sample, size: int, geometry: CropGeometry, mask: OcclusionFill | None = None
):
    """(crop, label_idx) pairs for one board-mode sample.

    ``mask`` mirrors the deployed occlusion-aware masking exactly (it IS
    ``crop_squares_photo``'s own ``mask`` param — same closer-square selection,
    same own-prism exclusion, same fill) so a cache built with ``mask="gray"``
    matches the ``BR_CROP_MASK=gray`` serving domain pixel-for-pixel.
    """
    img = cv2.imread(str(sample.image_path))
    if img is None:
        return None
    crops = crop_squares_photo(img, sample.corners, geometry=geometry, size=size, mask=mask)
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
    mask: OcclusionFill | None = None,
) -> CacheMeta:
    """Crop every board of ``style`` and write shards + ``meta.json``.

    Idempotent: returns the existing meta if a complete cache is already present
    (unless ``force``).

    ``mask``: occlusion-aware fill (E6/E7), applied only to BOARD-mode styles —
    they carry per-sample ``corners``, which is what the occlusion geometry needs
    to find closer neighbor prisms. CROP-mode styles (already-cut single-square
    images, no corners) are built unmasked regardless of ``mask``; the caller is
    warned once so this never silently no-ops.
    """
    out_dir = out_root / style.name
    meta_path = out_dir / "meta.json"
    n_samples = len(style)
    effective_mask = mask if style.mode == BOARD else None
    if mask is not None and style.mode != BOARD:
        print(f"  {style.name}: crop-mode style (no board corners) — building unmasked despite mask={mask!r}")
    if meta_path.exists() and not force:
        with open(meta_path) as f:
            existing = CacheMeta(**json.load(f))
        # Reuse only if the source style is unchanged; a differing sample count
        # (e.g. a capped smoke cache vs a full run, or new images added) is stale.
        if existing.n_samples == n_samples and existing.size == size and existing.mask == effective_mask:
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
            _board_crops(s, size, geometry, effective_mask)
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
        mask=effective_mask,
        mode=style.mode,
    )
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)
    return meta


def geo_cache_root(cfg):  # cfg: config.Config — annotated loosely to keep this
    # module torch-free AND config-import-free (config imports nothing heavy, but
    # the dependency arrow cache<-config stays one-way).
    """Geometry-keyed cache dir so different geometries don't collide/reuse.
    The original ``BboxGeometry(0.15, 0.15)`` maps to the default root (run (a)'s
    cache), which predates the tagging scheme. A ``mask`` (E7) is folded into the
    tag too — a masked cache must never collide with (or be silently reused as)
    an unmasked one, since the pixels differ."""
    geometry = cfg.geometry
    is_default = geometry == BboxGeometry(headroom=0.15, side_pad=0.15)
    if is_default and cfg.mask is None:
        return CACHE_ROOT
    tag = "hr0.15_sp0.15" if is_default else geometry_tag(geometry)
    if cfg.mask is not None:
        tag = f"{tag}_mask{cfg.mask}"
    return CACHE_ROOT / tag

# --- style resolution against caches (torch-free: also used by tests/pods) ---

def cached_style_modes(cfg: Config) -> dict[str, str]:
    """Styles resolvable from a complete crop cache alone, name -> mode.

    A cache with ``meta.json`` is self-contained for training (``load_styles``
    reads only shards), so on a machine without the raw datasets — a cloud pod
    holds nothing but staged caches — the cache itself is the style registry.
    """
    root = geo_cache_root(cfg)
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for d in sorted(root.iterdir()):
        if (d / "meta.json").exists():
            out[d.name] = load_meta(d.name, root).mode
    return out


def resolve_styles(cfg: Config) -> list[str]:
    available = discover_styles()
    if cfg.require_mode is not None:
        available = {n: s for n, s in available.items() if s.mode == cfg.require_mode}
    # The raw registry wins; complete caches fill the gaps. Note the two orders
    # differ (registry: manifest-path order; cache: sorted) — a run that must be
    # order-reproducible across machines should list its styles explicitly.
    cached = {
        n: m
        for n, m in cached_style_modes(cfg).items()
        if n not in available and cfg.require_mode in (None, m)
    }
    if cfg.styles == ["all"]:
        return list(available) + list(cached)
    known = set(available) | set(cached)
    missing = [s for s in cfg.styles if s not in known]
    if missing:
        raise SystemExit(f"styles not found: {missing}; available: {sorted(known)}")
    return list(cfg.styles)



def ensure_caches(names: list[str], cfg: Config) -> None:
    root = geo_cache_root(cfg)
    # A style whose cache is already complete never needs its raw data — that is
    # what lets a pod train from staged caches alone. (build_cache would no-op on
    # the existing meta anyway, but only after a registry lookup that dies when
    # the raw datasets are absent.)
    todo = [n for n in names if not (root / n / "meta.json").exists()]
    if not todo:
        return
    available = discover_styles()
    for name in todo:
        style = available[name]
        if name == "chessred" and cfg.max_boards is not None:
            style = chessred_style(max_boards=cfg.max_boards)
        build_cache(style, out_root=root, size=cfg.size, geometry=cfg.geometry, mask=cfg.mask)


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
    ap.add_argument(
        "--mask", choices=["gray", "blur"], default=None,
        help="occlusion-aware fill (E6/E7), BOARD-mode styles only — matches "
             "BR_CROP_MASK at inference. A masked cache must live under its own "
             "--out-root (e.g. <geometry-tag>_mask<fill>); it never collides with "
             "an unmasked one.",
    )
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
        print(f"caching {name} ({len(style)} boards) {geometry} mask={args.mask} -> {args.out_root}")
        meta = build_cache(style, out_root=args.out_root, shard_size=args.shard_size,
                           geometry=geometry, force=args.force, mask=args.mask)
        nz = {c: n for c, n in meta.class_counts.items() if n}
        print(f"  -> {meta.n_crops} crops in {len(meta.shards)} shards; classes: {nz}")


if __name__ == "__main__":
    main()
