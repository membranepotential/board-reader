"""Final held-out evaluation on the USER's own set (never trained/validated on).

Two eval sets, both cropped with the deployed photo geometry:

* ``user_photos`` — ``data/eval/ground_truth.json`` (5 labelled photos; originals
  referenced by each record's ``source``),
* ``uploads_calibrate`` — the dim-light start-position shots in ``data/uploads``
  (``kind=calibrate``, ``placement=start``, corners in the sidecar json).

Reports per-square accuracy, the empty->piece / piece->empty / piece->piece splits
(see :mod:`metrics`), and whole-board-perfect count, per set and overall. Evaluates
a torch checkpoint or, with ``--onnx``, the exported model through onnxruntime.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import _bootstrap
import cv2
import numpy as np
import torch
from augment import IMAGENET_MEAN, IMAGENET_STD, eval_transform, to_rgb_tensor
from config import Config
from metrics import SquareStats

from board_reader_core.protocol import CLASS_INDEX, START_PLACEMENT, parse_placement
from models import build_model

from board_reader_inference.squares import (
    BboxGeometry,
    CropGeometry,
    OcclusionFill,
    crop_squares_photo,
    geometry_from_dict,
)

ROOT = _bootstrap.ROOT
HEADROOM, SIDE_PAD, SIZE = 0.15, 0.15, 224
DEFAULT_EVAL_GEOMETRY: CropGeometry = BboxGeometry(HEADROOM, SIDE_PAD)


@dataclass
class EvalBoard:
    name: str
    image_path: Path
    corners: np.ndarray  # (4,2)
    labels: np.ndarray  # (64,) int class indices, crop order


def _labels_from_placement(placement: str) -> np.ndarray:
    grid = parse_placement(placement)
    return np.array([CLASS_INDEX[c] for row in grid for c in row], dtype=np.int64)


def load_user_photos(gt_path: Path = ROOT / "data/eval/ground_truth.json") -> list[EvalBoard]:
    boards: list[EvalBoard] = []
    if not gt_path.exists():
        return boards
    with open(gt_path) as f:
        gt = json.load(f)
    for name, rec in gt.items():
        src = Path(rec["source"]).expanduser()
        if not src.is_absolute():  # repo-relative sources (e.g. data/uploads/...) resolve against ROOT
            src = ROOT / src
        if not src.exists():  # fall back to a rectified sibling if the original moved
            alt = gt_path.parent / f"{name}_rect.png"
            if not alt.exists():
                print(f"  WARN {name}: source {rec['source']} missing, skipping")
                continue
            src = alt
        corners = np.asarray(rec["corners_a8_h8_h1_a1"], dtype=np.float32).reshape(4, 2)
        boards.append(EvalBoard(name, src, corners, _labels_from_placement(rec["placement"])))
    return boards


def load_style(name: str) -> list[EvalBoard]:
    """A registered style's boards as an eval set (the unseen-style validation).

    ``chesscog_transfer`` is the one that matters: 20 real photos of a physical
    set that appears in no training run, so it is the honest real-photo
    generalisation number. Keep it out of ``styles``, score it after every run.
    """
    from styles import discover_styles

    style = discover_styles().get(name)
    if style is None:
        print(f"  WARN style {name!r} not found")
        return []
    return [
        EvalBoard(
            f"{name}/{s.image_path.stem}",
            s.image_path,
            s.corners,
            _labels_from_placement(s.placement),
        )
        for s in style.samples()
    ]


def load_uploads_calibrate(uploads: Path = ROOT / "data/uploads") -> list[EvalBoard]:
    boards: list[EvalBoard] = []
    for meta in sorted(uploads.glob("*-calibrate.json")):
        with open(meta) as f:
            d = json.load(f)
        img = meta.with_suffix(".jpg")
        if not img.exists() or "corners" not in d:
            continue
        nums = [float(v) for v in str(d["corners"]).split(",")]
        if len(nums) != 8:
            continue
        corners = np.asarray(nums, dtype=np.float32).reshape(4, 2)
        placement = d.get("placement", START_PLACEMENT).split()[0]
        boards.append(EvalBoard(meta.stem, img, corners, _labels_from_placement(placement)))
    return boards


class TorchPredictor:
    def __init__(self, ckpt_path: Path, device: torch.device | None = None, *, grayscale: bool = False) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = Config(**ckpt["config"]) if isinstance(ckpt.get("config"), dict) else Config()
        self.model = build_model(
            cfg.model, pretrained=False, unfreeze_blocks=cfg.unfreeze_blocks,
            freeze_backbone=cfg.freeze_backbone,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.tf = eval_transform(SIZE, grayscale=grayscale)

    @torch.no_grad()
    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        batch = torch.stack([self.tf(to_rgb_tensor(c)) for c in crops]).to(self.device)
        return self.model(batch).argmax(1).cpu().numpy()


class OnnxPredictor:
    def __init__(self, onnx_path: Path, *, grayscale: bool = False) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.grayscale = grayscale
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.resize(crop[:, :, ::-1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            rgb = np.stack([g, g, g], axis=-1)
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        return (x - self.mean) / self.std

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([self._prep(c) for c in crops]).astype(np.float32)
        logits = self.sess.run(None, {self.input: batch})[0]
        return logits.argmax(1)


def _make_predictor(args, *, grayscale: bool):
    return (
        OnnxPredictor(args.onnx, grayscale=grayscale)
        if args.onnx
        else TorchPredictor(args.ckpt, grayscale=grayscale)
    )


def eval_boards(
    boards: list[EvalBoard],
    predictor,
    *,
    geometry: CropGeometry,
    mask: OcclusionFill | None = None,
    per_board: bool = False,
) -> tuple[SquareStats, int]:
    stats = SquareStats()
    perfect = 0
    for b in boards:
        img = cv2.imread(str(b.image_path))
        if img is None:
            print(f"  WARN unreadable {b.image_path}")
            continue
        crops = crop_squares_photo(img, b.corners, geometry=geometry, size=SIZE, mask=mask)
        pred = predictor.predict(crops)
        stats.update(b.labels, pred)
        n_correct = int((pred == b.labels).sum())
        if n_correct == 64:
            perfect += 1
        if per_board:
            print(f"    {b.name:22s} {n_correct}/64")
    return stats, perfect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, help="torch checkpoint (best.pt)")
    ap.add_argument("--onnx", type=Path, help="exported ONNX model (evaluate through onnxruntime)")
    ap.add_argument("--headroom", type=float, default=None, help="override bbox crop geometry")
    ap.add_argument("--side-pad", type=float, default=None, help="override bbox crop geometry")
    ap.add_argument(
        "--style", action="append", default=["chesscog_transfer"],
        help="also score this registered style as an unseen-style set "
             "(default: chesscog_transfer; repeatable)",
    )
    ap.add_argument(
        "--mask", choices=["gray", "blur"], default=None,
        help="occlusion-aware crop masking (E7), matching BR_CROP_MASK at serve "
             "time. Applied only to sets that carry real board corners "
             "(user_photos, uploads_calibrate, and any --style that is a "
             "BOARD-mode style, e.g. chesscog_transfer); a CROP-mode style has no "
             "corners to derive occlusion from and is always scored unmasked, "
             "noted per-set below.",
    )
    ap.add_argument("--no-grayscale", action="store_true", help="skip the shape-bias diagnostic")
    args = ap.parse_args()
    if not args.ckpt and not args.onnx:
        ap.error("give --ckpt or --onnx")

    # Geometry travels with the model, never with the caller: a torch checkpoint
    # carries its training Config, an ONNX its `.meta.json` sidecar. Guessing it
    # from flags is how a tall-crop model got served on short crops once.
    geometry = DEFAULT_EVAL_GEOMETRY
    if args.ckpt:
        import torch
        c = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("config", {})
        geometry = (Config(**c) if isinstance(c, dict) else Config()).geometry
    elif args.onnx:
        sidecar = Path(str(args.onnx) + ".meta.json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            geometry = geometry_from_dict(meta.get("crop_geometry", meta))
        else:
            print(f"  WARN no sidecar at {sidecar}; assuming {geometry}")
    if args.headroom is not None or args.side_pad is not None:
        base = geometry if isinstance(geometry, BboxGeometry) else DEFAULT_EVAL_GEOMETRY
        geometry = BboxGeometry(
            headroom=args.headroom if args.headroom is not None else base.headroom,
            side_pad=args.side_pad if args.side_pad is not None else base.side_pad,
        )
    print(f"eval geometry: {geometry}")

    # (boards, mask) per set. user_photos/uploads_calibrate always carry real
    # corners (ground_truth.json / upload sidecars), so a requested mask always
    # applies. A --style set only gets one if it is BOARD-mode (has corners);
    # CROP-mode styles (pre-cut single squares, e.g. the Zenodo auxiliary sets)
    # have nothing to derive occlusion from and stay unmasked — noted below so
    # it reads as a documented caveat, not a silent mismatch.
    from styles import BOARD
    from styles import discover_styles as _discover_styles

    available_styles = _discover_styles()
    sets: dict[str, tuple[list[EvalBoard], OcclusionFill | None]] = {
        "user_photos": (load_user_photos(), args.mask),
        "uploads_calibrate": (load_uploads_calibrate(), args.mask),
    }
    for style_name in dict.fromkeys(args.style):
        boards = load_style(style_name)
        style = available_styles.get(style_name)
        set_mask = args.mask
        if args.mask is not None and (style is None or style.mode != BOARD):
            print(f"  NOTE: {style_name} is crop-level (no board corners) — "
                  f"scoring unmasked despite --mask {args.mask}")
            set_mask = None
        sets[style_name] = (boards, set_mask)

    def run(grayscale: bool) -> SquareStats:
        predictor = _make_predictor(args, grayscale=grayscale)
        tag = "GRAYSCALE (shape-bias diagnostic)" if grayscale else "COLOUR"
        print(f"\n=== {tag} ===")
        overall = SquareStats()
        total_perfect = total_boards = 0
        for name, (boards, set_mask) in sets.items():
            if not boards:
                print(f"{name}: (no boards)")
                continue
            stats, perfect = eval_boards(boards, predictor, geometry=geometry,
                                         mask=set_mask, per_board=not grayscale)
            for f in overall.__dataclass_fields__:
                setattr(overall, f, getattr(overall, f) + getattr(stats, f))
            total_perfect += perfect
            total_boards += len(boards)
            print(f"{name}: {len(boards)} boards, perfect {perfect}/{len(boards)}")
            print(f"  {stats.summary()}")
        print(f"OVERALL: perfect {total_perfect}/{total_boards} | total {overall.correct}/{overall.total}")
        print(f"  {overall.summary()}")
        return overall

    colour = run(grayscale=False)
    if not args.no_grayscale:
        gray = run(grayscale=True)
        drop = colour.accuracy - gray.accuracy
        print(f"\nshape-bias proxy: colour {colour.accuracy:.3f} -> grayscale {gray.accuracy:.3f} "
              f"(drop {drop:+.3f}; smaller = more shape-biased)")


if __name__ == "__main__":
    main()
