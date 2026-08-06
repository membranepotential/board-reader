"""Build the shipped DINOv2 exemplar bank from the labeled eval photos.

Embeds every labeled photo's 64 squares (plus photometric variants) with the
DINOv2 backbone and writes the kNN exemplar bank the gateway loads via
``BR_CENTROIDS=models/exemplars_dinov2.npz``. Re-run after adding photos to
``data/eval/ground_truth.json``.

Box geometry (headroom 0.15, side_pad 0.15) is the LOO-validated optimum on the
eval set (294/320 vs 261 for class centroids at headroom 0.3).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from board_reader_core.protocol import CLASSES, parse_placement
from board_reader_inference.centroid import photometric_variants
from board_reader_inference.embed import DinoV2Embedder, l2_normalize
from board_reader_inference.squares import crop_squares_photo

ROOT = Path(__file__).resolve().parent.parent
HEADROOM, SIDE_PAD, TEMPERATURE = 0.15, 0.15, 0.1
OUT = ROOT / "models/exemplars_dinov2.npz"


def main() -> None:
    gt = json.load(open(ROOT / "data/eval/ground_truth.json"))
    emb = DinoV2Embedder(ROOT / "models/dinov2_small.onnx")
    crops: list[np.ndarray] = []
    labels: list[str] = []
    for key, entry in gt.items():
        img = cv2.imread(str(Path(entry["source"]).expanduser()))
        if img is None:
            raise FileNotFoundError(entry["source"])
        corners = np.array(entry["corners_a8_h8_h1_a1"], dtype=np.float32)
        sq = crop_squares_photo(img, corners, headroom=HEADROOM, side_pad=SIDE_PAD)
        lab = [c for row in parse_placement(entry["placement"]) for c in row]
        for crop, l in zip(sq, lab):
            variants = photometric_variants(crop)
            crops.extend(variants)
            labels.extend([l] * len(variants))
        print(f"{key}: {len(sq)} squares -> {len(crops)} exemplars so far")
    exemplars = l2_normalize(emb.embed(crops), axis=1)
    np.savez(
        OUT,
        kind=np.array("exemplar"),
        exemplars=exemplars.astype(np.float32),
        labels=np.array(labels),
        classes=np.array(CLASSES),
        temperature=np.float64(TEMPERATURE),
        backbone=np.array("dinov2"),
        crop_mode=np.array("photo"),
        box_headroom=np.float64(HEADROOM),
        box_side_pad=np.float64(SIDE_PAD),
    )
    print(f"wrote {OUT}: {exemplars.shape[0]} exemplars, dim {exemplars.shape[1]}")


if __name__ == "__main__":
    main()
