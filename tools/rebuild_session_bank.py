"""Rebuild the session exemplar bank from retained calibrate uploads.

Recovery + hygiene tool (2026-07-18 field incident: an orientation-blind
/calibrate pushed color-swapped exemplars into the persisted bank). Rebuilds
``models/exemplars_session.npz`` from scratch:

  base bank (models/exemplars_dinov2.npz)
    + every ``data/uploads/*-calibrate.jpg``, with its label orientation
      VERIFIED against the trained model's own reading of that photo (same
      agreement rule the server's /calibrate guard now applies) — so a
      calibrate taken with mislabeled corners contributes correctly-labeled
      exemplars instead of poison.

The old session bank is moved aside (``.replaced-<ts>`` suffix), never deleted.

Usage:  CUDA_VISIBLE_DEVICES= uv run python tools/rebuild_session_bank.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from board_reader_core.protocol import CLASSES, encode_placement, parse_placement
from board_reader_core.wire import Frame
from board_reader_inference.centroid import build_exemplars, load_classifier
from board_reader_inference.embed import make_embedder
from board_reader_inference.squares import crop_squares_photo
from board_reader_inference.trained import TrainedEstimator

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models/exemplars_dinov2.npz"
OUT = ROOT / "models/exemplars_session.npz"
MODEL = ROOT / "models/piece_classifier_multistyle_v1.onnx"
UPLOADS = ROOT / "data/uploads"


def main() -> None:
    classifier = load_classifier(BASE)
    print(f"base bank: {BASE.name} {classifier.exemplars.shape} backbone={classifier.backbone}")
    embedder = make_embedder(classifier.backbone)
    trained = TrainedEstimator(MODEL)

    for meta_path in sorted(UPLOADS.glob("*-calibrate.json")):
        meta = json.loads(meta_path.read_text())
        img = cv2.imread(str(meta_path.with_suffix(".jpg")))
        if img is None or "corners" not in meta:
            print(f"  skip {meta_path.name}: unreadable")
            continue
        corners = np.array(
            [float(v) for v in meta["corners"].split(",")], np.float32
        ).reshape(4, 2)
        placement = meta.get("placement")
        if not placement:
            print(f"  skip {meta_path.name}: no placement")
            continue
        probe = trained.estimate_photo(
            Frame(board_id="rebuild", seq=0, capture_ts=time.time(), image=img), corners
        )
        pred = np.array(
            [CLASSES[i] for i in np.asarray(probe.square_dist).reshape(64, len(CLASSES)).argmax(-1)]
        )
        grid = np.array(parse_placement(placement), dtype="<U5")
        agrees = [int((np.rot90(grid, k).reshape(-1) == pred).sum()) for k in range(4)]
        best = int(np.argmax(agrees))
        claimed = np.rot90(grid, best).reshape(-1)
        pieces = claimed != "empty"
        piece_agree = int((claimed[pieces] == pred[pieces]).sum())
        if piece_agree < 0.6 * int(pieces.sum()):
            # Same floor as the server's CALIBRATE_MIN_PIECE_AGREE: the photo
            # does not show the claimed position (e.g. a mid-game board
            # calibrated as "start") — poison, not data.
            print(
                f"  REJECT {meta_path.name}: piece agreement {piece_agree}/{int(pieces.sum())}"
            )
            continue
        rotation = best if best != 0 and agrees[best] >= agrees[0] + 8 else 0
        if rotation:
            placement = encode_placement(np.rot90(grid, rotation).tolist())
        crops = crop_squares_photo(
            img,
            corners,
            headroom=classifier.box_headroom,
            side_pad=classifier.box_side_pad,
            size=224,
        )
        new = build_exemplars(
            embedder,
            crops,
            placement,
            backbone=classifier.backbone,
            box_headroom=classifier.box_headroom,
            box_side_pad=classifier.box_side_pad,
        )
        classifier = classifier.extended(new.exemplars, new.labels)
        note = f"ROTATED {rotation * 90}°" if rotation else "as-given"
        print(
            f"  + {meta_path.name}: agree={agrees} -> {note}, bank now {classifier.exemplars.shape[0]}"
        )

    if OUT.exists():
        backup = OUT.with_name(f"{OUT.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}")
        OUT.rename(backup)
        print(f"old bank moved to {backup.name}")
    classifier.save(OUT)
    print(f"saved {OUT} {classifier.exemplars.shape}")


if __name__ == "__main__":
    main()
