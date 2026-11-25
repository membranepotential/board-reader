"""Board outlines for the recorded takes, from the classical detector
(``tools/video/boarddetect.py``) and a temporal median.

    CUDA_VISIBLE_DEVICES= uv run python tools/probes/autocrop_label.py TAKE [TAKE ...]

The gate takes' ``frames.ndjson`` corners are the 2026-08 edge's per-frame
estimates and sit up to ~5 % of the board width off (checked by eye: the
overlay misses whole board lines on 172736 and 181802). The live 2026-11-24/25
takes have no corners at all. So every 5th frame of a take goes through the
detector, and a frame's outline is the corner-wise median of the detections
within +-20 rows that agree with each other (a camera bump starts a new median
within 20 rows). A frame with fewer than 3 agreeing detections gets none.

Writes SCRATCH/quads/<take>.json: {row: [[x, y] x 4]} in TL, TR, BR, BL order
(geometric: the detector cannot know which corner is a8).
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video"))
from autocrop_lib import ROOT, SCRATCH
from boarddetect import detect_board_quad

STEP = 5
WINDOW = 20
AGREE = 0.02  # of the image width, per corner


def detect_row(args: tuple[str, int, str]) -> tuple[int, list | None]:
    take, row, jpeg = args
    cv2.setNumThreads(1)
    q = detect_board_quad(cv2.imread(str(ROOT / "recordings" / take / jpeg)))
    return row, None if q is None else q.tolist()


def label(take: str, pool: Pool) -> None:
    out = SCRATCH / "quads" / f"{take}.json"
    if out.exists():
        return
    lines = [
        json.loads(s)
        for s in (ROOT / "recordings" / take / "frames.ndjson").read_text().splitlines()
        if s.strip()
    ]
    jobs = [(take, r, ln["jpeg"]) for r, ln in enumerate(lines) if r % STEP == 0]
    found = {r: np.array(q, np.float32) for r, q in pool.map(detect_row, jobs) if q}
    width = cv2.imread(str(ROOT / "recordings" / take / lines[0]["jpeg"])).shape[1]
    quads = {}
    for row in range(len(lines)):
        near = [q for r, q in found.items() if abs(r - row) <= WINDOW]
        if len(near) < 3:
            continue
        med = np.median(near, axis=0)
        agree = [q for q in near if np.abs(q - med).max() < AGREE * width]
        if len(agree) >= 3:
            quads[row] = np.median(agree, axis=0).round(1).tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(quads))
    print(
        f"{take}: {len(found)}/{len(jobs)} detected, {len(quads)}/{len(lines)} rows labelled"
    )


if __name__ == "__main__":
    with Pool(10) as pool:
        for take in sys.argv[1:]:
            label(take, pool)
