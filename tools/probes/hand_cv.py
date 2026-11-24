"""Plain-CV hand signals for the hand-signal probe: per frame and per square,
computed from the JPEGs alone plus square positions taken from ChessQ's attention.

    uv run python tools/probes/hand_cv.py FEAT_DIR OUT_DIR TAKE [TAKE ...]

FEAT_DIR holds hand_extract.py's <take>.npz (for the attention centroids). Writes
OUT_DIR/<take>.npz with, per row of frames.ndjson:

* background difference against two background models, both the per-pixel median
  of a ring of 15 frames sampled >= 1 s apart:
  - ``ring`` takes every sampled frame (a hand resting > ~7 s becomes background);
  - ``gated`` takes a sampled frame only when its border-touching changed blob
    (below) is under GATE_FRAC of the board, so a resting hand never becomes
    background while ordinary moves (small blobs inside the board) still do
    (reset after 20 s of continuous refusals, so a moved camera is re-learned);
  the colour difference subtracts the median lightness shift over the board
  first, so the phone's auto-exposure does not count as change;
  per square: the fraction of the square's disc whose colour moved by > DIFF_T
  (``sq_ring``, ``sq_gated``); per frame: that fraction over the board hull
  (``fr_ring``, ``fr_gated``) and the largest changed blob that touches the hull's
  border, as a fraction of the hull (``edge_blob_ring``, ``edge_blob_gated``);
* skin: per square / per frame the fraction of skin-coloured pixels (YCrCb box)
  that were not skin-coloured in the gated background (``sq_skin``, ``fr_skin``);
* texture: per square the change in Canny edge density vs the gated background
  (``sq_edge``).

Square discs: centre = decoder layer 2 head 0's attention centroid, median over
the last 9 frames (a hand drags single-frame centroids); radius = 0.45 of the
median distance between neighbouring centres. Frames are processed at 480 px wide.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 480
RING = 15
SAMPLE_S = 1.0
DIFF_T = 30.0  # summed |dL|+|da|+|db| on blurred Lab, 0..255 scale each
GATE_FRAC = 0.03
RESET_S = 20.0
G = 46


def lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(
        np.int16
    )


def skin(image: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycc, (60, 140, 80), (255, 180, 125)) > 0


def edges(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 60, 150) > 0


class Background:
    """Median of a ring of sampled frames; optionally refuses busy frames."""

    def __init__(self, gated: bool) -> None:
        self.gated = gated
        self.ring: deque[np.ndarray] = deque(maxlen=RING)
        self.ring_skin: deque[np.ndarray] = deque(maxlen=RING)
        self.ring_edge: deque[np.ndarray] = deque(maxlen=RING)
        self.last_ts = -1e18
        self.refused_since: float | None = None
        self.model = self.skin = self.edge = None

    def offer(
        self, ts: float, image_lab: np.ndarray, sk: np.ndarray, ed: np.ndarray, busy: float
    ) -> None:
        if ts - self.last_ts < SAMPLE_S:
            return
        if self.gated and self.model is not None and busy >= GATE_FRAC:
            self.refused_since = self.refused_since or ts
            if ts - self.refused_since < RESET_S:
                return
            self.ring.clear(), self.ring_skin.clear(), self.ring_edge.clear()
        self.refused_since = None
        self.last_ts = ts
        self.ring.append(image_lab), self.ring_skin.append(sk), self.ring_edge.append(ed)
        self.model = np.median(np.stack(self.ring), axis=0).astype(np.int16)
        self.skin = np.mean(np.stack(self.ring_skin), axis=0) > 0.5
        self.edge = np.mean(np.stack(self.ring_edge), axis=0)


def centres(feat: np.lib.npyio.NpzFile, scale_x: float, scale_y: float) -> np.ndarray:
    """(n, 64, 2) square centres in the 480-px frame, median over the last 9 frames."""
    cx, cy = feat["cx"][:, 2, 0], feat["cy"][:, 2, 0]
    n = len(cx)
    out = np.zeros((n, 64, 2), np.float32)
    for i in range(n):
        lo = max(0, i - 8)
        out[i, :, 0] = (np.median(cx[lo : i + 1], axis=0) + 0.5) * 14 * scale_x
        out[i, :, 1] = (np.median(cy[lo : i + 1], axis=0) + 0.5) * 14 * scale_y
    return out


def pitch(c: np.ndarray) -> float:
    grid = c.reshape(8, 8, 2)
    d = np.concatenate(
        [
            np.linalg.norm(np.diff(grid, axis=0), axis=-1).ravel(),
            np.linalg.norm(np.diff(grid, axis=1), axis=-1).ravel(),
        ]
    )
    return float(np.median(d))


def run(feat_dir: Path, out_dir: Path, take: str) -> None:
    rec = ROOT / "recordings" / take
    lines = [
        json.loads(s) for s in (rec / "frames.ndjson").read_text().splitlines() if s.strip()
    ]
    feat = np.load(feat_dir / f"{take}.npz")
    first = cv2.imread(str(rec / lines[0]["jpeg"]))
    h = int(first.shape[0] * WIDTH / first.shape[1])
    cen = centres(feat, WIDTH / 644, h / 644)
    n = len(lines)
    keys_sq = ("sq_ring", "sq_gated", "sq_skin", "sq_edge")
    keys_fr = ("fr_ring", "fr_gated", "fr_skin", "edge_blob_ring", "edge_blob_gated")
    res = {k: np.zeros((n, 64), np.float32) for k in keys_sq} | {
        k: np.zeros(n, np.float32) for k in keys_fr
    }
    bgs = {"ring": Background(gated=False), "gated": Background(gated=True)}
    yy, xx = np.mgrid[0:h, 0:WIDTH]
    started = time.monotonic()
    for i, line in enumerate(lines):
        image = cv2.resize(
            cv2.imread(str(rec / line["jpeg"])), (WIDTH, h), interpolation=cv2.INTER_AREA
        )
        if image.shape[:2] != (h, WIDTH):
            image = cv2.resize(image, (WIDTH, h))
        lb, sk, ed = lab(image), skin(image), edges(image)
        c = cen[i]
        r = 0.45 * pitch(c)
        hull_pts = cv2.convexHull(c.astype(np.float32))
        # Grow the hull by half a square so edge squares' tall pieces are inside.
        centre = c.mean(axis=0)
        grown = (hull_pts - centre) * (
            1 + 0.5 * r / max(np.linalg.norm(hull_pts - centre, axis=-1).mean(), 1)
        ) + centre
        hull = np.zeros((h, WIDTH), np.uint8)
        cv2.fillConvexPoly(hull, grown.astype(np.int32), 1)
        hull_b = hull > 0
        border = hull_b & ~cv2.erode(hull, np.ones((5, 5), np.uint8)).astype(bool)
        discs = [((xx - x) ** 2 + (yy - y) ** 2) <= r * r for x, y in c]
        busy_now = {}
        for name, bg in bgs.items():
            if bg.model is None:
                diff = np.zeros((h, WIDTH), bool)
            else:
                delta = lb - bg.model
                delta[..., 0] -= np.int16(np.median(delta[..., 0][hull_b]))
                diff = np.abs(delta).sum(axis=-1) > DIFF_T
            busy = float(diff[hull_b].mean())
            res[f"fr_{name}"][i] = busy
            res[f"sq_{name}"][i] = [diff[d].mean() for d in discs]
            m = (diff & hull_b).astype(np.uint8)
            _, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            touching = set(np.unique(lab_cc[border & (lab_cc > 0)]).tolist())
            best = max((stats[j, cv2.CC_STAT_AREA] for j in touching), default=0)
            res[f"edge_blob_{name}"][i] = best / max(hull_b.sum(), 1)
            busy_now[name] = res[f"edge_blob_{name}"][i]
        g = bgs["gated"]
        new_skin = sk & ~(g.skin if g.skin is not None else np.zeros_like(sk))
        res["fr_skin"][i] = float(new_skin[hull_b].mean())
        res["sq_skin"][i] = [new_skin[d].mean() for d in discs]
        ed_bg = g.edge if g.edge is not None else ed.astype(float)
        res["sq_edge"][i] = [abs(ed[d].mean() - ed_bg[d].mean()) for d in discs]
        for name, bg in bgs.items():
            bg.offer(float(line["capture_ts"]), lb, sk, ed, busy_now[name])
    np.savez(out_dir / f"{take}.npz", **res)
    print(
        f"{take}: {n} frames, {(time.monotonic() - started) / n * 1000:.0f} ms/frame",
        flush=True,
    )


def main() -> None:
    feat_dir, out, *takes = sys.argv[1:]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for take in takes:
        run(Path(feat_dir), out_dir, take)


if __name__ == "__main__":
    main()
