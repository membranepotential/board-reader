"""Auto-frame probe, GPU part: locate the board with ChessQ's own attention on
zoomed-out canvases, crop to the proposal, read again, and score.

    flock /tmp/claude-1000/br-gpu.lock uv run python tools/probes/autocrop_run.py \
        HEADS.onnx OUT_DIR chessq        # ChessQ strategies
    flock ... autocrop_run.py HEADS.onnx OUT_DIR classical   # reads after the
        # classical detector's proposals (needs OUT_DIR/classical.jsonl from
        # autocrop_classical.py)

``HEADS.onnx`` is ``autocrop_model.py SOURCE OUT 1:0 2:0 3:0``. Per (image,
condition) it writes one JSON line:

* ``hand``   -- the read of the hand-drawn crop (the image as delivered, cut
  back out of the canvas at the canvas's resolution);
* ``pass1``  -- the grid fitted on one read of the whole canvas;
* ``c2f``    -- coarse to fine: the canvas cut to pass 1's box grown by 50 % on
  every side, read again, fitted again;
* ``tiles``  -- five tiles (four 60 % corners + the 60 % centre), each read and
  fitted; the tile whose fit has the most inlier squares, then refined like
  ``c2f`` (6 reads + pass 1). Skipped when pass 1 is clean (>= 60 inliers,
  RMS <= 0.15 squares): the fallback never runs then;
* ``e2e``    -- per proposal and margin: the canvas cut to the proposal's box
  grown by the margin, read, and scored against truth, plus that read's own
  grid fit and frame energy (what a verify step would see).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from autocrop_lib import (
    GATE_TAKES,
    LIVE_TAKES,
    SCRATCH,
    Grid,
    Item,
    Model,
    bbox,
    build_backdrop,
    corner_error_any,
    crop_box,
    fit_grid,
    iou,
    layout,
    place,
    score,
    stills,
    take_frames,
)

HEADS = ["attn1h0", "attn2h0", "attn3h0"]
#: The heads whose centroids the grid is fitted to (autocrop_heads.py).
FIT = ["attn1h0", "attn2h0", "attn3h0"]
MARGINS = (0.10,)  # the 5 / 15 % sweep was cut from the timeboxed run
REFINE = 0.5  # coarse-to-fine: grow the rough box by this on every side
TILES = [
    (0.0, 0.0, 0.6, 0.6),
    (0.4, 0.0, 1.0, 0.6),
    (0.0, 0.4, 0.6, 1.0),
    (0.4, 0.4, 1.0, 1.0),
    (0.2, 0.2, 0.8, 0.8),
]


@dataclass(frozen=True)
class Cond:
    share: float | None  # board bbox width / canvas width; None = as delivered
    backdrop: str | None  # "grey" | "room"
    width: int | None = None  # downscale the canvas to this width (a 1080p phone)
    #: Locate on the canvas downscaled to this long side and JPEG'd at quality 60:
    #: the phone's whole-frame preview (PREVIEW_LONG_SIDE, PREVIEW_QUALITY). The
    #: crop is still read at the canvas's resolution, as the phone crops its capture.
    preview: int | None = None


@cache
def conditions() -> dict[str, Cond]:
    out = {"as-is": Cond(None, None)}
    for bd in ("grey", "room"):
        for s in (0.5, 0.35, 0.25, 0.15):
            out[f"{bd}-{int(s * 100)}"] = Cond(s, bd)
    for s in (0.35, 0.25, 0.15):
        out[f"room1080-{int(s * 100)}"] = Cond(s, "room", 1920)
    for s in (0.5, 0.35, 0.25):
        out[f"prev480-{int(s * 100)}"] = Cond(s, "room", preview=480)
    return out


@cache
def load_items() -> list[Item]:
    return (
        stills()
        + [it for t in GATE_TAKES for it in take_frames(t, per_segment=1)]
        + [it for t in LIVE_TAKES for it in take_frames(t, per_segment=1, untruthed=8)]
    )


#: The canvases of the timeboxed run: the room backdrop at 35, 25 and 15 %.
TARGETED = {"room-35", "room-25", "room-15"}


def in_run(cname: str, item: Item) -> bool:
    """Every image is read as delivered (``as-is``). The canvases take about 30
    images: every third still, every eighth gate frame, three frames per live take."""
    if cname == "as-is":
        return True
    if cname not in TARGETED:
        return False
    same = [it.key for it in load_items() if it.group == item.group]
    i = same.index(item.key)
    if item.group == "still":
        return i % 3 == 0
    if item.group == "gate":
        return i % 8 == 0
    take = item.key.split("#")[0]
    in_take = [k for k in same if k.startswith(take + "#")]
    return in_take.index(item.key) in (0, 3, 6)


@cache
def backdrop() -> np.ndarray:
    path = SCRATCH / "backdrop.jpg"
    if not path.exists():
        cv2.imwrite(
            str(path),
            build_backdrop(load_items(), (6400, 3600)),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    return cv2.imread(str(path))


def seed_of(n: int, cond: Cond) -> int:
    return n * 101 + int((cond.share or 0) * 100)


def canvas_quad(item: Item, n: int, cond: Cond) -> np.ndarray:
    """The board quad on the condition's canvas, without building the canvas."""
    if cond.share is None:
        return item.quad
    W, _, tx, ty, _ = layout(item.quad, cond.share, seed_of(n, cond))
    k = cond.width / W if cond.width else 1.0
    return (item.quad + np.float32([tx, ty])) * np.float32(k)


def canvas_for(
    item: Item, n: int, cond: Cond
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """The condition's canvas, the board quad on it, and the hand-drawn crop."""
    img = cv2.imread(str(item.path))
    if cond.share is None:
        h, w = img.shape[:2]
        return img, item.quad, (0, 0, w, h)
    canvas, quad, hand = place(
        img,
        item.quad,
        cond.share,
        backdrop() if cond.backdrop == "room" else None,
        seed=seed_of(n, cond),
    )
    if cond.width:
        k = cond.width / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        quad = quad * np.float32(k)
        hand = tuple(round(v * k) for v in hand)
    return canvas, quad, hand


def locator_view(canvas: np.ndarray, cond: Cond) -> tuple[np.ndarray, float]:
    """What the locator sees (the canvas, or the phone's preview of it) and its
    scale against the canvas."""
    if cond.preview is None:
        return canvas, 1.0
    k = cond.preview / max(canvas.shape[:2])
    small = cv2.resize(canvas, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return cv2.imdecode(jpeg, cv2.IMREAD_COLOR), k


def cut(canvas: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    x0, y0, x1, y1 = box
    return canvas[y0:y1, x0:x1] if x1 - x0 >= 32 and y1 - y0 >= 32 else None


def shifted(g: Grid | None, box: tuple[int, ...], scale: float = 1.0) -> Grid | None:
    """``g`` fitted inside ``box`` -> the full image's pixels, divided by ``scale``."""
    if g is None:
        return None
    quad = (g.quad + np.float32(box[:2])) / np.float32(scale)
    return Grid(quad, g.r2, g.rms_sq, g.inliers, g.mass, g.energy)


def summary(g: Grid | None, truth: np.ndarray, ms: float) -> dict:
    if g is None:
        return {"found": False, "ms": ms}
    return {
        "found": True,
        "quad": g.quad.round(1).tolist(),
        "r2": g.r2,
        "rms_sq": g.rms_sq,
        "inliers": g.inliers,
        "mass": g.mass,
        "energy": g.energy,
        "corner_err": corner_error_any(g.quad, truth),
        "iou": iou(g.box, bbox(truth)),
        "ms": ms,
    }


class Locator:
    def __init__(self, model: Model) -> None:
        self.model = model

    def locate(self, image: np.ndarray) -> tuple[Grid | None, float]:
        r = self.model(image)
        h, w = image.shape[:2]
        return fit_grid(r, FIT, w, h), r.ms

    def refine(self, canvas: np.ndarray, g: Grid | None) -> tuple[Grid | None, float]:
        if g is None:
            return None, 0.0
        h, w = canvas.shape[:2]
        box = crop_box(g.box, REFINE, w, h)
        part = cut(canvas, box)
        if part is None:
            return None, 0.0
        g2, ms = self.locate(part)
        return shifted(g2, box), ms

    def verify_read(self, canvas: np.ndarray, box: tuple[int, ...], item: Item) -> dict:
        part = cut(canvas, box)
        if part is None:
            return {"read": False}
        r = self.model(part)
        h, w = part.shape[:2]
        g = fit_grid(r, FIT, w, h)
        return {
            "read": True,
            "box": list(box),
            "correct": score(r.logits, item.truth) if item.truth else None,
            "argmax": r.logits.argmax(-1).tolist(),
            "energy": g.energy if g else None,
            "r2": g.r2 if g else None,
            "rms_sq": g.rms_sq if g else None,
            "inliers": g.inliers if g else None,
        }


def run_chessq(loc: Locator, item: Item, n: int, cname: str, cond: Cond) -> dict:
    canvas, quad, hand = canvas_for(item, n, cond)
    h, w = canvas.shape[:2]
    rec: dict = {"key": item.key, "group": item.group, "cond": cname, "size": [w, h]}
    # At scale 1 the hand-drawn crop is the delivered image itself (bar the 24 px
    # feather), so its read is the as-is record's; only a downscaled canvas
    # (room1080) reads its own.
    if cond.width is not None:
        rec["hand"] = loc.verify_read(canvas, hand, item)
    # Every grid below is located on ``view`` and stored in canvas pixels.
    view, k = locator_view(canvas, cond)
    vh, vw = view.shape[:2]
    g1, ms1 = loc.locate(view)
    g2, ms2 = loc.refine(view, g1)
    g1, g2 = shifted(g1, (0, 0), k), shifted(g2, (0, 0), k)
    rec["pass1"] = summary(g1, quad, ms1)
    rec["c2f"] = summary(g2, quad, ms2)
    tile_grids = []
    # Tiles are the fallback for an untrusted pass 1; a clean pass 1 skips them
    # (looser than any rule autocrop_analyze applies, so every fallback has them).
    clean = g1 is not None and g1.inliers >= 60 and g1.rms_sq <= 0.15
    for fx0, fy0, fx1, fy1 in [] if clean else TILES:
        box = (int(fx0 * vw), int(fy0 * vh), int(fx1 * vw), int(fy1 * vh))
        g, ms = loc.locate(view[box[1] : box[3], box[0] : box[2]])
        tile_grids.append((shifted(g, box), ms))
    best = max(
        (g for g, _ in tile_grids if g is not None),
        key=lambda g: (g.inliers, -g.rms_sq),
        default=None,
    )
    gt, mst = loc.refine(view, best)
    gt = shifted(gt, (0, 0), k)
    rec["tile_each"] = [summary(shifted(g, (0, 0), k), quad, ms) for g, ms in tile_grids]
    rec["tiles"] = summary(gt, quad, mst)
    rec["e2e"] = {}
    # The margin sweep runs on the realistic canvases (room, preview); pass 1
    # alone is a candidate only while the board is big (>= 50 %).
    realistic = cond.share is None or (cond.backdrop == "room" and cond.width is None)
    sweep = MARGINS if realistic else (0.10,)
    plans = [("c2f", g2, sweep), ("tiles", gt, (0.10,))]
    if cond.share is None or cond.share >= 0.5:
        plans.append(("pass1", g1, (0.10,)))
    for name, g, margins in plans:
        if g is None:
            continue
        rec["e2e"][name] = {
            str(m): loc.verify_read(canvas, crop_box(g.box, m, w, h), item) for m in margins
        }
    return rec


def run_classical(
    loc: Locator, item: Item, n: int, cname: str, cond: Cond, quads: dict
) -> dict:
    canvas, _, _ = canvas_for(item, n, cond)
    h, w = canvas.shape[:2]
    rec: dict = {"key": item.key, "cond": cname, "e2e": {}}
    # classical.jsonl stores quads in canvas pixels (autocrop_classical._job).
    seen: dict[str, dict] = {}
    for ladder, q in quads.items():
        if q is None:
            continue
        box = crop_box(bbox(np.array(q, np.float32)), 0.10, w, h)
        k = str(box)
        if k not in seen:
            seen[k] = loc.verify_read(canvas, box, item)
        rec["e2e"][ladder] = seen[k]
    return rec


def main(model_path: Path, out_dir: Path, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loc = Locator(Model(model_path, HEADS))
    items = load_items()
    conds = conditions()
    path = (
        out_dir / f"{mode}.jsonl" if mode == "chessq" else out_dir / "classical_e2e.jsonl"
    )
    done = set()
    if path.exists():
        done = {
            (r["key"], r["cond"]) for r in map(json.loads, path.read_text().splitlines())
        }
    classical: dict[tuple[str, str], dict] = {}
    if mode == "classical":
        for r in map(json.loads, (out_dir / "classical.jsonl").read_text().splitlines()):
            classical.setdefault((r["key"], r["cond"]), {})[r["ladder"]] = r["quad"]
    print(f"{len(items)} items x {len(conds)} conditions", flush=True)
    with path.open("a") as f:
        for cname, cond in conds.items():
            for n, item in enumerate(items):
                if (item.key, cname) in done or not in_run(cname, item):
                    continue
                if mode == "chessq":
                    rec = run_chessq(loc, item, n, cname, cond)
                else:
                    if (item.key, cname) not in classical:
                        continue
                    rec = run_classical(
                        loc, item, n, cname, cond, classical[(item.key, cname)]
                    )
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"  {cname} done", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
