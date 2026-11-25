"""Shared pieces of the auto-frame probe (docs/reports/2026-11-25-autocrop-probe.md):
the images and their board outlines, the zoomed-out canvases, ChessQ's grid fit
from its square-localising attention, and the scoring.

Coordinates: a board quad is four (x, y) image points in chess order
a8, h8, h1, a1 -- the outer corners of the 8x8 playing area. "Board coordinates"
put a8's outer corner at (0, 0) and h1's at (8, 8), so query ``i`` (FEN order,
a8..h8, a7..h1) has its square centre at ``(i % 8 + 0.5, i // 8 + 0.5)``.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from board_reader_inference.recognition.chessq import (
    INPUT_PX,
    frame_energy,
    preprocess_frame,
)

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(
    "/tmp/claude-1000/-home-felix-Documents-chess-board-reader/"
    "8a2d0a57-e76c-4166-856a-3013be0fed9f/scratchpad/autocrop"
)
G = INPUT_PX // 14  # 46 patches a side
CLASSES = ".PNBRQKpnbrqk"
BOARD_CORNERS = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)
SQUARE_CENTRES = np.array([[i % 8 + 0.5, i // 8 + 0.5] for i in range(64)], np.float32)

# --------------------------------------------------------------------- images


@dataclass(frozen=True)
class Item:
    key: str
    group: str  # "still", "gate", "live"
    path: Path
    quad: np.ndarray  # (4, 2) a8 h8 h1 a1, in this image's pixels
    truth: str | None  # FEN placement, None when the frame has no truth


def fen_grid(placement: str) -> list[str]:
    out: list[str] = []
    for ch in placement.split()[0].replace("/", ""):
        out.extend("." * int(ch) if ch.isdigit() else ch)
    if len(out) != 64:
        raise ValueError(placement)
    return out


def stills() -> list[Item]:
    """The 21 phone photos with hand-checked corners (curated + heldout-0831)."""
    out = []
    gt = json.loads((ROOT / "data/eval/ground_truth.json").read_text())
    for k, v in gt.items():
        out.append(
            Item(
                f"still:{k}",
                "still",
                ROOT / v["source"],
                np.array(v["corners_a8_h8_h1_a1"], np.float32),
                v["placement"],
            )
        )
    for f in sorted(glob.glob(str(ROOT / "data/eval/heldout-20260831/*.label.json"))):
        v = json.loads(Path(f).read_text())
        quad = np.array([float(x) for x in v["corners"].split(",")], np.float32)
        out.append(
            Item(
                f"still:{Path(f).name.split('.')[0]}",
                "still",
                Path(f).parent / v["photo"],
                quad.reshape(4, 2),
                v["placement"],
            )
        )
    return out


#: Board outlines of the recorded takes, per frame row: ``autocrop_label.py``
#: (the classical detector plus a temporal median, checked by eye on overlays).
#: The gate takes' own ``frames.ndjson`` corners are too loose to be the truth.
QUADS = SCRATCH / "quads"


def take_frames(take: str, per_segment: int, untruthed: int = 10) -> list[Item]:
    """Labelled frames of one recorded take.

    With a ``truth.json``: ``per_segment`` frames from the middle half of every
    truth segment (away from moves). Without one: ``untruthed`` frames spread
    over the take, with no placement."""
    rec = ROOT / "recordings" / take
    quads = {int(r): q for r, q in json.loads((QUADS / f"{take}.json").read_text()).items()}
    lines = [
        json.loads(s) for s in (rec / "frames.ndjson").read_text().splitlines() if s.strip()
    ]
    group = "gate" if take.startswith("202608") else "live"

    def item(row: int, placement: str | None) -> Item:
        return Item(
            f"{group}:{take}#{row}",
            group,
            rec / lines[row]["jpeg"],
            np.array(quads[row], np.float32),
            placement,
        )

    if not (rec / "truth.json").exists():
        rows = sorted(quads)
        return [item(r, None) for r in rows[:: max(1, len(rows) // untruthed)][:untruthed]]
    segs = json.loads((rec / "truth.json").read_text())["segments"]
    row_of: dict[int, int] = {}
    for r, ln in enumerate(lines):
        row_of.setdefault(ln["seq"], r)
    last = max(row_of)
    out = []
    for i, seg in enumerate(segs):
        if not isinstance(seg.get("placement"), str):
            continue
        lo = seg["from_seq"]
        hi = segs[i + 1]["from_seq"] if i + 1 < len(segs) else last + 1
        a, b = lo + (hi - lo) // 4, hi - (hi - lo) // 4
        cand = [row_of[s] for s in range(a, b) if s in row_of and row_of[s] in quads]
        for r in cand[:: max(1, len(cand) // per_segment)][:per_segment]:
            out.append(item(r, seg["placement"]))
    return out


GATE_TAKES = [
    "20260813-193308",
    "20260827-163352",
    "20260827-170333",
    "20260827-172736",
    "20260827-175539",
    "20260827-181802",
    "20260828-111421",
]
LIVE_TAKES = ["20261124-200631", "20261124-204025", "20261124-221950", "20261125-004038"]


# ------------------------------------------------------------------ canvases

ASPECT = 16 / 9  # the phone's full 1080p frame, landscape


def bbox(quad: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    return float(x0), float(y0), float(x1), float(y1)


def build_backdrop(sources: list[Item], size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """A patchwork of board-free strips of real frames: table, sofa, rug,
    clothes, pavement -- what a zoomed-out phone frame shows around the board.
    Each strip is the part of a source image above, below, left or right of its
    board (plus 8 % of the board's size, so no board border sneaks in)."""
    rng = np.random.default_rng(seed)
    strips = []
    for it in sources:
        img = cv2.imread(str(it.path))
        h, w = img.shape[:2]
        x0, y0, x1, y1 = bbox(it.quad)
        mx, my = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
        x0, y0, x1, y1 = int(x0 - mx), int(y0 - my), int(x1 + mx) + 1, int(y1 + my) + 1
        for s in (
            img[: max(0, y0)],
            img[min(h, y1) :],
            img[:, : max(0, x0)],
            img[:, min(w, x1) :],
        ):
            if min(s.shape[:2]) >= 80:
                strips.append(s)
    W, H = size
    canvas = np.zeros((H, W, 3), np.uint8)
    y = 0
    while y < H:
        # Rows 150-450 px tall: about the scale of the strips' own pixels, so a
        # board 800 px wide sits among room details of their natural size.
        row_h = int(rng.integers(150, 450))
        x = 0
        while x < W:
            s = strips[int(rng.integers(len(strips)))]
            sh, sw = s.shape[:2]
            scale = row_h / sh
            tw = max(40, int(sw * scale))
            # A random flip, so a strip used twice does not repeat exactly.
            tile = cv2.resize(s, (tw, row_h), interpolation=cv2.INTER_AREA)
            if rng.random() < 0.5:
                tile = tile[:, ::-1]
            take = min(tw, W - x)
            canvas[y : min(H, y + row_h), x : x + take] = tile[: min(row_h, H - y), :take]
            x += take
        y += row_h
    return canvas


def layout(
    quad: np.ndarray, share: float, seed: int
) -> tuple[int, int, int, int, np.random.Generator]:
    """Where :func:`place` puts the image: the canvas size (W, H), the image's
    offset (tx, ty) on it, and the random stream, advanced past those draws.
    Cheap, so a scorer can recover a canvas's board quad without its pixels."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bbox(quad)
    bw, bh = x1 - x0, y1 - y0
    W = round(bw / share)
    H = round(W / ASPECT)
    if bh > 0.96 * H:
        raise ValueError("board too tall for the canvas")
    # The board's top-left on the canvas: anywhere with a 2 % margin.
    bx = rng.uniform(0.02 * W, W - bw - 0.02 * W)
    by = rng.uniform(0.02 * H, H - bh - 0.02 * H)
    return W, H, round(bx - x0), round(by - y0), rng


def place(
    img: np.ndarray,
    quad: np.ndarray,
    share: float,
    backdrop: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Put ``img`` at scale 1 on a 16:9 canvas whose width makes the board's
    bounding box ``share`` of it, at a random position (board fully inside).

    Outside the image the canvas shows ``backdrop`` (a random window of it), or
    flat RGB 128 when ``backdrop`` is None. The image's own edges are feathered
    over 24 px so the paste is no hard rectangle. Returns the canvas, the quad
    in canvas pixels, and the image's rectangle on the canvas (x0, y0, x1, y1):
    the hand-drawn crop."""
    W, H, tx, ty, rng = layout(quad, share, seed)
    if backdrop is None:
        canvas = np.full((H, W, 3), 128, np.uint8)
    else:
        bH, bW = backdrop.shape[:2]
        if bW < W or bH < H:
            k = max(W / bW, H / bH)
            backdrop = cv2.resize(backdrop, (int(np.ceil(bW * k)), int(np.ceil(bH * k))))
            bH, bW = backdrop.shape[:2]
        ox, oy = int(rng.integers(0, bW - W + 1)), int(rng.integers(0, bH - H + 1))
        canvas = backdrop[oy : oy + H, ox : ox + W].copy()
    h, w = img.shape[:2]
    # The part of the image that lands on the canvas.
    cx0, cy0, cx1, cy1 = max(0, tx), max(0, ty), min(W, tx + w), min(H, ty + h)
    src = img[cy0 - ty : cy1 - ty, cx0 - tx : cx1 - tx].astype(np.float32)
    ramp = 24
    ay = np.clip(np.minimum(np.arange(h) + 1, h - np.arange(h)) / ramp, 0, 1)
    ax = np.clip(np.minimum(np.arange(w) + 1, w - np.arange(w)) / ramp, 0, 1)
    alpha = np.minimum.outer(ay, ax)[cy0 - ty : cy1 - ty, cx0 - tx : cx1 - tx, None]
    dst = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = (alpha * src + (1 - alpha) * dst).astype(np.uint8)
    return canvas, quad + np.float32([tx, ty]), (cx0, cy0, cx1, cy1)


def crop_box(
    box: tuple[float, float, float, float], margin: float, w: int, h: int
) -> tuple[int, int, int, int]:
    """A board box grown by ``margin`` of its own width/height on every side,
    clipped to the image: the rectangle /control would set."""
    x0, y0, x1, y1 = box
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (
        max(0, int(np.floor(x0 - mx))),
        max(0, int(np.floor(y0 - my))),
        min(w, int(np.ceil(x1 + mx))),
        min(h, int(np.ceil(y1 + my))),
    )


# ------------------------------------------------------------------- ChessQ


@dataclass(frozen=True)
class Read:
    logits: np.ndarray  # (64, 13)
    maps: dict[str, np.ndarray]  # head name -> (64, G, G), CLS dropped
    ms: float


class Model:
    """A scratch export from ``autocrop_model.py`` with sliced heads."""

    def __init__(self, path: Path, heads: list[str], cpu: bool = False) -> None:
        import onnxruntime as ort

        from board_reader_inference.recognition.onnxrt import cuda_providers

        providers = ["CPUExecutionProvider"] if cpu else cuda_providers()
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.heads = heads

    def __call__(self, image: np.ndarray) -> Read:
        import time

        x = preprocess_frame(image)
        t = time.perf_counter()
        logits, *attn = self.sess.run(["logits", *self.heads], {"image": x})
        ms = (time.perf_counter() - t) * 1000
        maps = {}
        for name, a in zip(self.heads, attn):
            a = a[0]  # (64, 2117) for a sliced head
            maps[name] = a[:, 1:].reshape(64, G, G)
        return Read(logits[0], maps, ms)


def centroids(amap: np.ndarray, radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Per query: the attention centroid in a (2r+1)^2 window around its peak,
    in patch units (x, y), and the mass inside that window (how peaked it is)."""
    n = amap.shape[0]
    p = amap / np.clip(amap.sum(axis=(1, 2), keepdims=True), 1e-12, None)
    flat = p.reshape(n, -1).argmax(axis=1)
    py, px = np.divmod(flat, G)
    pts = np.zeros((n, 2), np.float32)
    mass = np.zeros(n, np.float32)
    for i in range(n):
        ya, yb = max(0, py[i] - radius), min(G, py[i] + radius + 1)
        xa, xb = max(0, px[i] - radius), min(G, px[i] + radius + 1)
        win = p[i, ya:yb, xa:xb]
        m = win.sum()
        yy, xx = np.mgrid[ya:yb, xa:xb]
        pts[i] = ((win * xx).sum() / m, (win * yy).sum() / m)
        mass[i] = m
    return pts, mass


@dataclass(frozen=True)
class Grid:
    """ChessQ's board as an 8x8 grid fitted to its square centroids."""

    quad: np.ndarray  # (4, 2) a8 h8 h1 a1, image pixels
    r2: float  # of the homography fit, all 64 centres
    rms_sq: float  # RMS residual of the inliers, in squares
    inliers: int  # centres within 0.35 squares of the fit
    mass: float  # median attention mass in the peak window
    energy: float  # frame energy (chessq.frame_energy)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return bbox(self.quad)


def fit_grid(read: Read, heads: list[str], w: int, h: int) -> Grid | None:
    """Fit board coordinates -> image pixels to the attention centroids of
    ``heads`` (averaged when several). LMedS ignores up to half the squares
    (a hand, a square outside the frame); R^2 is over all 64."""
    pts, masses = zip(*(centroids(read.maps[k]) for k in heads))
    pt = np.mean(pts, axis=0)
    # Patch units -> 644 px -> the image's pixels (the squash resize).
    img = np.stack(
        [(pt[:, 0] + 0.5) * 14 * w / INPUT_PX, (pt[:, 1] + 0.5) * 14 * h / INPUT_PX], 1
    )
    H, _ = cv2.findHomography(SQUARE_CENTRES, img, cv2.LMEDS)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(SQUARE_CENTRES[:, None], H)[:, 0]
    quad = cv2.perspectiveTransform(BOARD_CORNERS[:, None], H)[:, 0]
    sq = max(1e-6, float(np.sqrt(abs(cv2.contourArea(quad)))) / 8)
    res = np.linalg.norm(proj - img, axis=1)
    ss_tot = float(((img - img.mean(0)) ** 2).sum())
    r2 = 1 - float((res**2).sum()) / max(ss_tot, 1e-9)
    inl = res < 0.35 * sq
    rms = float(np.sqrt((res[inl] ** 2).mean()) / sq) if inl.any() else 9.9
    return Grid(
        quad.astype(np.float32),
        r2,
        rms,
        int(inl.sum()),
        float(np.median(np.mean(masses, axis=0))),
        frame_energy(read.logits[None]),
    )


# ------------------------------------------------------------------- scoring


def corner_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean corner distance in % of the truth board's bbox width. Chess order is
    compared as given (ChessQ knows which corner is a8); a classical quad has
    only geometric order, so the caller passes ``ordered=False`` via
    :func:`corner_error_any`."""
    x0, _, x1, _ = bbox(truth)
    return float(np.linalg.norm(pred - truth, axis=1).mean() / (x1 - x0) * 100)


def corner_error_any(pred: np.ndarray, truth: np.ndarray) -> float:
    """:func:`corner_error` under the best cyclic order and direction of ``pred``."""
    cands = [np.roll(p, k, axis=0) for p in (pred, pred[::-1]) for k in range(4)]
    return min(corner_error(c, truth) for c in cands)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(logits: np.ndarray, placement: str) -> int:
    """Correct squares of ChessQ's argmax read against the truth placement."""
    pred = [CLASSES[j] for j in logits.argmax(-1)]
    return sum(p == t for p, t in zip(pred, fen_grid(placement)))
