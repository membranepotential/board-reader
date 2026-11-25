"""Auto-frame follow-up, GPU part: locate the board on the phone's whole-frame
PREVIEW (480 px long side, JPEG q 0.6, every 1.5 s) and measure what a
camera-moved ("drift") check on those previews would see.
docs/reports/2026-11-25-autocrop-probe.md, section "Follow-up: the preview".

    AUTOCROP_SCRATCH=<scratch> flock /tmp/claude-1000/br-gpu.lock \
      uv run python tools/probes/autocrop_preview.py locate OUT_DIR
    ... autocrop_preview.py drift OUT_DIR
    CUDA_VISIBLE_DEVICES= uv run python tools/probes/autocrop_preview.py cost OUT_DIR

It reads the served attention files (``models/chessq/*-attn.onnx``, output
``square_attention``) and the probe's outlines (``SCRATCH/quads``, from
``autocrop_label.py``).

* ``locate``: the probe's 30-image subset plus every live-take image, on the
  room canvas at the board shares as-is / 50 / 35 / 25 %. Per image and share:
  the single pass on the preview, and coarse to fine on the preview (pass 1's box
  grown by 50 %, cut out of the 480 px preview and read again: an upsample).
* ``drift``: per recorded take, the frames 1.5 s apart (the preview cadence) on
  one fixed canvas (the camera did not move; hands and pieces do), located on
  their previews. Then "moved" events: from an event preview on, the full canvas
  is shifted, rotated or zoomed before it is downscaled, for three previews.
  Every record carries the located quad and the truth quad (both in canvas
  pixels) so ``autocrop_preview_analyze.py`` can compute any drift metric.
* ``cost``: ms per preview read on the CPU int8 file (run without the GPU).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from autocrop_lib import (
    GATE_TAKES,
    LIVE_TAKES,
    QUADS,
    ROOT,
    Grid,
    Read,
    bbox,
    corner_error_any,
    crop_box,
    fit_grid,
    layout,
    place,
)
from autocrop_run import Cond, backdrop, canvas_for, cut, in_run, load_items, seed_of

from board_reader_inference.recognition.chessq import preprocess_frame

MODELS = ROOT / "models/chessq"
FP32 = MODELS / "cq-vit-s-644-v6-seed1-onnx-fp32-attn.onnx"
INT8 = MODELS / "cq-vit-s-644-v6-seed1-onnx-int8-r1-attn.onnx"
HEADS = ["L1H0", "L2H0", "L3H0"]
#: The phone's preview (packages/viewer/src/lib/components/camera/state.ts).
PREVIEW_LONG_SIDE = 480
PREVIEW_QUALITY = 60  # canvas.toBlob quality 0.6
PREVIEW_S = 1.5
REFINE = 0.5
SHARES = {"as-is": None, "50": 0.5, "35": 0.35, "25": 0.25}
#: The full frame's board share for the drift run: a live phone take's board is
#: about 47 % of the uncropped 1080p ultra-wide frame (autocrop probe, Data).
DRIFT_SHARE = 0.5
PREVIEWS_PER_TAKE = 40
EVENTS_PER_TAKE = 2
EVENT_LEN = 3  # previews after a move (4.5 s)
#: (kind, magnitude): shift in % of the board width (diagonal), rotation in
#: degrees about the frame centre, zoom in % about the frame centre.
MOVES = (
    [("shift", s) for s in (1, 2, 3, 5, 10)]
    + [("rotate", r) for r in (1, 2, 3, 5, 10)]
    + [("zoom", z) for z in (-20, -10, -5, 5, 10, 20)]
)


class Reader:
    """A served ``-attn`` file: logits plus ``square_attention`` (1, 3, 64, 46, 46)."""

    def __init__(self, path: Path, cpu: bool = False) -> None:
        import onnxruntime as ort

        from board_reader_inference.recognition.onnxrt import cuda_providers

        providers = ["CPUExecutionProvider"] if cpu else cuda_providers()
        self.sess = ort.InferenceSession(str(path), providers=providers)

    def __call__(self, image: np.ndarray) -> Read:
        x = preprocess_frame(image)
        t = time.perf_counter()
        logits, attn = self.sess.run(["logits", "square_attention"], {"image": x})
        ms = (time.perf_counter() - t) * 1000
        return Read(logits[0], dict(zip(HEADS, attn[0])), ms)


def preview(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """The phone's preview of a whole frame, and its scale against the frame."""
    k = PREVIEW_LONG_SIDE / max(frame.shape[:2])
    small = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
    return cv2.imdecode(jpeg, cv2.IMREAD_COLOR), k


def locate(reader: Reader, image: np.ndarray) -> tuple[Grid | None, float, float]:
    """One read and its grid fit; the read's and the fit's ms."""
    r = reader(image)
    t = time.perf_counter()
    h, w = image.shape[:2]
    g = fit_grid(r, HEADS, w, h)
    return g, r.ms, (time.perf_counter() - t) * 1000


def scaled(g: Grid | None, off: tuple[int, int], k: float) -> Grid | None:
    if g is None:
        return None
    quad = (g.quad + np.float32(off)) / np.float32(k)
    return Grid(quad, g.r2, g.rms_sq, g.inliers, g.mass, g.energy)


def on_preview(reader: Reader, frame: np.ndarray, refine: bool = True) -> dict:
    """Single pass and (``refine``) coarse to fine on the frame's preview, in
    frame pixels."""
    view, k = preview(frame)
    vh, vw = view.shape[:2]
    g1, ms, fit_ms = locate(reader, view)
    out = {"pass1": grid_rec(scaled(g1, (0, 0), k)), "ms": ms, "fit_ms": fit_ms}
    g2 = None
    if refine and g1 is not None:
        box = crop_box(g1.box, REFINE, vw, vh)
        part = cut(view, box)
        if part is not None:
            g2, _, _ = locate(reader, part)
            g2 = scaled(g2, box[:2], k)
    out["c2f"] = grid_rec(g2)
    return out


def grid_rec(g: Grid | None) -> dict | None:
    if g is None:
        return None
    return {"quad": g.quad.round(2).tolist(), "r2": g.r2, "inliers": g.inliers}


# ------------------------------------------------------------------- locate


def run_locate(reader: Reader, out: Path) -> None:
    items = load_items()
    with out.open("w") as f:
        for n, item in enumerate(items):
            if not (item.group == "live" or in_run("room-35", item)):
                continue
            for name, share in SHARES.items():
                cond = Cond(share, "room" if share else None)
                canvas, quad, _ = canvas_for(item, n, cond)
                rec = {"key": item.key, "group": item.group, "share": name}
                rec |= on_preview(reader, canvas)
                for s in ("pass1", "c2f"):
                    if rec[s] is not None:
                        rec[s]["corner_err"] = corner_error_any(
                            np.array(rec[s]["quad"], np.float32), quad
                        )
                rec["board_px"] = (bbox(quad)[2] - bbox(quad)[0]) * preview(canvas)[1]
                f.write(json.dumps(rec) + "\n")
            f.flush()
    print(f"wrote {out}")


# -------------------------------------------------------------------- drift


def frames_of(take: str) -> list[tuple[int, Path]]:
    """(row, jpeg) of the take's frames 1.5 s apart, at most PREVIEWS_PER_TAKE."""
    rec = ROOT / "recordings" / take
    lines = [
        json.loads(s) for s in (rec / "frames.ndjson").read_text().splitlines() if s.strip()
    ]
    rows, due = [], -np.inf
    for r, ln in enumerate(lines):
        t = ln.get("capture_ts") or ln["received_ts"]
        if t >= due:
            rows.append((r, rec / ln["jpeg"]))
            due = t + PREVIEW_S
    return rows[:PREVIEWS_PER_TAKE]


def truth_of(quads: dict[int, np.ndarray], row: int) -> np.ndarray:
    """The outline of the nearest labelled row (outlines are ±20-row medians)."""
    return quads[min(quads, key=lambda r: abs(r - row))]


def move_matrix(kind: str, mag: float, w: int, h: int, board_w: float) -> np.ndarray:
    """A 2x3 affine of the whole frame: what the phone moving looks like."""
    c = (w / 2, h / 2)
    if kind == "shift":
        d = mag / 100 * board_w / np.sqrt(2)
        return np.float32([[1, 0, d], [0, 1, d]])
    if kind == "rotate":
        return cv2.getRotationMatrix2D(c, mag, 1.0).astype(np.float32)
    return cv2.getRotationMatrix2D(c, 0, 1 + mag / 100).astype(np.float32)


def run_drift(reader: Reader, out: Path) -> None:
    bd = backdrop()
    with out.open("w") as f:
        for ti, take in enumerate(GATE_TAKES + LIVE_TAKES):
            quads = {
                int(r): np.array(q, np.float32)
                for r, q in json.loads((QUADS / f"{take}.json").read_text()).items()
            }
            # One canvas layout per take: the take's median outline decides it,
            # so the recorded frame sits at the same place in every preview.
            median = np.median(list(quads.values()), axis=0).astype(np.float32)
            seed = seed_of(ti, Cond(DRIFT_SHARE, "room"))
            _, _, tx, ty, _ = layout(median, DRIFT_SHARE, seed)
            shift = np.float32([tx, ty])  # recorded-frame pixels -> canvas pixels
            rows = frames_of(take)
            canvases = []
            for i, (row, jpeg) in enumerate(rows):
                img = cv2.imread(str(jpeg))
                tq = truth_of(quads, row)
                for share in (DRIFT_SHARE,):
                    if share is None:
                        canvas, off = img, np.float32([0, 0])
                    else:
                        canvas, _, _ = place(img, median, share, bd, seed)
                        off = shift
                        canvases.append(canvas)
                    rec = {
                        "take": take,
                        "i": i,
                        "row": row,
                        "share": "as-is" if share is None else str(int(share * 100)),
                        "move": None,
                        "truth": (tq + off).round(2).tolist(),
                    }
                    rec |= on_preview(reader, canvas, refine=False)
                    f.write(json.dumps(rec) + "\n")
            # Moved events on the 50 % canvas: preview e .. e+EVENT_LEN-1 moved.
            n = len(canvases)
            starts = np.linspace(3, n - EVENT_LEN, EVENTS_PER_TAKE).astype(int)
            for e in starts:
                for kind, mag in MOVES:
                    for j in range(e, min(n, e + EVENT_LEN)):
                        canvas = canvases[j]
                        h, w = canvas.shape[:2]
                        row = rows[j][0]
                        tq = truth_of(quads, row) + shift
                        board_w = bbox(tq)[2] - bbox(tq)[0]
                        M = move_matrix(kind, mag, w, h, board_w)
                        moved = cv2.warpAffine(
                            canvas,
                            M,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT,
                        )
                        mtq = cv2.transform(tq[None], M)[0]
                        rec = {
                            "take": take,
                            "i": j,
                            "row": row,
                            "share": "50",
                            "move": [kind, mag, int(e)],
                            "truth": mtq.round(2).tolist(),
                        }
                        rec |= on_preview(reader, moved, refine=False)
                        f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  {take}: {len(rows)} previews", flush=True)
    print(f"wrote {out}")


# --------------------------------------------------------------------- cost


def run_cost(out: Path) -> None:
    """CPU int8: one read of 20 previews (the GPU fp32 ms are in the drift run)."""
    reader = Reader(INT8, cpu=True)
    items = [it for it in load_items() if it.group == "live"][:20]
    ms, fit = [], []
    for it in items:
        view, _ = preview(cv2.imread(str(it.path)))
        _, m, fm = locate(reader, view)
        ms.append(m)
        fit.append(fm)
    rec = {"int8_cpu_ms": ms[1:], "fit_ms": fit[1:]}
    out.write_text(json.dumps(rec))
    print(f"int8 CPU median {np.median(ms[1:]):.0f} ms, fit {np.median(fit[1:]):.1f} ms")


if __name__ == "__main__":
    mode, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "cost":
        run_cost(out_dir / "preview_cost.json")
    else:
        r = Reader(FP32)
        {"locate": run_locate, "drift": run_drift}[mode](
            r, out_dir / f"preview_{mode}.jsonl"
        )
