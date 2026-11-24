"""Read every frame of a take through the probe model (hand_model.py) and cache
compact per-frame features for the hand-signal probe.

    flock /tmp/claude-1000/br-gpu.lock uv run python tools/probes/hand_extract.py \
        MODEL.onnx OUT_DIR TAKE [TAKE ...]

Writes OUT_DIR/<take>.npz, one row per line of frames.ndjson (in file order, so a
seq restart is kept as its own run of rows):

* ``seq``, ``ts``, ``stability``, ``jpeg`` — the index line's fields;
* ``logits`` (n, 64, 13) and ``direction`` (n, 2) — the release outputs;
* ``query`` (n, 64, 384) f16 — each square query's final embedding;
* ``ent``, ``cls``, ``cx``, ``cy``, ``spread`` (n, 4, 8, 64) — per decoder layer and
  head, the cross-attention's entropy over the patches, its mass on the CLS token,
  its centroid on the 46x46 patch grid and its RMS radius around that centroid.
  Layer 2 head 0 and layer 3 head 0 are the heads that localise each square (a
  grid fit of their centroids has R^2 >= 0.99); the others attend diffusely;
* ``amap`` (n, 64, 23, 23) f16 — layer 2 head 0's attention map, 2x2 pooled;
* ``mem`` (n, 46, 46, 32) f16 — the patch tokens projected on the take's first
  frame's top 32 principal axes (``basis``, ``mean``), for reference comparisons.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from board_reader_inference.recognition.chessq import preprocess_frame
from board_reader_inference.recognition.onnxrt import cuda_providers

ROOT = Path(__file__).resolve().parents[2]
G = 46  # 644 / 14 patches a side
PCA = 32
_YY, _XX = np.mgrid[0:G, 0:G].astype(np.float32)


def attention_stats(attn: np.ndarray) -> tuple[np.ndarray, ...]:
    """(8, 64, 1+G*G) softmax -> per head: entropy, CLS mass, centroid x/y, spread, map."""
    cls = attn[:, :, 0]
    p = attn[:, :, 1:] / np.clip(attn[:, :, 1:].sum(axis=2, keepdims=True), 1e-12, None)
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=2)
    grid = p.reshape(8, 64, G, G)
    cx = (grid * _XX).sum(axis=(2, 3))
    cy = (grid * _YY).sum(axis=(2, 3))
    spread = np.sqrt(
        (grid * ((_XX - cx[..., None, None]) ** 2 + (_YY - cy[..., None, None]) ** 2)).sum(
            axis=(2, 3)
        )
    )
    return ent, cls, cx, cy, spread, grid


def extract(sess: ort.InferenceSession, take: str, out: Path) -> None:
    rec = ROOT / "recordings" / take
    lines = [
        json.loads(s) for s in (rec / "frames.ndjson").read_text().splitlines() if s.strip()
    ]
    n = len(lines)
    names = ["logits", "direction", "query", "memory", "attn0", "attn1", "attn2", "attn3"]
    acc: dict[str, np.ndarray] = {
        "logits": np.zeros((n, 64, 13), np.float32),
        "direction": np.zeros((n, 2), np.float32),
        "query": np.zeros((n, 64, 384), np.float16),
        **{
            k: np.zeros((n, 4, 8, 64), np.float32)
            for k in ("ent", "cls", "cx", "cy", "spread")
        },
        "amap": np.zeros((n, 64, G // 2, G // 2), np.float16),
        "mem": np.zeros((n, G, G, PCA), np.float16),
    }
    basis = mean = None
    started = time.monotonic()
    for i, line in enumerate(lines):
        image = cv2.imread(str(rec / line["jpeg"]))
        logits, direction, query, memory, *attn = sess.run(
            names, {"image": preprocess_frame(image)}
        )
        acc["logits"][i], acc["direction"][i], acc["query"][i] = (
            logits[0],
            direction[0],
            query[0],
        )
        for layer, a in enumerate(attn):
            ent, cls, cx, cy, spread, grid = attention_stats(a[0])
            for k, v in zip(
                ("ent", "cls", "cx", "cy", "spread"), (ent, cls, cx, cy, spread)
            ):
                acc[k][i, layer] = v
            if layer == 2:
                acc["amap"][i] = grid[0].reshape(64, G // 2, 2, G // 2, 2).sum(axis=(2, 4))
        patches = memory[0, 1:]
        if basis is None:
            mean = patches.mean(axis=0)
            _, _, vt = np.linalg.svd(patches - mean, full_matrices=False)
            basis = vt[:PCA]
        acc["mem"][i] = ((patches - mean) @ basis.T).reshape(G, G, PCA)
        if i % 100 == 0:
            print(
                f"  {take}: {i}/{n} {(i + 1) / (time.monotonic() - started):.1f} fps",
                flush=True,
            )
    np.savez(
        out / f"{take}.npz",
        seq=np.array([ln["seq"] for ln in lines]),
        ts=np.array([ln["capture_ts"] for ln in lines]),
        stability=np.array(
            [ln.get("stability", np.nan) or np.nan for ln in lines], dtype=float
        ),
        jpeg=np.array([ln["jpeg"] for ln in lines]),
        basis=basis,
        mean=mean,
        **acc,
    )
    print(f"{take}: {n} frames in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    model, out, *takes = sys.argv[1:]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(model, providers=cuda_providers())
    print("providers", sess.get_providers(), flush=True)
    for take in takes:
        if (out_dir / f"{take}.npz").exists():
            print(f"{take}: cached")
            continue
        extract(sess, take, out_dir)


if __name__ == "__main__":
    main()
