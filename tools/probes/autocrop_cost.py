"""What does the attention output cost per read, and does the int8 file
localise like the fp32 one?

    flock /tmp/claude-1000/br-gpu.lock uv run python tools/probes/autocrop_cost.py \
        FP32_HEADS.onnx INT8_HEADS.onnx

Times ``session.run`` on the release files (logits + direction) against the
scratch exports with three sliced heads, fp32 on the GPU and int8 on the CPU,
plus the Python grid fit. Then fits the grid with both precisions on the stills
and one gate frame per take and compares the corners.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from autocrop_lib import (
    GATE_TAKES,
    ROOT,
    Model,
    corner_error_any,
    fit_grid,
    stills,
    take_frames,
)
from autocrop_run import FIT, HEADS

from board_reader_inference.recognition.chessq import preprocess_frame
from board_reader_inference.recognition.onnxrt import cuda_providers

RELEASE = ROOT / "models/chessq"


def time_runs(path: Path, outputs: list[str], cpu: bool, n: int, x: np.ndarray) -> float:
    providers = ["CPUExecutionProvider"] if cpu else cuda_providers()
    sess = ort.InferenceSession(str(path), providers=providers)
    for _ in range(3):
        sess.run(outputs, {"image": x})
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(outputs, {"image": x})
        ms.append((time.perf_counter() - t) * 1000)
    return float(np.median(ms))


def main(fp32_heads: Path, int8_heads: Path) -> None:
    items = stills()[::2] + [take_frames(t, per_segment=1)[0] for t in GATE_TAKES]
    x = preprocess_frame(cv2.imread(str(items[0].path)))
    base = ["logits", "direction"]
    print("median session.run ms")
    for label, path, outs, cpu, n in (
        (
            "fp32 GPU release",
            RELEASE / "cq-vit-s-644-v6-seed1-onnx-fp32.onnx",
            base,
            False,
            40,
        ),
        ("fp32 GPU + 3 heads", fp32_heads, base + HEADS, False, 40),
        (
            "int8 CPU release",
            RELEASE / "cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx",
            base,
            True,
            10,
        ),
        ("int8 CPU + 3 heads", int8_heads, base + HEADS, True, 10),
    ):
        print(f"  {label:22s} {time_runs(path, outs, cpu, n, x):7.1f}")

    fp32 = Model(fp32_heads, HEADS)
    int8 = Model(int8_heads, HEADS, cpu=True)
    diffs, err32, err8, fit_ms = [], [], [], []
    for it in items:
        img = cv2.imread(str(it.path))
        h, w = img.shape[:2]
        r32 = fp32(img)
        t = time.perf_counter()
        g32 = fit_grid(r32, FIT, w, h)
        fit_ms.append((time.perf_counter() - t) * 1000)
        g8 = fit_grid(int8(img), FIT, w, h)
        diffs.append(corner_error_any(g8.quad, g32.quad))
        err32.append(corner_error_any(g32.quad, it.quad))
        err8.append(corner_error_any(g8.quad, it.quad))
    print(f"grid fit (Python): median {np.median(fit_ms):.1f} ms")
    print(
        f"n={len(items)}: int8 vs fp32 corners median {np.median(diffs):.2f} % "
        f"(max {np.max(diffs):.2f}); vs outline fp32 {np.median(err32):.2f} %, "
        f"int8 {np.median(err8):.2f} %"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
