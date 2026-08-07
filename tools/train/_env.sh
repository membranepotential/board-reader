#!/usr/bin/env bash
# Shared interpreter resolution + preflight for the training driver scripts.
#
# The harness needs ONE interpreter carrying torch (with CUDA), onnx, onnxscript,
# onnxruntime, cv2 and yaml. That is the uv-built venv documented in README.md
# ("Setup"), and defaulting to it — rather than demanding the caller export PY —
# is the whole point of this file: run_f's first real attempt was launched with
# PY=/usr/bin/python, which happens to have CUDA torch but none of the ONNX stack.
# Training ran 20 minutes, then export AND evaluate died, and the driver printed a
# green summary over the wreckage. Check the interpreter BEFORE spending the GPU.
#
# Source it (`. "$(dirname "$0")/_env.sh"`); it sets ROOT and PY.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/scratch/torchvenv/bin/python}"

if [ ! -x "$PY" ]; then
    echo "[preflight] no interpreter at $PY" >&2
    echo "[preflight] build it with the uv recipe in tools/train/README.md (Setup)" >&2
    exit 1
fi

"$PY" - <<'PREFLIGHT' || exit 1
import sys

missing = []
for m in ("torch", "torchvision", "onnx", "onnxscript", "onnxruntime", "cv2", "yaml"):
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
if missing:
    sys.exit(
        f"[preflight] {sys.executable} is missing: {', '.join(missing)}\n"
        f"[preflight] uv pip install --python {sys.executable} <them>"
    )

import torch

if not torch.cuda.is_available():
    sys.exit(f"[preflight] no CUDA in {sys.executable} — training would take days")
print(f"[preflight] {sys.executable}")
print(f"[preflight] torch {torch.__version__} · CUDA ok · ONNX stack ok")
PREFLIGHT
