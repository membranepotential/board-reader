#!/usr/bin/env bash
# Calibration fine-tune, with the interpreter preflight in front of the GPU time.
#
# Usage (arguments go straight to calibrate_finetune.py):
#   ./calibrate_finetune.sh --dir ../../data/uploads --out ../../scratch/tuned/mine
#   ./calibrate_finetune.sh --frames a.jpg b.jpg c.jpg --out <dir>
#   ./calibrate_finetune.sh --dir <dir> --out <dir> --dry-run   # just list the sessions
#
# The wrapper exists for the same reason run_f.sh does: run_f's first real attempt
# was launched with an interpreter that had CUDA torch but no ONNX stack, and the
# export died after the training had already been spent.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/_env.sh"
cd "$HERE"
exec "$PY" calibrate_finetune.py "$@"
