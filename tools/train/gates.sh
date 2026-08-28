#!/usr/bin/env bash
# gates.sh — the LOCAL ship gates for the E7 masked cascade, in one command.
#
#   tools/train/gates.sh                 # scores scratch/runs/run_g + run_h
#   BR_GATE_NAME=e7b tools/train/gates.sh scratch/runs/x.onnx scratch/runs/y.onnx
#
# Runs the continuation runbook (docs/reports/2026-08-28-retrain-program.md
# §Continuation) steps 2–3: the four replay gates and the curated gate, with
# the pinned 08-28 bank so numbers are comparable to the frozen baselines
# (e2-control / e6f-* rows, curated 304/320). Cloud pods produce candidates;
# ONLY these local gates decide shipping. Serial: each replay wants the GPU.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

MODEL="${1:-scratch/runs/run_g/model.onnx}"
MODEL_B="${2:-scratch/runs/run_h/model.onnx}"
NAME="${BR_GATE_NAME:-e7-masked}"
BANK="data/eval/bank-baseline-20260828.npz"

for f in "$MODEL" "$MODEL_B" "$BANK"; do
  [ -f "$f" ] || { echo "[gates] missing $f" >&2; exit 1; }
done

export BR_CROP_MASK=gray BR_MODEL="$MODEL" BR_MODEL_B="$MODEL_B" BR_SESSION_BANK="$BANK"

# 08-13 predates the stability restamp fix; the others are native.
TAKES=(20260828-111421 20260813-193308 20260827-181802 20260827-175539)
for take in "${TAKES[@]}"; do
  extra=()
  [ "$take" = 20260813-193308 ] && extra=(--restamp-stability)
  echo "===GATE_REPLAY $take==="
  uv run board-reader-inference replay "recordings/$take" \
    --policy burst-end --score --name "$NAME" ${extra[@]+"${extra[@]}"}
done

echo "===GATE_CURATED (baseline 304/320)==="
uv run board-reader-inference eval data/eval/ground_truth.json

echo "===GATES_DONE name=$NAME model=$MODEL model_b=$MODEL_B==="
