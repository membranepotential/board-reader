#!/usr/bin/env bash
# Run one candidate end-to-end and print the report table:
#   train (val = holdout style) -> export ONNX -> evaluate (operator photos, report-only).
# Usage: PY=<venv python> ./run_candidate.sh configs/<cfg>.yaml [extra train.py args, e.g. --styles a b c]
# Per-epoch val streams to <out_dir>/val.log.
set -euo pipefail
CFG="${1:?usage: run_candidate.sh <config.yaml> [extra train.py args]}"
shift
PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

read -r OUT_DIR HOLDOUT HEADROOM SIDE_PAD <<<"$("$PY" -c "
from config import Config; from train import ROOT; import os
c=Config.load('$CFG'); p=c.out_dir
print(p if os.path.isabs(p) else os.path.join(str(ROOT),p), c.holdout, c.headroom, c.side_pad)")"

echo ">>> TRAIN $CFG $*  (tail $OUT_DIR/val.log)"
"$PY" train.py --config "$CFG" "$@"

echo ">>> EXPORT"
"$PY" export.py --ckpt "$OUT_DIR/best.pt"

echo ">>> EVALUATE (operator photos + uploads — REPORT ONLY; geometry $HEADROOM/$SIDE_PAD)"
"$PY" evaluate.py --onnx "$OUT_DIR/model.onnx" --headroom "$HEADROOM" --side-pad "$SIDE_PAD" \
  | tee "$OUT_DIR/eval.txt"

VAL="$("$PY" -c "import json; h=json.load(open('$OUT_DIR/history.json'))
print(round(max(e['val']['accuracy'] for e in h),4))")"
echo
echo "===================== CANDIDATE RESULT ====================="
echo "config: $CFG   holdout=$HOLDOUT   geometry: headroom=$HEADROOM side_pad=$SIDE_PAD"
echo "val (best per-square acc on $HOLDOUT): $VAL"
grep -E 'shape-bias proxy|^OVERALL|^  acc|/64$' "$OUT_DIR/eval.txt" | tail -20
echo "==========================================================="
