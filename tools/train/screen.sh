#!/usr/bin/env bash
# Screen a candidate config in ~1/3 the time of a scoring run.
#
#   ./screen.sh configs/run_f.yaml [extra train.py args...]
#
# WHAT THIS BUYS, AND WHAT IT DOES NOT
#
# A scoring run is ~40 min: 20 epochs x (90s train + 30s val). Two facts, both
# measured, decide where that time can be cut:
#
#   1. Training data volume is NOT the lever. `steps_per_epoch: 300` x
#      `batch_size: 48` = 14,400 images an epoch out of 663,680 available — 2.2%.
#      Halving the dataset changes wall clock by nothing. (`max_boards` only helps
#      the one-off cropcache build for a NEW geometry.)
#   2. The curve does not converge early. run_e: e6 0.835, e11 0.880, e17 0.898.
#      Truncating it to 8 epochs reads 0.859 and looks like a failure against a
#      0.898 bar it goes on to clear.
#
# So a short run cannot produce a deployable NUMBER — but it can produce a
# RANKING: at matched epoch 6, run_e sat at 0.835 and run_f at 0.846. Screening
# exists to kill clear losers cheaply. Promote a survivor to its full driver
# before believing any absolute figure.
#
# Cuts vs a scoring run: 8 epochs (not 20), val capped to 4000 crops (not 16000,
# which is ~25% of epoch wall time), early stop off — 8 epochs is already short
# enough that stopping early only risks ranking on noise. ~13 min, then export +
# evaluate (~1 min) for the chesscog_transfer number, which is the honest one.
#
# CAVEAT: screening assumes candidate curves do not cross between e8 and e20. We
# have exactly one run-pair supporting that. Treat a screen as a filter for
# losers, not a picker of winners.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/_env.sh"   # sets ROOT and PY; refuses an interpreter missing the stack
cd "$HERE"

CFG="${1:?usage: screen.sh <config.yaml> [extra train.py args...]}"
shift
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 1; }
NAME="$(basename "$CFG" .yaml)"
OD="$ROOT/scratch/runs/screen_$NAME"
mkdir -p "$OD"

EPOCHS="${SCREEN_EPOCHS:-8}"
VAL_MAX="${SCREEN_VAL_MAX:-4000}"

echo "===SCREEN_START $NAME==="
echo "screening $CFG -> $OD (${EPOCHS} epochs, val capped to ${VAL_MAX}, no early stop)"
"$PY" train.py --config "$CFG" \
    --epochs "$EPOCHS" --val-max "$VAL_MAX" --patience 0 --out-dir "$OD" "$@"
"$PY" export.py --ckpt "$OD/best.pt"
"$PY" evaluate.py --onnx "$OD/model.onnx" >"$OD/eval.txt" 2>>"$OD/pipe.log"
VAL="$("$PY" -c "import json;h=json.load(open('$OD/history.json'));print(round(max(e['val']['accuracy'] for e in h),4))")"
echo "===SCREEN_RESULT $NAME==="
echo "best capped-val per-square acc: $VAL   <- RANKING ONLY, not comparable to a scoring run"
grep -E '^user_photos|^uploads_calibrate|^chesscog_transfer|^OVERALL|shape-bias proxy' "$OD/eval.txt"
echo "===SCREEN_DONE $NAME==="
