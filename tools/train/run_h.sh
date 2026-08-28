#!/usr/bin/env bash
# Self-driving run (h) — E7: run_f (pose-aware crop) retrained on occlusion-masked
# (BR_CROP_MASK=gray) crops. Identical to run_f.sh but for the mask.
#
# `set -e` is load-bearing (see run_f.sh): a dead export/evaluate stage must
# abort, never summarise a blank result as CANDIDATE_RESULT.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/_env.sh"   # sets ROOT and PY; refuses an interpreter missing the stack
cd "$HERE"
GEO="$ROOT/data/external/_cropcache/pose_h1.9_r0.5_f1_maskgray"
STYLES=(chessred chessrender360 chessr_cana chessr_russiangreytown chessr_vlab_wooden
        w2_bellanger w2_cemkalyoncu w2_chessr_cana w2_chessr_rgt w2_pjhanzlik w2_polyhaven)
OD="$ROOT/scratch/runs/run_h"

echo "===H_CACHE_START==="
"$PY" cropcache.py "${STYLES[@]}" --pose 1.9 0.5 --mask gray --out-root "$GEO"
echo "===H_TRAIN_START==="
"$PY" train.py --config configs/run_h.yaml --styles "${STYLES[@]}"
echo "===H_TRAIN_DONE==="
"$PY" export.py --ckpt "$OD/best.pt"
"$PY" evaluate.py --onnx "$OD/model.onnx" --mask gray >"$OD/eval.txt" 2>>"$OD/pipe.log"
VAL="$("$PY" -c "import json;h=json.load(open('$OD/history.json'));print(round(max(e['val']['accuracy'] for e in h),4))")"
echo "===CANDIDATE_RESULT run_h==="
echo "geometry pose h1.9/r0.5 mask=gray   holdout chessr_vlab_wooden   styles: 11 (chesscog_transfer excluded, it is the unseen test)"
echo "val (best per-square acc on vlab): $VAL      [run_f baseline 0.894 / run_e 0.898]"
echo "chesscog_transfer is the real-photo number      [run_e baseline 0.819]"
grep -E '/64$|^user_photos|^uploads_calibrate|^chesscog_transfer|^OVERALL|shape-bias proxy' "$OD/eval.txt"
echo "===END run_h==="
echo "===H_DONE==="
