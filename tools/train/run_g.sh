#!/usr/bin/env bash
# Self-driving run (g) — E7: run_e retrained on occlusion-masked (BR_CROP_MASK=gray)
# crops. Identical to run_e.sh but for the mask: cache the same 11 styles at
# hr1.6/sp0.15 WITH masking (a fresh cache — nothing was ever cached masked
# before E7), train, export (mask recorded in the .onnx.meta.json sidecar), eval.
#
# `set -e` is load-bearing (see run_f.sh): a dead export/evaluate stage must
# abort, never summarise a blank result as CANDIDATE_RESULT.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/_env.sh"   # sets ROOT and PY; refuses an interpreter missing the stack
cd "$HERE"
GEO="$ROOT/data/external/_cropcache/hr1.6_sp0.15_maskgray"
W2=(w2_bellanger w2_cemkalyoncu w2_chessr_cana w2_chessr_rgt w2_pjhanzlik w2_polyhaven)
ORIG=(chessred chessrender360 chessr_cana chessr_russiangreytown chessr_vlab_wooden)
OD="$ROOT/scratch/runs/run_g"

echo "===G_CACHE_START==="
# Idempotent: returns immediately for styles already cached at this geometry+mask.
"$PY" cropcache.py "${ORIG[@]}" "${W2[@]}" --headroom 1.6 --side-pad 0.15 --mask gray --out-root "$GEO"
echo "===G_TRAIN_START==="
"$PY" train.py --config configs/run_g.yaml --styles "${ORIG[@]}" "${W2[@]}"
echo "===G_TRAIN_DONE==="
"$PY" export.py --ckpt "$OD/best.pt"
# No --headroom/--side-pad/--mask: geometry AND mask now travel in the sidecar
# (crop_geometry + occlusion_mask); evaluate.py reads --mask separately since
# masking is a serve-time choice the sidecar records but does not enforce.
"$PY" evaluate.py --onnx "$OD/model.onnx" --mask gray >"$OD/eval.txt" 2>>"$OD/pipe.log"
VAL="$("$PY" -c "import json;h=json.load(open('$OD/history.json'));print(round(max(e['val']['accuracy'] for e in h),4))")"
echo "===CANDIDATE_RESULT run_g==="
echo "geometry hr1.6/sp0.15 mask=gray   holdout chessr_vlab_wooden   styles: 5 orig + 6 w2 = 11"
echo "val (best per-square acc on vlab): $VAL      [run_e baseline 0.898]"
echo "chesscog_transfer is the real-photo number      [run_e baseline 0.819]"
grep -E '/64$|^user_photos|^uploads_calibrate|^chesscog_transfer|^OVERALL|shape-bias proxy' "$OD/eval.txt"
echo "===END run_g==="
echo "===G_DONE==="
