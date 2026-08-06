#!/usr/bin/env bash
# Self-driving run (e): cache the 6 wave-2 styles at hr1.6/sp0.15 (the 5 originals
# are already cached there), train resnet18 on all 11 styles (holdout vlab), export
# with geometry sidecar, eval on the 10-board GT, print CANDIDATE RESULT.
set -uo pipefail
PY="${PY:?set PY}"; HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ROOT="$(cd ../.. && pwd)"
GEO="$ROOT/data/external/_cropcache/hr1.6_sp0.15"
W2=(w2_bellanger w2_cemkalyoncu w2_chessr_cana w2_chessr_rgt w2_pjhanzlik w2_polyhaven)
ORIG=(chessred chessrender360 chessr_cana chessr_russiangreytown chessr_vlab_wooden)
OD="$ROOT/scratch/runs/run_e"

echo "===E_CACHE_START==="
"$PY" cropcache.py "${W2[@]}" --headroom 1.6 --side-pad 0.15 --out-root "$GEO"
echo "===E_TRAIN_START==="
"$PY" train.py --config configs/run_e.yaml --styles "${ORIG[@]}" "${W2[@]}"
echo "===E_TRAIN_DONE==="
"$PY" export.py --ckpt "$OD/best.pt"
"$PY" evaluate.py --onnx "$OD/model.onnx" --headroom 1.6 --side-pad 0.15 >"$OD/eval.txt" 2>>"$OD/pipe.log"
VAL="$("$PY" -c "import json;h=json.load(open('$OD/history.json'));print(round(max(e['val']['accuracy'] for e in h),4))")"
echo "===CANDIDATE_RESULT run_e==="
echo "geometry hr1.6/sp0.15   holdout chessr_vlab_wooden   styles: 5 orig + 6 w2 = 11"
echo "val (best per-square acc on vlab): $VAL"
grep -E '/64$|^user_photos|^uploads_calibrate|^OVERALL|shape-bias proxy' "$OD/eval.txt"
echo "===END run_e==="
echo "===E_DONE==="
