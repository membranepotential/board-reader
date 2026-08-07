#!/usr/bin/env bash
# Self-driving run (f): the pose-aware-crop bet.
#
# Identical to run_e in every respect but the crop geometry, so the comparison is
# clean: same 11 styles, same holdout, same optimiser, same seed. Cache at the
# pose geometry, train, export with the geometry sidecar, then score BOTH the
# synthetic holdout (comparable to run_e's 0.898) and the repaired real-photo
# chesscog_transfer set (run_e: 0.819) — the latter is the honest generalisation
# number, since no run has ever trained on it.
set -uo pipefail
PY="${PY:?set PY}"; HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ROOT="$(cd ../.. && pwd)"
GEO="$ROOT/data/external/_cropcache/pose_h1.9_r0.5_f1"
STYLES=(chessred chessrender360 chessr_cana chessr_russiangreytown chessr_vlab_wooden
        w2_bellanger w2_cemkalyoncu w2_chessr_cana w2_chessr_rgt w2_pjhanzlik w2_polyhaven)
OD="$ROOT/scratch/runs/run_f"
mkdir -p "$OD"

echo "===F_CACHE_START==="
# Idempotent: returns immediately for styles already cached at this geometry.
"$PY" cropcache.py "${STYLES[@]}" --pose 1.9 0.5 --out-root "$GEO"
echo "===F_TRAIN_START==="
"$PY" train.py --config configs/run_f.yaml --styles "${STYLES[@]}"
echo "===F_TRAIN_DONE==="
"$PY" export.py --ckpt "$OD/best.pt"
# No --headroom/--side-pad: geometry now travels in the sidecar, and passing it
# by hand is exactly how a tall-crop model got served on short crops once.
"$PY" evaluate.py --onnx "$OD/model.onnx" >"$OD/eval.txt" 2>>"$OD/pipe.log"
VAL="$("$PY" -c "import json;h=json.load(open('$OD/history.json'));print(round(max(e['val']['accuracy'] for e in h),4))")"
echo "===CANDIDATE_RESULT run_f==="
echo "geometry pose h1.9/r0.5   holdout chessr_vlab_wooden   styles: 11 (chesscog_transfer excluded, it is the unseen test)"
echo "val (best per-square acc on vlab): $VAL      [run_e baseline 0.898]"
echo "chesscog_transfer is the real-photo number      [run_e baseline 0.819]"
grep -E '/64$|^user_photos|^uploads_calibrate|^chesscog_transfer|^OVERALL|shape-bias proxy' "$OD/eval.txt"
echo "===END run_f==="
echo "===F_DONE==="
