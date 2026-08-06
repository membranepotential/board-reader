#!/usr/bin/env bash
# Self-driving headroom sweep: d16 (hr1.6/sp0.15) then d18 (hr1.8/sp0.25).
# Runs the WHOLE pipeline without waiting on any external actor. Emits ===MARKER===
# lines to stdout for a monitor to relay (candidate results + the d18-train-done GPU
# handoff signal). d18 cache builds on CPU in parallel with d16 GPU training; each
# eval runs on CPU in the background so it never blocks the next GPU phase.
set -uo pipefail
PY="${PY:?set PY to the venv python}"
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ROOT="$(cd ../.. && pwd)"
STYLES=(chessred chessrender360 chessr_cana chessr_russiangreytown chessr_vlab_wooden)

candidate() {  # $1=tag $2=out_dir $3=headroom $4=side_pad
  local tag="$1" od="$2" hr="$3" sp="$4"
  "$PY" export.py --ckpt "$od/best.pt" >>"$od/pipe.log" 2>&1
  "$PY" evaluate.py --onnx "$od/model.onnx" --headroom "$hr" --side-pad "$sp" >"$od/eval.txt" 2>>"$od/pipe.log"
  local val; val="$("$PY" -c "import json;h=json.load(open('$od/history.json'));print(round(max(e['val']['accuracy'] for e in h),4))")"
  echo "===CANDIDATE_RESULT $tag==="
  echo "geometry: headroom=$hr side_pad=$sp   holdout=chessr_vlab_wooden"
  echo "val (best per-square acc on vlab): $val"
  grep -E '/64$|^user_photos|^uploads_calibrate|^OVERALL|shape-bias proxy' "$od/eval.txt"
  echo "===END $tag==="
}

# d18 cache (CPU) in parallel with d16 GPU training
D18ROOT="$ROOT/data/external/_cropcache/hr1.8_sp0.25"
( "$PY" cropcache.py "${STYLES[@]}" --headroom 1.8 --side-pad 0.25 --out-root "$D18ROOT" \
    >"$ROOT/scratch/runs/cache_d18.log" 2>&1; echo "===D18_CACHE_DONE===" ) &
D18C=$!

echo "===D16_TRAIN_START==="
"$PY" train.py --config configs/d16.yaml --styles "${STYLES[@]}"
echo "===D16_TRAIN_DONE==="
( candidate d16 "$ROOT/scratch/runs/d16" 1.6 0.15 ) &   # eval on CPU, don't block d18

wait "$D18C"   # d18 cache must be ready before its training
echo "===D18_TRAIN_START==="
"$PY" train.py --config configs/d18.yaml --styles "${STYLES[@]}"
echo "===D18_TRAIN_DONE==="   # <-- GPU is free here; hand off to dataset-bank
candidate d18 "$ROOT/scratch/runs/d18" 1.8 0.25
wait
echo "===SWEEP_DONE==="
