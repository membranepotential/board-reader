#!/usr/bin/env bash
# remote_run.sh — one training config, start to finish, ON THE POD.
#
#   bash cloud/remote_run.sh configs/run_g.yaml run_g      # cwd: tools/train
#
# Shape mirrors run_g.sh / run_h.sh (===MARKER=== lines a driver can grep for),
# minus the cache stage: cache building is CPU work that stays local (SKILL.md
# "Division of labour"), and the cache arrives on the network volume via
# `rp stage`. Building it here would need data/external/, which never travels.
#
# `set -e` is load-bearing, exactly as in run_f.sh: a dead export or evaluate
# stage must abort, never summarise a blank result.
set -euo pipefail

CFG="${1:?usage: remote_run.sh <config.yaml> <run-name>}"
NAME="${2:?usage: remote_run.sh <config.yaml> <run-name>}"

HERE="$(cd "$(dirname "$0")" && pwd)"      # tools/train/cloud
# shellcheck source-path=SCRIPTDIR source=../_env.sh
. "$HERE/../_env.sh"                        # sets ROOT and PY; refuses a bad interpreter
cd "$HERE/.."                               # tools/train

# Under setsid this process is its own process-group leader, so aborting the
# run means `kill -9 -<PGID>` — never SIGSTOP, which turns a CUDA process into
# a zombie-spinner (GPU at 100%, throughput down ~30x, every health check green).
mkdir -p "$ROOT/scratch"
echo "$$" >"$ROOT/scratch/rp_pgid.$NAME"

# out_dir and mask come from the config, the same way the local drivers hard-code
# them — read them rather than duplicating them.
mapfile -t CFG_VALS < <("$PY" - "$CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(cfg.get("out_dir") or "scratch/runs/run")
print(cfg.get("mask") or "-")
PY
)
OUT_DIR="${CFG_VALS[0]}"
MASK="${CFG_VALS[1]}"
case "$OUT_DIR" in
  /*) OD="$OUT_DIR" ;;
  *)  OD="$ROOT/$OUT_DIR" ;;
esac
mkdir -p "$OD"
echo "$OD" >"$ROOT/scratch/rp_out_dir.$NAME"   # rp reads this to pull artifacts

MASK_ARGS=()
if [ "$MASK" != "-" ]; then MASK_ARGS=(--mask "$MASK"); fi

# The crop cache must already be on the volume: train.py would otherwise try to
# build it from data/external/, which is not staged, and fail 20 minutes in.
CACHE="$("$PY" - "$CFG" <<'PY'
import sys
sys.path.insert(0, ".")
from config import Config
from cropcache import geo_cache_root
print(geo_cache_root(Config.load(sys.argv[1])))
PY
)"
if [ ! -d "$CACHE" ]; then
  echo "[rp] no crop cache at $CACHE" >&2
  echo "[rp] stage it first: tools/train/cloud/rp stage data/external/_cropcache/$(basename "$CACHE")" >&2
  exit 1
fi

echo "===RP_RUN_START $NAME==="
echo "config $CFG   cache $CACHE   out $OD"

echo "===RP_TRAIN_START $NAME==="
"$PY" train.py --config "$CFG"
echo "===RP_TRAIN_DONE $NAME==="

"$PY" export.py --ckpt "$OD/best.pt"
echo "===RP_EXPORT_DONE $NAME==="

# Geometry and mask travel in the .onnx.meta.json sidecar; evaluate.py still
# takes --mask separately, because masking is a serve-time choice the sidecar
# records but does not enforce.
"$PY" evaluate.py --onnx "$OD/model.onnx" ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} \
  >"$OD/eval.txt" 2>>"$OD/pipe.log"
echo "===RP_EVAL_DONE $NAME==="

VAL="$("$PY" -c "import json,sys;h=json.load(open(sys.argv[1]));print(round(max(e['val']['accuracy'] for e in h),4))" "$OD/history.json")"
echo "===CANDIDATE_RESULT $NAME==="
echo "val (best per-square acc on holdout): $VAL"
grep -E '/64$|^user_photos|^uploads_calibrate|^chesscog_transfer|^OVERALL|shape-bias proxy' "$OD/eval.txt" || true
echo "===END $NAME==="
echo "===RP_RUN_DONE $NAME==="
