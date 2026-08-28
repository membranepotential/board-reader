#!/usr/bin/env bash
# bench_grid.sh — the Phase 2.1 + 2.2 throughput grid, ON THE POD.
#
#   bash cloud/bench_grid.sh [configs/run_g_cloud.yaml]     # cwd: tools/train
#
# Each grid point is the base config with one knob turned, cut to 2 epochs
# (epoch 0 absorbs worker spin-up / cudnn autotune / compile warmup; epoch 1 is
# the measurement) and val capped to 4000 crops (val time is not what is being
# measured; the val acc column only proves the numerics did not collapse).
# train.py ONLY — no export, no evaluate: a grid point is a throughput sample,
# not a candidate. Then bench_loader.py measures the loader ceiling with the
# model removed (the number every train img/s is bounded by).
#
# Output: $ROOT/scratch/bench/report.txt (+ per-point logs/history beside it).
set -euo pipefail

BASE="${1:-configs/run_g_cloud.yaml}"

HERE="$(cd "$(dirname "$0")" && pwd)"      # tools/train/cloud
# shellcheck source-path=SCRIPTDIR source=../_env.sh
. "$HERE/../_env.sh"                        # sets ROOT and PY; refuses a bad interpreter
cd "$HERE/.."                               # tools/train

BDIR="$ROOT/scratch/bench"
mkdir -p "$BDIR/cfgs"
echo "$$" >"$ROOT/scratch/rp_pgid.bench"

# The grid, as (name, yaml-overrides) pairs. bf16/channels_last first (the
# Ada-native levers the local amp:false Turing measurement inverts), then
# compile, then batch, then in-loop workers; the loader pass below isolates
# workers with no GPU in the way.
"$PY" - "$BASE" "$BDIR" <<'PY'
import sys, yaml

base = yaml.safe_load(open(sys.argv[1]))
bdir = sys.argv[2]

GRID = [
    ("fp32_w4",          {}),  # the parity-run configuration — the reference point
    ("bf16",             {"amp": True}),
    ("bf16_cl",          {"amp": True, "channels_last": True}),
    ("fp16_cl",          {"amp": True, "amp_dtype": "float16", "channels_last": True}),
    ("bf16_cl_compile",  {"amp": True, "channels_last": True, "compile": True}),
    ("bf16_cl_b128",     {"amp": True, "channels_last": True, "batch_size": 128}),
    ("bf16_cl_b256",     {"amp": True, "channels_last": True, "batch_size": 256}),
    ("bf16_cl_b128_w8",  {"amp": True, "channels_last": True, "batch_size": 128,
                          "num_workers": 8}),
]
for name, over in GRID:
    cfg = dict(base)
    cfg.update(over)
    cfg.update(epochs=2, patience=0, val_max=4000, steps_per_epoch=300,
               out_dir=f"scratch/bench/{name}")
    with open(f"{bdir}/cfgs/{name}.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(name)
PY

echo "===BENCH_GRID_START==="
for cfg in "$BDIR"/cfgs/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  echo "===BENCH_POINT $name==="
  # A dead point must not kill the grid — a batch-256 OOM is a *result* (the
  # report prints FAILED), not a reason to lose the other seven points.
  if ! "$PY" train.py --config "$cfg" >"$BDIR/$name.log" 2>&1; then
    echo "  FAILED — tail of $name.log:"
    tail -n 5 "$BDIR/$name.log" | sed 's/^/  /'
  else
    tail -n 2 "$BDIR/$name.log" | sed 's/^/  /'
  fi
done

echo "===BENCH_LOADER_START==="
"$PY" bench_loader.py --config "$BASE" --workers 4 8 16 24 \
  | tee "$BDIR/loader.txt"

# The report: epoch-1 (post-warmup) throughput per point, from history.json.
"$PY" - "$BDIR" <<'PY' | tee "$BDIR/report.txt"
import json, pathlib, sys

bdir = pathlib.Path(sys.argv[1])
print(f"{'point':<18} {'img/s':>7} {'e0 img/s':>9} {'peak MiB':>9} {'val acc':>8}")
for cfg in sorted(bdir.glob("cfgs/*.yaml")):
    name = cfg.stem
    hist = bdir / name / "history.json"
    if not hist.exists():
        print(f"{name:<18} {'FAILED':>7}   (see {name}.log)")
        continue
    h = json.load(open(hist))
    e1 = h[-1]
    e0 = h[0]
    print(f"{name:<18} {e1['imgs_per_s']:>7.0f} {e0['imgs_per_s']:>9.0f} "
          f"{e1['gpu_mem_peak_mib']:>9} {e1['val']['accuracy']:>8.4f}")
print()
print(open(bdir / "loader.txt").read().rstrip())
PY
echo "===BENCH_GRID_DONE==="
