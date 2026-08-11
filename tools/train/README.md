# Piece-classifier training harness (`tools/train/`)

Trains a **crop → 13-class** piece classifier (empty + 6 types × 2 colours) that
must generalise to **unseen chess sets** ("styles"). This exists because frozen
DINOv2-S features proved style-entangled (DESIGN §4.6: kNN 255/320, linear probe
254/320 cross-style vs 294/320 same-style) — so we must *learn* features, on
**multiple styles**, validated **leave-style-out**.

Everything runs in a **separate torch venv** (the project workspace stays
torch-free). Crops are cut by the **real deployed cropper** —
`board_reader_inference.squares.crop_squares_photo` — imported from the package
via `_bootstrap.py`, so training input can never drift from what the server
feeds at inference.

The crop **geometry** is per-run config (`headroom`, `side_pad`, `size`) and
travels with the exported model in a `<model>.onnx.meta.json` sidecar that
`TrainedEstimator` reads back. This matters: an early pipeline-wide drift to
`headroom=0.15` (near-base-only crops) *decapitated* tall pieces at steep camera
angles and cost ~15 points of piece-type accuracy. The deployed geometry is
**`headroom=1.6, side_pad=0.15, size=224`** — a whole standing piece. Never
hard-code it on either side; let the sidecar carry it.

## Layout

| File | Role |
|---|---|
| `_bootstrap.py` | Puts the project `core`/`inference` src on `sys.path`; stubs the inference package init so only the pure-geometry `squares` module loads (no FastAPI). |
| `styles.py` | Style registry. Auto-discovers every source under `data/external` and exposes each as a `Style`. Board-level (`corners`+`placement`) and crop-level (`piece`+`chess_set`) manifests both supported; ChessReD has its own adapter. |
| `cropcache.py` | Builds per-style crop caches (`data/external/_cropcache/<style>/`): JPEG-recompressed 224² crops in pickle-free shards + `meta.json` class balance. |
| `dataset.py` | Torch dataset over the cache; **leave-style-out** and **leave-groups-out** splits; inverse-frequency class sampler. |
| `augment.py` | Heavy train augmentation (colour/gamma to very dim, blur, rotation+perspective, erasing); deterministic eval transform. |
| `models.py` | Two recipes → 13 logits: `resnet18` (ImageNet-init or scratch) and `dinov2` (official Meta ViT-S/14 + last-N-block fine-tune, LoRA-free). |
| `metrics.py` | Per-square accuracy split into empty→piece / piece→empty / piece→piece + occupancy accuracy + full 13×13 confusion. |
| `train.py` | Config-driven LOSO training loop, cosine schedule, best-checkpoint, fail-fast (non-finite-loss abort, early-stop patience), throughput-instrumented. |
| `runlog.py` | `events.jsonl` writer/reader — the machine-readable record of a run (meta/step/epoch/abort/end events). |
| `plots.py` | Learning curves + diagnostics + confusion heatmap PNGs, re-rendered each epoch; also a CLI for any (old) run dir. |
| `evaluate.py` | Final held-out eval on the user's own set (`data/eval` + dim-light `data/uploads` calibrate shots). Torch or ONNX. |
| `export.py` | Checkpoint → ONNX (`crop`→`logits`, dynamic batch) + onnxruntime round-trip parity check. |
| `calibrate_finetune.py` | **Calibration fine-tune** (`calibrate_finetune.sh` wraps it with the preflight): LP-FT a deployed model on one operator's own calibrate frames, ~60 s, → `tuned.onnx` + geometry sidecar. Refuses frames from fewer than 3 capture sessions, and the result is only ever *served averaged with its base* (`BR_MODEL_TUNED`) — see docs/TESTING.md. |
| `configs/` | `smoke.yaml` (10-min sanity), `loso_resnet.yaml`, `loso_dinov2.yaml`, `run_f.yaml`. |
| `_env.sh` | Resolves `PY` to the scratch torch venv and **preflights** it (torch+CUDA, onnx, onnxscript, onnxruntime, cv2, yaml) before any GPU time. Sourced by the driver scripts. |
| `run_f.sh` | Self-driving scoring run: cache → train → export → evaluate → summary. `set -e`, so a dead stage aborts instead of printing a blank result. |
| `screen.sh` | Same pipeline, ~1/3 the wall clock — for **ranking** candidates, not scoring them. See below. |
| `tests/` | Unit tests for the split/sampling helpers. Run under the torch venv (see Tests). |

## Setup (scratch venv)

The venv lives at **`scratch/torchvenv`** (gitignored), and the driver scripts
default `PY` to it — you should not need to set `PY` at all.

```bash
uv venv scratch/torchvenv --python 3.12
uv pip install --python scratch/torchvenv/bin/python \
  --exclude-newer 2026-06-09T00:00:00Z \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision timm opencv-python-headless numpy pyyaml pillow \
  onnx onnxscript onnxruntime matplotlib pytest
```

(GTX 1650 / driver 610 resolves to a working `torch 2.12+cu13x` wheel.
`--exclude-newer` mirrors the workspace's 7-day dependency seasoning — see the
root `pyproject.toml`; bump both together.)

**`onnxscript` is not optional.** From torch 2.9 on, `torch.onnx.export` routes
through the dynamo exporter, which imports it — without it `export.py` dies
*after* training has already spent the GPU. `matplotlib` backs `plots.py`, and
`pytest` runs the harness tests below. All three were missing once, and a run was
launched against `/usr/bin/python` (CUDA torch, no ONNX stack) instead of this
venv: training completed, then export and evaluate both crashed. `_env.sh` now
preflights the interpreter for exactly this and aborts before any GPU time.

## Tests

The harness cannot be tested by the workspace suite — that venv is deliberately
torch-free, and `dataset.py` imports torch. Run them with the torch venv:

```bash
scratch/torchvenv/bin/python -m pytest tools/train/tests -q
```

## Run

```bash
cd tools/train
./run_f.sh                            # a full self-driving candidate run
./screen.sh configs/run_f.yaml        # ~1/3 the time — RANKS candidates, see below

PY="$ROOT/scratch/torchvenv/bin/python"   # only if driving the stages by hand
$PY cropcache.py                      # cache all discovered styles
$PY train.py --config configs/loso_resnet.yaml --holdout <style>   # one LOSO fold
$PY export.py   --ckpt <run>/best.pt  # -> <run>/model.onnx + round-trip check
$PY evaluate.py --onnx  <run>/model.onnx     # final numbers on the user's set
```

## Iterating faster (`screen.sh`)

A scoring run is ~40 min. Two measured facts decide where that can be cut, and
one popular idea is ruled out:

- **Less training data saves nothing.** `steps_per_epoch: 300` × `batch_size: 48`
  = 14,400 images an epoch out of 663,680 available — **2.2%**, or 0.43 passes
  over the corpus across a whole 20-epoch run. Halving the dataset changes wall
  clock by zero. (`max_boards` only helps the one-off `cropcache` build for a
  *new* geometry, which is where data volume genuinely costs.)
- **Validation does cost.** The full leave-style-out holdout is 16,000 crops =
  ~30s of each 120s epoch. `val_max` caps it with a seeded, class-stratified
  draw — stratified because piece-type accuracy turns on the rare classes.
- **The curve does not converge early.** run_e: e6 0.835, e11 0.880, **e17
  0.898**. Truncating to 8 epochs reads 0.859 — a "failure" against a bar it
  goes on to clear.

So short runs cannot produce a deployable **number**, but they can produce a
**ranking**: at matched epoch 6, run_e was 0.835 and run_f 0.846.

```bash
./screen.sh configs/run_f.yaml            # 8 epochs, val capped to 4000, no early stop
SCREEN_EPOCHS=12 ./screen.sh configs/x.yaml   # or tune it
```

~13 min of training plus ~1 min of export+evaluate (the `chesscog_transfer`
number is the honest one and is cheap, so screening includes it). Promote a
survivor to a full driver before believing any absolute figure.

**Caveat:** this assumes candidate curves do not cross between e8 and e20. We
have one run-pair supporting that. A screen filters out losers; it does not pick
winners.

**Do not shrink `patience` to save time.** It buys back almost nothing on a
40-minute run and truncates real ones: run_f's first attempt died at e8 on a
4-epoch stall, having peaked at 0.847, and answered nothing. Scoring configs use
`patience: 8`; `screen.sh` disables it outright.

## Watching a run

Every run directory carries four synchronized views of the same training:

| File | Audience | Contents |
|---|---|---|
| `val.log` | humans, `tail -f` | one line per epoch: loss, val summary, timing |
| `events.jsonl` | agents/scripts | JSONL: `meta` (config, split sizes), `step` every `log_every` steps (loss, lr, img/s, GPU mem), `epoch` (full val stats incl. confusion), `abort`/`end` |
| `history.json` | agents/scripts | per-epoch records, rewritten each epoch (crash-safe) |
| `curves.png` · `diagnostics.png` · `confusion.png` | humans | learning curves, error-category rates / lr / throughput, val confusion heatmap — refreshed each epoch by `plots.py` |

A run **aborts immediately** on a non-finite loss (`abort` event carries where),
and `patience: N` early-stops after N epochs without val gain. Throughput
(`imgs_per_s` in step events) is the liveness signal: a healthy fp32 run holds
~160+ img/s — if it collapses while GPU util looks fine, the run is wedged (see
learnings about SIGSTOP zombie-spinners).

Perf notes (GTX 1650, measured): plain fp32 beats bf16 (emulated, ~2.7× slower)
and fp16 (NaN in forward + slow) — `amp: false` is the default. cuDNN benchmark
autotune is always on; DataLoader workers each pin `cv2`/torch to one thread so
worker parallelism is real. `channels_last`/`compile` are config flags, off by
default (no tensor cores / no triton on this box — re-benchmark elsewhere).

## Deployment integration (shipped)

The exported ONNX is `crop (N,3,224,224) → logits (N,13)`; `softmax(logits)` is the
per-square distribution, in protocol `CLASSES` order (index `i` → `CLASSES[i]`, no
transpose). It fuses the old embed→classify split, and is served by
**`board_reader_inference.trained.TrainedEstimator`**, selected via `BR_MODEL`:

1. read `<model>.onnx.meta.json` for the crop geometry;
2. `crop_squares_photo(image, corners, **geometry)` — 64 crops;
3. one batched ONNX run → `(64, 13)` logits → softmax;
4. reshape to `(8, 8, 13)` and hand to `PositionEvent.from_square_dist`.

Preprocessing lives **outside** the graph: BGR→RGB, `/255`, ImageNet mean/std,
NCHW. `evaluate.py`'s `OnnxPredictor._prep` is the reference implementation, and
`models/deploy_fixture/` pins a trainer-side fixture so the served estimator can
be verified bit-for-bit against training (max logit deviation 4.5e-5 at deploy).

`TrainedEstimator` sits beside `CentroidEstimator`/`ExemplarClassifier` rather
than replacing them — `EnsembleEstimator` blends the two, and `/calibrate` adds
a per-set booster bank on top. No protocol change: everything emits the same
canonical `square_dist`.

## Data contract

`data/external/<source>/**/labels_normalized.json`, auto-discovered. Either:
- **board-level**: `[{image, placement (FEN field 1), corners_a8_h8_h1_a1:[[x,y]×4], group?}]`
- **crop-level**: `[{image, piece (FEN letter/None), chess_set, ...}]` → one style per `chess_set`, empties shared.

## Evaluation tiers (keep them separate)

1. **Train** — `chessred` + ChessRender360 + synthetic styles (board geometry).
2. **Validation (unseen style, tune against it)** — `chesscog_transfer` (**20**
   real photos of a set in no training set; label orientation repaired 2026-08-07
   — see its README). Never put it in `styles`; score it after every run.
   `run_e` baseline: **0.819** per-square. Held-out synthetic
   `chessr_vlab_wooden` stays the `--holdout` for comparability with `d16`/`run_e`.
3. **Final test (report only, never tune)** — Felix's photos (`data/eval`) + the
   dim-light `data/uploads` calibrate shots, via `evaluate.py`.
4. **Auxiliary** — `zenodo_chesspieces` (4 crop-only sets, top-down; geometry ≠
   deployed). Separate crop-LOSO (`configs/zenodo_loso.yaml`) for a cross-style
   feature signal; held out of the board-geometry runs via `require_mode: board`.
