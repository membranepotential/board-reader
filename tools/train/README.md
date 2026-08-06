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
| `metrics.py` | Per-square accuracy split into empty→piece / piece→empty / piece→piece + occupancy accuracy. |
| `train.py` | Config-driven LOSO training loop, AMP, cosine schedule, best-checkpoint. |
| `evaluate.py` | Final held-out eval on the user's own set (`data/eval` + dim-light `data/uploads` calibrate shots). Torch or ONNX. |
| `export.py` | Checkpoint → ONNX (`crop`→`logits`, dynamic batch) + onnxruntime round-trip parity check. |
| `configs/` | `smoke.yaml` (10-min sanity), `loso_resnet.yaml`, `loso_dinov2.yaml`. |

## Setup (scratch venv)

```bash
uv venv "$SCRATCH/torchvenv" --python 3.12
VIRTUAL_ENV="$SCRATCH/torchvenv" uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision timm opencv-python-headless numpy pyyaml onnx onnxruntime pillow
```

(GTX 1650 / driver 610 resolves to a working `torch 2.12+cu13x` wheel.)

## Run

```bash
PY="$SCRATCH/torchvenv/bin/python"
cd tools/train
$PY cropcache.py                      # cache all discovered styles
$PY train.py --config configs/loso_resnet.yaml --holdout <style>   # one LOSO fold
$PY export.py   --ckpt <run>/best.pt  # -> <run>/model.onnx + round-trip check
$PY evaluate.py --onnx  <run>/model.onnx     # final numbers on the user's set
```

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
2. **Validation (unseen style, tune against it)** — `chesscog_transfer` (22 real
   photos of a set in no training set). Use `configs/val_chesscog.yaml` or
   `--holdout chesscog_transfer`. This is the number to tune on.
3. **Final test (report only, never tune)** — Felix's photos (`data/eval`) + the
   dim-light `data/uploads` calibrate shots, via `evaluate.py`.
4. **Auxiliary** — `zenodo_chesspieces` (4 crop-only sets, top-down; geometry ≠
   deployed). Separate crop-LOSO (`configs/zenodo_loso.yaml`) for a cross-style
   feature signal; held out of the board-geometry runs via `require_mode: board`.
