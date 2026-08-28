---
aliases: [training, datasets, styles, licensing]
tags: [board-reader/training, board-reader/data]
---

# Training and data

The offline half of recognition: how the trained classifier is produced, what data feeds it,
and what may legally be done with the result. The harness lives in `tools/train/` and is not
part of any deployable package — it runs on its own torch interpreter (currently system
`/usr/bin/python`, torch 2.13 / Python 3.14; formerly a scratch venv) so the inference
runtime stays torch-free.

## The thesis

Piece-type style-invariance must be **learned from many piece styles**. No single-style
dataset and no frozen-feature trick reaches it — seven were measured and all failed
([[recognition-pipeline#Why piece type is the hard part]]).

Confirmed: a ResNet18 trained on five board styles with heavy shape-bias augmentation reads a
never-seen set at 92.5% per-square (99.7% occupancy), matching a *calibrated* per-set bank
with zero calibration.

## How the harness is wired

| Piece | Role |
|---|---|
| `_bootstrap.py` | puts the real `core`/`inference` src on `sys.path` and stubs the package init so only the pure-geometry `squares` module loads |
| `styles.py` | style registry; auto-discovers every source under `data/external` |
| `cropcache.py` | per-style crop caches, JPEG-recompressed 224² shards + class balance |
| `dataset.py` | torch dataset; **leave-style-out** and leave-groups-out splits; inverse-frequency sampler |
| `augment.py` | shape-bias augmentation — channel shuffle, grayscale, posterize/solarize, noise, AdaIN-style colour-stat randomization, dim gamma |
| `models.py` | two recipes → 13 logits: `resnet18` (ImageNet / scratch / SIN init) and `dinov2` (last-N-block fine-tune) |
| `metrics.py` | per-square accuracy split into empty→piece / piece→empty / piece→piece, plus occupancy and the full 13×13 confusion |
| `train.py` / `evaluate.py` / `export.py` | config-driven loop (fail-fast, instrumented — next section), held-out eval, ONNX export with round-trip parity |
| `calibrate_finetune.py` | the **per-board** LP-FT tool: fine-tunes the deployed model on one board's stored calibrate frames and exports a `tuned.onnx` served 50/50 with the base ([[calibration-and-the-bank#The second adaptation mechanism the calibration fine-tune]]). Refuses input that does not span ≥3 sessions |
| `runlog.py` / `plots.py` | the run observability pair: JSONL event stream + per-epoch PNG rendering |
| `configs/` | one YAML per run — every result below is reproducible from one |

> [!important] Training and serving must cut the same crops
> Crops are cut by the **real deployed cropper**, imported from the package via
> `_bootstrap.py`, so training input cannot drift from what the server feeds. The crop
> *geometry* travels with the exported model in a `<model>.onnx.meta.json` sidecar. A silent
> train/serve geometry mismatch decapitates crops and costs ~15 points of piece-type accuracy
> — it has happened.

## Watching and diagnosing a run

Every run directory carries four synchronized views of the same training (since 2026-08-07):

| View | Audience | What it holds |
|---|---|---|
| `val.log` | humans, `tail -f` | one line per epoch: loss, val summary, timing |
| `events.jsonl` | agents/scripts | `meta` / `step` (loss, lr, img/s, GPU MiB, every `log_every` steps) / `epoch` (full val stats incl. confusion) / `abort` / `end` |
| `history.json` | agents/scripts | per-epoch records, rewritten each epoch — crash-safe |
| `curves.png` · `diagnostics.png` · `confusion.png` | humans | learning curves with best-epoch marker; error-category rates, LR, throughput; row-recall confusion heatmap — refreshed each epoch (`plots.py <run_dir>` re-renders any run, including pre-instrumentation ones) |

A run **fails fast** rather than degrading: a non-finite loss aborts immediately (the
`abort` event says where), a too-full GPU aborts instead of silently training on CPU
(`device: cpu` to force), and `patience: N` early-stops a plateau. The `imgs_per_s` in
step events is the liveness signal — GPU util can read 100% while true throughput has
collapsed (the SIGSTOP zombie-spinner failure), and only throughput catches it.

> [!note] The perf ceiling is the card, not the pipeline (measured 2026-08-07 — **local GTX 1650 only**; the cloud section below re-measures every one of these on Ada-class GPUs)
> fp32 ResNet18@224 on the GTX 1650 is compute-bound at ~165 img/s: batch 48/64/96 and
> 6/8/12 workers are all flat, `channels_last` *loses* (136 img/s, no fp32-NHWC kernels
> on Turing), bf16 is emulated (~2.7× slower) and fp16 NaNs in the forward — `amp: false`
> is the default. VRAM peaks ~250 MiB; memory is never the constraint. The remaining
> wall-time levers are early stopping, the 2× val batch (val was ~40% of epoch time),
> and not sharing the GPU with the inference server during a run.

> [!warning] DataLoader workers must fork
> Python 3.14 made `forkserver` the multiprocessing default, which pickles a **full copy**
> of the ~9 GB RAM-resident crop dataset into every worker — 8 workers OOM-killed the
> first `run_f` launch ("pickle data was truncated" is the tell). `train.py` pins
> `multiprocessing_context="fork"` so workers share the dataset copy-on-write (verified:
> worker Rss 1.0 GB, Pss 160 MB). Workers never touch CUDA, so fork-after-CUDA-init is safe.

## Data contract

One directory per style at `data/external/<source>/<style>/labels_normalized.json`, either:

- **board-level** — `[{image, placement (FEN field 1), corners_a8_h8_h1_a1: [[x,y] × 4], group?}]`
- **crop-level** — `[{image, piece (FEN letter | None), chess_set, ...}]`, one style per set

Board corners are always `a8, h8, h1, a1` — the playing surface, in that order.

## Evaluation tiers — keep them separate

| Tier | What | Rule |
|---|---|---|
| **Train** | ChessReD, ChessRender360, Blender synthetic styles | fit on it |
| **Validation** | an unseen style held out by `--holdout` | tune against it |
| **Final test** | the operator's own photos (`data/eval/`) + dim-light calibrate uploads | **report only, never tune** |
| **Auxiliary** | Zenodo piece crops — top-down framing, not the deployed oblique geometry | crop-level signal only |

> [!success] `chesscog_transfer` — repaired 2026-08-07, usable again
> It was retired as "geometrically broken" (self-kNN 0.597, and it made a good run look like
> a 0.50 failure while the model was reading real photos at 0.88). The diagnosis was wrong in
> an instructive way: **the crops and corners were never broken.** All 22 warps grid-align.
> What was broken was the *label orientation* — `build_transfer_labels.py` decided which side
> was White **by eye**, and got it wrong on 14 of 22 boards, rotating every label 180°. Two
> more boards (`test/22,24`) were silent corner-detection failures — quads of the table, not
> the board.
>
> `repair_orientation.py` now resolves orientation photometrically (checkerboard parity for
> the 90° ambiguity, light-vs-dark piece wood for the 180° one) and drops the non-boards.
> Result: **20 real photos of an unseen physical set**, with `run_e` — which never trained on
> it — scoring **0.819** where it scored 0.500 before. That is the honest real-photo
> generalisation baseline any new run has to beat.
>
> Two durable lessons. *A per-image ambiguity resolved by eye needs an exhaustive check, not
> a 4-board spot check* — the original validation sampled four boards and missed three broken
> ones. And *"the data is broken" is a hypothesis, not a diagnosis*: the 0.597 was real, but
> it pointed at the wrong layer, and the set sat unused for weeks as a result.

> [!success] `uploads_calibrate` — repaired 2026-08-11, and it is **three sets, not one**
> The same label-orientation bug bit here, and it floored the number for months: 19 of the
> non-user boards carried corner labels rotated off true, so `run_e` read them at 17–26/64 and
> the set reported ~**0.59**. `tools/repair_upload_orientation.py` repairs them the same way
> `repair_orientation.py` does for `chesscog_transfer`, but demands that **two independent
> channels agree** — the photometric one and a side-score of the model's own reading under all
> four rotations — before it writes; every known-sound board came back rotation 0, and backups
> land at `*.json.pre-rotation-repair`. The set went 0.59 → **0.75** and hallucinations
> 254 → 30.
>
> Quote the subset, not the mean. The corpus mixes three populations: **17 real photos read at
> 0.919** (comparable to `user_photos`), 13 pre-rectified 384×384 warps at 0.63 — out-of-domain
> crops that are not what the deployed cropper cuts — and 5 genuinely-hard dim frames. A single
> average over the three says nothing about any of them.
>
> One class of board the photometric channel **cannot** judge: this physical board is set up
> a8-**dark**, which inverts the checkerboard-parity assumption. Those 5 were resolved by the
> model channel plus a human look, not automatically.

## Run history

All under `scratch/runs/` (git-ignored; reproducible from `tools/train/configs/`).

| Run | What it tested | Outcome |
|---|---|---|
| `smoke` | 10-minute sanity | — |
| `val_chesscog` | run (a): the multi-style bet, 5 styles | **won** — 0.925 on user photos, deployed as `v1` |
| `loso_resnet_sin` | candidate (b): Stylized-ImageNet init | lost (0.665 holdout) — consistent with the frozen SIN negative |
| `d16` | headroom sweep, `hr=1.6` | **won** — holdout 0.665 → 0.835; deployed 2026-07-18 → 2026-08-06 |
| `d18` | headroom sweep, `hr=1.8 / sp=0.25` | ties the holdout, loses on real photos (409 vs 430) |
| `run_e` | wave-2: 11 styles incl. wooden-Staunton dim renders | **won** — holdout 0.835 → 0.898; **deployed as the primary member** |
| `run_f` | the pose-aware crop bet: a camera-pose-derived prism (`h1.9/r0.5`) instead of the axis-aligned box | **did not win its pre-registered rule** (holdout 0.894 ties, `user_photos` 0.938 → 0.894). Kept anyway: it fails *differently* — cross-style wrongtype 173 → 121, at the cost of hallucinations 1 → 12 — which is what makes it usable as the cascade's second member (2026-08-11) |
| `run_g` / `run_h` | E7: retrain both cascade members on **occlusion-masked** crops so training matches masked serving | member-level a clean win (holdout ties, chesscog_transfer 0.819 → 0.851–0.860 for both geometries) — and **benched anyway**: deployed as the cascade the pair *lost* every gate take, because the masking win is a bank-lookup effect the retrained classifiers overrule ([[2026-08-28-improvement-report]]). Kept as OOD assets awaiting a bank re-embedded from masked crops |
| `run_g_cloud` / `run_h_cloud` | the RunPod parity + first cloud runs | parity PASS (0.8995 vs local 0.8983, ±0.005 gate); same chesscog within noise. Cloud numbers are comparable to *each other*, not to historical local ones (cuDNN kernel selection differs per GPU) |

> [!note] Crop geometry is an **exhausted knob** (closed 2026-08-10)
> `d16`/`d18` swept `headroom`/`side_pad` — two scalars on an axis-aligned bounding box. The
> pose-aware 3D geometry that note used to call untried was then built and measured as `run_f`
> (a `CropGeometry` ADT rides the `.onnx` sidecar, so a train/serve mismatch is
> unrepresentable): it trades cross-style piece **type** for occupancy on the user's own board
> and wins neither pre-registered condition. Both ends of the knob are now measured, so
> geometry is not where the remaining accuracy lives — what survives is *combining* the two
> geometries (the cascade) rather than choosing between them.
>
> The sharper lesson from that run: **+4.8pp of synthetic-holdout gain transferred as 0.0pp on
> real photos** (e3-partial 0.8466 vs final 0.8942 → `user_photos` 0.900 vs 0.894). The
> synthetic holdout does not predict deployment quality on this task; do not select a
> checkpoint on it alone.
>
> Still open: **crop attribution**. Even the prism is the piece's *bounding* volume, so a piece
> standing behind can fall inside the crop unflagged. See STATUS *Next*.

### How `run_e` was decided (2026-08-06)

`run_e` finished 2026-07-19 and sat undeployed for two weeks because it had only ever been
scored *alone*, by `tools/train/evaluate.py`, while both models serve **as an ensemble** with
the kNN bank. Settling it meant scoring the stack that actually serves, on both slices:

| Configuration | 12 dim calibrate (768 sq) | `data/eval` (640 sq) |
|---|---|---|
| `d16` alone | 600 | **605** · 2/10 perfect |
| `run_e` alone | **653** (+53) | 599 · 0/10 perfect |
| `d16` + base bank (unseen by the dim boards) | 591 | — |
| `run_e` + base bank | **630** (+39) | — |
| `d16` ⊕ session bank *(what was deployed)* | — | 633 · 7/10 perfect |
| `run_e` ⊕ session bank | — | **638 · 9/10 perfect** |

The model-alone regression on bright boards is real, but it does not survive the ensemble:
`run_e` ⊕ bank wins `data/eval` outright, including **62/64 vs 59/64 on `up_20260718_0001`** —
the one board no bank has ever seen, and therefore the only honest generalisation signal in
the set. Combined with +53 / +39 squares on the dim boards the wave-2 renders targeted, that
decided it.

> [!note] Two caveats on the numbers above
> The 12 calibrate boards are all *start positions*, so perfect-board rate is meaningless
> there (a constant start-position predictor scores 12/12) — per-square is the metric. And
> the session bank contains 9 of the 10 `data/eval` boards, so that column's absolute values
> are an upper bound for both models; only the comparison and the `_0001` row are honest.
>
> The daylight base bank *lowers* both models on the dim slice (600→591, 653→630). That is
> [[calibration-and-the-bank|the known lighting failure]] reproducing, not a `run_e` effect.

## Training in the cloud (RunPod, since 2026-08-28)

The GTX 1650 caps training at ~170 img/s and serializes every GPU workstream, so training
(and only training — see the division of labour) can run on rented RunPod GPUs. Decision
record: [[2026-08-28-cloud-training-migration|migration analysis]]; plan with measured
checkboxes: `docs/plans/2026-08-28-runpod-migration.md`; operating guide (keys, cost
discipline, traps): `.claude/skills/runpod/SKILL.md`.

**The architecture** — everything durable lives on a network volume (`board-reader`,
EU-RO-1, 30 GB) so pods are disposable:

- `tools/train/cloud/rp` — the runner. `sweep <configs…>` rents one GPU pod for the whole
  queue, provisions it (repo rsync, volume symlinks, on-volume venv via `setup.sh`), runs
  each config through `remote_run.sh` (train → export → evaluate, `===RP_*===` markers),
  pulls artifacts back to `scratch/runs/<name>/`, and deletes the pod. **The EXIT/INT/TERM
  trap that deletes the pod is armed before the create call** — that property has a
  48-assertion stub suite (`cloud/tests/test_rp.sh`) because a forgotten pod is the one
  failure that costs real money. Also: `bench` (the Phase-2 throughput grid), `stage`,
  `adopt` (take over an orphan), `reap`, `cost`.
- **Staging is pod-free** over RunPod's S3-compatible volume API (`tools/runpod/s3-env.sh`,
  key in the login keyring) — resumable `aws s3 sync` straight onto the volume.
- **The pod trains from crop caches alone.** A complete cache (`meta.json`) is
  self-contained: `resolve_styles` treats the raw-dataset registry as authoritative and
  fills gaps from cache scans, so `data/external/` (raw datasets, undocumented provenance)
  never travels. Cache building stays local (pure CPU).
- **Style order is pinned explicitly** in the `*_cloud.yaml` configs: `"all"` resolves in
  manifest order locally but sorted-scan order on a pod, and batch composition differences
  break the ±0.005 parity bar.

**Division of labour** — the cloud is a GPU, nothing more. Cache building, the replay
harness, trial recordings, truth authoring, and **every ship/no-ship decision** stay local;
the cloud produces candidates that must pass the local replay + curated + settled gates.

**Measured:** parity gate PASS (`run_g_cloud` 0.8995 vs local 0.8983); L4 throughput 628
img/s vs 170 local (3.7×), tuned to **721** by the Phase-2 grid — whose real finding is
that **the loader, not the GPU, binds at cloud speeds**: bf16, channels_last, fp16,
compile and batch 128/256 are all throughput-neutral (624–644 img/s), the no-op loader
ceiling peaks at 737 img/s at 8 workers on 6 vCPUs and degrades past it. Rent vCPUs, not
GPU flops (~110 img/s per vCPU). Full grid: `tools/train/README.md` "Cloud perf". Spend
through the program's close $0.66, + $0.29 for the grid.

## Synthetic data

Real multi-style data does not exist in the quantity this needs, so styles are rendered in
Blender: geometry sets × material family re-rolled per render (matte / plastic / wood /
marble / metal / glass) × lighting including dim × camera elevation × board and background.
Breadth beats per-set volume — ~12 styles × 250 boards, 70% random placements and 30% real
FENs.

Wave 2 targets the measured field failure specifically: classic wooden Staunton on a
coordinate-border wooden board, warm 2700–3500 K light with ~45% dim exposures, camera aspect
sampled 0.5–0.9 with a deliberate near-level tail. The generation tooling lives outside this
repo (agent scratchpads); the ingested output lands under `data/external/`.

## Licensing ledger

> [!danger] The deployed model is not redistributable
> It is trained on ChessReD (**CC BY-NC-SA 4.0**, non-commercial) plus ChessR-derived geometry
> whose license is murky (CGTrader-derived). Fine for personal use; **not** for commercial use
> or redistribution. A clean-data retrain is the path if that ever changes.

| Source | License | Status |
|---|---|---|
| ChessReD | CC BY-NC-SA 4.0 | non-commercial; nothing committed |
| ChessRender360 | CC BY 4.0 | clean |
| chesscog transfer set | CC BY | clean (but broken, above) |
| Zenodo piece crops | CC BY | clean, auxiliary only |
| ChessR geometry | murky (CGTrader-derived) | segregated in its own provenance-marked dirs so it can be dropped |
| 8 scouted 3D sets | CC0 / CC BY | catalogued |

Consequence: `data/` and `models/` are git-ignored and no dataset file or derived artifact is
ever committed. The repo describes them; it does not contain them.

Model weights are only ever downloaded from **first-party sources** (Meta's DINOv2) and
converted to ONNX locally — never a third-party pre-converted re-host. Same supply-chain
posture as the 7-day dependency seasoning in `pyproject.toml`.

## Related

- [[recognition-pipeline]] — what the trained model does at serve time
- [[2026-07-18-zero-shot-campaign]] — the full investigation
- [[operating-envelope]] — the regimes wave-2 was aimed at
