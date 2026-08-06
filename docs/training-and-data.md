---
aliases: [training, datasets, styles, licensing]
tags: [board-reader/training, board-reader/data]
---

# Training and data

The offline half of recognition: how the trained classifier is produced, what data feeds it,
and what may legally be done with the result. The harness lives in `tools/train/` and is not
part of any deployable package — it runs in its own torch venv so the inference runtime stays
torch-free.

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
| `metrics.py` | per-square accuracy split into empty→piece / piece→empty / piece→piece, plus occupancy |
| `train.py` / `evaluate.py` / `export.py` | config-driven loop, held-out eval, ONNX export with round-trip parity |
| `configs/` | one YAML per run — every result below is reproducible from one |

> [!important] Training and serving must cut the same crops
> Crops are cut by the **real deployed cropper**, imported from the package via
> `_bootstrap.py`, so training input cannot drift from what the server feeds. The crop
> *geometry* travels with the exported model in a `<model>.onnx.meta.json` sidecar. A silent
> train/serve geometry mismatch decapitates crops and costs ~15 points of piece-type accuracy
> — it has happened.

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
| **Final test** | Felix's own photos (`data/eval/`) + dim-light calibrate uploads | **report only, never tune** |
| **Auxiliary** | Zenodo piece crops — top-down framing, not the deployed oblique geometry | crop-level signal only |

> [!failure] `chesscog_transfer` is a broken validation set
> Frozen-DINOv2 self-kNN *within* it scores 0.597 accuracy / 0.747 occupancy, where clean
> crops give ~95%+ occupancy — its crops and labels are geometrically broken. It made a good
> run look like a 0.50 failure while the model was actually reading real photos at 0.88.
> Repair its corners before using it as a holdout again.

## Run history

All under `scratch/runs/` (git-ignored; reproducible from `tools/train/configs/`).

| Run | What it tested | Outcome |
|---|---|---|
| `smoke` | 10-minute sanity | — |
| `val_chesscog` | run (a): the multi-style bet, 5 styles | **won** — 0.925 on user photos, deployed as `v1` |
| `loso_resnet_sin` | candidate (b): Stylized-ImageNet init | lost (0.665 holdout) — consistent with the frozen SIN negative |
| `d16` | headroom sweep, `hr=1.6` | **won** — holdout 0.665 → 0.835; **currently deployed** |
| `d18` | headroom sweep, `hr=1.8 / sp=0.25` | ties the holdout, loses on real photos (409 vs 430) |
| `run_e` | wave-2: 11 styles incl. wooden-Staunton dim renders | **finished, never deployed** — see below |

### The open `run_e` question

`run_e` completed 2026-07-19 and was never evaluated as an ensemble. Alone, against the
deployed `d16` on the boards both were scored on:

| Eval slice | `d16` | `run_e` |
|---|---|---|
| 12 dim calibrate boards (768 squares) | 600 | **653** (+53) |
| 10 bright user boards (640 squares) | **605** | 599 (−6) |

A clear win exactly in the regime wave-2 targeted, and a small regression outside it — but
both models are deployed *as ensembles* with the bank, and that comparison has not been run.
Do that before deciding. Tracked in [[STATUS#Next]].

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
