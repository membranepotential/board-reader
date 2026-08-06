# tools/

One-off developer tooling that is **not** part of any deployable package.

## `export_dinov2_onnx.py` — DINOv2 ViT-S/14 → ONNX

Produces the recognition backbone (`models/dinov2_small.onnx`, ~90 MB, git-ignored)
from **official Meta weights** (`facebook/dinov2-small`, Apache-2.0). We download the
first-party weights and convert ourselves — we never pull a third-party pre-converted
ONNX (supply-chain trust; see DESIGN §4.6).

The heavy export deps (torch, transformers, onnx) are kept **out of the board-reader
workspace** in an isolated uv project at `../dinov2-export/` so they never touch the
lean inference runtime. The runtime only needs `onnxruntime` (a dependency of
`board-reader-inference`) to load the result.

```bash
# from the isolated tooling project (CPU torch):
cd ../dinov2-export
uv run python ../board-reader/tools/export_dinov2_onnx.py \
  --out ../board-reader/models/dinov2_small.onnx
```

The exported model: input `pixel_values [batch,3,224,224]` (ImageNet-normalised RGB),
output `embedding [batch,384]` (the pooled CLS token), dynamic batch. `DinoV2Embedder`
(in `board_reader_inference.embed`) loads it via ONNX Runtime; point elsewhere with
`BR_DINOV2_ONNX`.

## `build_exemplar_bank.py` — ship bank from the operator's labeled photos

Builds `models/exemplars_dinov2.npz` (the default kNN bank) from the labeled eval
photos in `data/eval/ground_truth.json`. See the module docstring.

## `build_exemplar_bank_chessred.py` — exemplar bank from public ChessReD2K

Builds `models/exemplars_chessred.npz` from the public **ChessReD2K** dataset, to test
whether square recognition can work *out of the box* (no per-user calibration) by
seeding the kNN bank from public data.

**Dataset provenance & license.** ChessReD (Masouris & van Gemert, *"End-to-End Chess
Recognition"*), 4TU.ResearchData, DOI `10.4121/99b5c721-280b-450b-b058-b2900b69a90f`.
License **CC BY-NC-SA 4.0** (non-commercial, share-alike) — fine for local/personal use
and research; **not** for commercial redistribution. We use only the corner-annotated
`chessred2k.zip` subset (20 games, 2078 images, 4.7 GB) + `annotations.json` (22 MB).
Download both into `data/external/chessred/` and unzip (`images/<game>/*.jpg`); the whole
`data/` tree is git-ignored, so **no dataset files or the resulting npz are ever
committed**.

```bash
# after downloading + unzipping chessred2k.zip and annotations.json:
uv run python tools/build_exemplar_bank_chessred.py           # 20 boards/game -> 25.6k exemplars
```

**Annotation geometry (validated).** Each corner record names the four board corners
`top_left / top_right / bottom_right / bottom_left` in a *board-anchored* frame
(top_left is always the a8 corner, regardless of camera angle) — mapping to our semantic
`a8,h8,h1,a1` order as `[top_left, top_right, bottom_right, bottom_left]`. Projecting
every labeled piece through this homography lands **98.9%** in its annotated square.
Piece categories map `white-*→uppercase`, `black-*→lowercase`, `empty→empty`.

**Finding (why this is not the default bank).** Out-of-the-box the ChessReD-only bank
scores **255/320 (79.7%)** on the operator's 5 photos — *below* the user-calibrated bank
(294/320) and even the old centroid baseline (261/320). The gap is almost entirely
**piece-type** confusion (53 of 65 errors): ChessReD uses a different physical piece set,
and nearest-exemplar type discrimination does not transfer across styles. **Occupancy**
(empty vs. occupied), by contrast, transfers excellently — **96.3%** out of the box.
Combining ChessReD with the user's photos *hurt* (272/320 vs. 291 user-only LOO), because
wrong-style ChessReD exemplars win the per-class-max match. So this bank is useful as an
occupancy prior, not as a drop-in piece recognizer; piece-type still needs a few user
calibration frames.
