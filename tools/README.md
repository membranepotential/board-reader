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

## `tools/video/` — YouTube broadcast footage as test data

Downloads a chess broadcast, lets an operator drag a board crop, and extracts cropped
still frames for training/eval. Absorbed from a scratch repo
(`../otb_video_recognition`) that predates the workspace. Four scripts plus a committed
registry:

- `crop_selector.py` — opens a still frame in an OpenCV window; the operator drags a
  rectangle, records it to `crops.csv`. Never re-encodes the video — every other tool
  crops on the fly from that row.
- `check_angle.py` — runs `board_reader_core.boarddetect.detect_board_quad` +
  `quad_aspect` over a video or a frame directory and reports the median projected
  board aspect, the same camera-elevation gate the server warns on.
- `extract_frames.py` — reads the crop from `crops.csv`, writes cropped PNGs at
  explicit or random timestamps.
- `watch_cropped.py` — plays the video with the crop applied live; saves a frame on
  keypress. The interactive alternative to `extract_frames.py`.
- `crops.csv` — the committed crop registry, keyed on the 11-character YouTube video
  id (not the downloaded file's name, which changes with resolution or a re-download).
  Also carries each stream's `check_angle.py` verdict (`median_aspect`, `status`). This
  is what makes any extracted frame set reproducible from nothing but a video id — see
  `tools/video/README.md` for the full CLI and the crop table format, and
  [docs/youtube-streams.md](../docs/youtube-streams.md) for the angle audit and a
  worked recovery of a lost crop row.

**Two interpreters, deliberately.** `crop_selector.py` and `watch_cropped.py` open an
OpenCV/mpv window and need the **system** `python3` — the uv workspace ships
`opencv-python-headless`, and `cv2.imshow` is a no-op there. `check_angle.py` needs
`uv run` instead, because it imports `board_reader_core`. Reaching for the wrong
interpreter fails loudly (`ModuleNotFoundError` either way), not silently.

**The angle gate rejects most broadcast footage.** Running `check_angle.py` over five
absorbed broadcasts and gating on the same threshold the server warns at (`quad_aspect
>= 0.62`, [docs/operating-envelope.md](../docs/operating-envelope.md)) rejected 4 of 5 —
table-level chess broadcasts are shot too shallow for a per-square CV pipeline. See
[docs/youtube-streams.md](../docs/youtube-streams.md) for the measured aspect per
broadcast.

These are third-party broadcast recordings. Downloaded videos and extracted frames
live under `data/streams/` (git-ignored), kept for local/personal research use only —
the same footing as `data/external/chessred`. The end-to-end operator workflow (when to
ask the user for a crop, which frame-extraction mode to use) is documented in the
`get-frames-from-yt` Claude Code skill (`.claude/skills/get-frames-from-yt/`).
