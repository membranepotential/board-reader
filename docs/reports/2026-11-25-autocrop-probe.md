# Auto-frame probe: can ChessQ find the board in a whole phone frame? (2026-11-25)

**Question.** `/control` sets the phone's crop by hand (DESIGN D11). Could the node
propose that crop itself, from ChessQ's own square attention, from the old classical
detector, or from both? How small may the board be in the whole frame, how does a bad
proposal show itself, and what does it cost?

**Answer: yes, with ChessQ and two reads (coarse to fine).** Read the whole frame
once, fit an 8×8 grid to the square-attention centroids, crop the frame to that rough
box plus 50 % on every side, read again and fit again. The final grid's box plus a
10 % margin is the proposed crop.

- **Reliable down to a board 25 % of the frame width.** At 35 % and 25 %, 30 of 30
  and 29 of 30 proposals are good, none is bad, and ChessQ reads the proposed crop as
  well as the hand-drawn one: 97.8 vs 98.1 % per square at 35 %, 98.0 vs 98.1 % at
  25 %, and the same perfect boards (67 % on this hard subset).
- **At 15 % add the tile fallback.** Coarse to fine finds 19 of 30 there. When its
  grid fails the check, five tiles plus one refine read find 28 of 30, with no bad
  proposal and an end-to-end read equal to the hand crop (98.4 vs 98.4 %).
- **Failure signal: the grid fit's R² ≥ 0.90.** On the final grid (coarse to fine, or
  tiles) every bad proposal measured R² ≤ 0.89 and almost every good one ≥ 0.90.
  Below the threshold the UI says "no board found" instead of proposing.
- **The classical detector loses.** It is exact when it fires (0.1 % corner error
  against outlines it made itself), but it fires on only 53–67 % of canvases at
  35–25 %, on 0 % at 15 %, and on 25–33 % of the dark live phone frames. It also costs
  0.4 s to 25 s of CPU per frame. As the proposer in a classical + ChessQ-verify
  combination it adds nothing that ChessQ's own localisation does not already give.
- **Cost:** the attention output adds 4 ms to a 196 ms fp32 GPU read (+2 %), and the
  grid fit takes 10 ms in Python. The int8 CPU file exposes the same attention (found
  by shape: its tensors are renamed). Its corners lie within 0.3 % of the fp32 corners
  (median), at 1,043 ms against 915 ms per read.

## Data

- **Images.** 144 images, read as delivered (`as-is`):
  - 21 phone stills (curated + heldout-0831), with hand-checked corners;
  - 88 frames of the seven gate takes, one per truth segment;
  - 35 frames of the four live phone takes of 2026-11-24/25.

  The zoomed-out canvases use a timeboxed subset of 30 images: every third still,
  every eighth gate frame, and three frames per live take.
- **Board outlines.** The gate takes' own `frames.ndjson` corners are too loose to
  serve as truth: an overlay misses whole board lines by up to ~5 % on 172736 and
  181802. The live takes have no corners at all. `autocrop_label.py` therefore runs
  the classical detector on every 5th frame of each take. A frame's outline is the
  corner-wise median of the agreeing detections within ±20 rows. Overlays were checked
  by eye on every take.
  - **Caveat.** The outline comes from the classical detector, so its corner error
    against itself is ~0.1 % by construction. ChessQ's 1.1–1.3 % is an upper bound
    that includes this bias.
  - **Caveat.** The dark live take 221950 got outlines on only 297 of 613 rows,
    because the detector found its board in 39 of 123 sampled frames. So the live
    images are biased toward frames the classical detector can read.
- **Canvases.** `autocrop_lib.place` puts the image at scale 1 on a 16:9 canvas at a
  random position, so that the board's bounding box is 35, 25 or 15 % of the canvas
  width. The image edges are feathered over 24 px.
  - **Room backdrop.** Outside the image, the canvas shows a patchwork of board-free
    strips cut from the same recordings and stills: sofa, rug, clothes, table, wooden
    garden table, pavement, hands, a chess clock. It holds real clutter, not mirrored
    boards.
  - **No uncropped phone recording exists.** The live frames are already the phone's
    crop (1080p ultra-wide, crop width 0.78). In the full frame their board would be
    about 47 % of the width.
  - The hand-drawn crop is the delivered image itself.
- **Head choice** (`autocrop_heads.py`, 109 images). Decoder cross-attention heads
  L1H0, L2H0 and L3H0 each localise the squares: median R² 0.997, corner error 1.1–1.3 %
  as-is and 1.6–1.8 % at 35 % (grey padding). Every other head is diffuse (L3H1 2.4 %,
  the rest ≥ 11 %). The probe fits the mean of the L1H0, L2H0 and L3H0 centroids. A
  centroid is taken in the 5×5-patch window around each query's attention peak, and
  the homography from square centres to image points is fitted with LMedS.

## Results

`good` = mean corner error ≤ 5 % of the board width; with a 10 % margin such a crop
still holds the whole board. Strategies propose only a grid with R² ≥ 0.90. Reads are
localisation reads; the crop's first live read comes on top. End-to-end values are
per-square % / perfect-board %, on the proposal's 10 % crop at canvas resolution, with
the hand-drawn crop in brackets (truth-labelled images only).

| board share | strategy | proposes | good of proposed | bad | corner err (median) | reads | end-to-end (hand) |
|---|---|---|---|---|---|---|---|
| as-is (~80 %), n=144 | single pass | 100 % | 99 % | 1 | 1.2 % | 1 | 99.10 / 76 (99.18 / 77) |
| | coarse to fine | 100 % | 99 % | 1 | 1.2 % | 2 | 99.05 / 75 (99.18 / 77) |
| | classical | 89 % | 100 % | 0 | 0.1 % * | 0 | 99.20 / 78 (99.37 / 79) |
| 35 %, n=30 | single pass | 87 % | 100 % | 0 | 1.7 % | 1 | — |
| | **coarse to fine** | **100 %** | **100 %** | 0 | 1.1 % | 2 | **97.84 / 67 (98.07 / 67)** |
| | classical (wide ladder) | 67 % | 100 % | 0 | 0.1 % * | 0 | 98.93 / 75 (99.02 / 75) |
| 25 %, n=30 | single pass | 63 % | 68 % | 6 | 2.7 % | 1 | — |
| | **coarse to fine** | **97 %** | **100 %** | 0 | 1.1 % | 2 | **97.99 / 67 (98.07 / 67)** |
| | classical (wide ladder) | 53 % | 100 % | 0 | 0.1 % * | 0 | 99.52 / 85 (99.52 / 85) |
| 15 %, n=30 | single pass | 7 % | 0 % | 2 | 11.1 % | 1 | — |
| | coarse to fine | 60 % | 100 % | 0 | 1.3 % | 2 | 98.33 / 67 (98.33 / 67) |
| | **coarse to fine, tiles when R² < 0.90** | **93 %** | **100 %** | 0 | 1.3 % | 4.4 (mean) | **98.36 / 70 (98.44 / 70)** |
| | classical (wide ladder) | 0 % | — | 0 | — | 0 | — |

\* against outlines the classical detector made itself (see Data). The classical
end-to-end values cover only the images it found, which are the easier ones; that is
why its "hand" column differs.

- **The single pass is not enough below 50 %.** At 25 % it proposes 6 bad crops
  that pass the R² check. A single pass at 25 % or below must not be trusted; the
  second read fixes this.
- **Coarse to fine fixes the small board.** Pass 1 needs only a rough box. Grown by
  50 %, the crop puts the board at about half the read's width, where the attention
  is sharp again. Corner error drops to 1.1 %.
- **Tiles rescue 15 %.** Pass 1 sees a 15 % board as about 6 patches a side, and its
  rough box is often wrong. Each 60 % tile shows the board at ~25 % of the tile.
- **Per group at 25 %** (coarse to fine, tiles as the fallback): stills 7/7, gate 11/11,
  live 11/12. The live misses are refusals, not bad crops. The classical detector found
  the board in 7/7 stills, 6/11 gate frames and 3/12 live frames.

## Failure detection

| final grid | good: R² ≥ 0.90 | bad: R² ≥ 0.90 | n good / bad |
|---|---|---|---|
| coarse to fine, 35 + 25 % | 100 % | 0 % | 59 / 1 |
| coarse to fine, 15 % | 95 % | 0 % | 19 / 10 |
| tiles + refine, 15 % | 100 % | 0 % | 28 / 2 |
| single pass, 35 + 25 % | 95 % | 32 % | 41 / 19 |

- **Gate the final grid, never pass 1.** Pass 1 grids at 25 % can be wrong with
  R² up to 0.95.
- **The other signals separate as well.** AUC for good vs bad final grids:
  - inliers within 0.35 squares: 0.999;
  - R²: 0.999;
  - frame energy: 0.997.

  R² is the simplest to explain and needs no tuning per take.
- **Not tested:** a frame with no board at all, and a frame with two boards. Every
  canvas here has exactly one board.

## The classical detector

`tools/video/boarddetect.py` is imported unchanged; it is the code the edge ran until
the dumb-camera change. It finds the board with `findChessboardCornersSB` x-corner
lattices, a phase search and an alternation check. The `wide` ladder adds 2400 and
3200 px rungs, so that a small board is not shrunk below the corner finder's reach.
It helped only at 35 % (+10 points).

- **Failure modes seen.**
  - **The dark wooden board** of the live phone takes: low-contrast squares. It found
    it in 25–33 % of live frames, against 100 % of the lit stills.
  - **Pieces covering the lattice** on crowded gate positions.
  - **Small boards:** 0 % at 15 %.
  - **Hands:** they knock out the sub-grid the finder grows from.
  - **When it fires, it is never wrong:** 0 bad quads in every condition.
- **Time** (Python, one core, but 10 processes shared 12 cores, so these values are
  high): median 0.45 s per as-is frame (960 px), 7 s at p90 on a failure. On the
  multi-megapixel canvases it took 8–25 s.
- **In the browser.** The stock OpenCV.js build exposes neither
  `findChessboardCornersSB` nor `cornerSubPix` (its `opencv_js.config.py` whitelist),
  so the phone would need a custom wasm build of several MB. The alternative is a
  hand-written TypeScript port, which needs these operations:
  - greyscale and a resize pyramid;
  - an x-corner (saddle) filter with non-maximum suppression;
  - lattice growth over neighbour points;
  - a DLT homography with RANSAC;
  - a perspective warp to sample square means.

  That is several hundred lines, and it would still miss the dark board.
- **The combination** (classical proposes, one ChessQ read of its crop verifies,
  ChessQ localisation only as the fallback) adds nothing measurable. Classical fails
  most often exactly where a fallback is needed (small boards, the dark board).
  ChessQ's own two reads are already good wherever classical is.

## Cost

| | median `session.run` |
|---|---|
| fp32 GPU, release outputs | 195.7 ms |
| fp32 GPU, + 3 attention heads (sliced in-graph, 3 × 0.5 MB) | 199.7 ms |
| int8 CPU, release outputs | 915 ms |
| int8 CPU, + 3 attention heads | 1,043 ms |
| grid fit (Python, 3 heads × 64 windows + LMedS) | 9.6 ms |

An auto-frame costs 2 reads (≈ 0.4 s on the GPU), or ~4.4 on average at 15 % with the
tiles. It runs once, on request, not per frame.

## Build sketch

1. **Model (node).** In `make models`, add a graph step that writes a sibling file per
   release file (fp32 export and int8 release) with the L1H0, L2H0 and L3H0
   cross-attention heads as extra outputs: `tools/probes/autocrop_model.py SRC OUT 1:0 2:0 3:0`,
   hardened into `tools/chessq_export/`. It finds the softmaxes by shape, so it works
   on the int8 file's renamed graph. Keep the release outputs unchanged. The serving
   read can ignore the extra outputs (+4 ms) or keep using the plain file.
2. **Locator (node, `recognition/`).** A pure function `locate(image) -> Grid | None`
   holds the centroid windows, the LMedS homography and R². A driver runs
   `autoframe(full_frame)`:
   - pass 1 on the whole frame;
   - pass 2 on the rough box grown by 50 %;
   - if R² < 0.90, the five tiles and a refine;
   - the result is the box plus 10 % as a `CropRect` in frame fractions, or
     "no board found".
3. **Trigger (control socket, no new HTTP endpoint**, per `.claude/rules/camera-control.md`):
   - `/control` sends `{"type": "autoframe"}` on `WS /boards/{id}/control`.
   - The node needs one whole frame. The phone already sends a whole-frame preview
     every 1.5 s while a control page is connected, but only at 480 px (JPEG q 0.6).
     That would put a 47 % board at ~225 px. Localisation on the preview was planned
     and not measured (see Gaps). The safe path is a `PhoneDown` message
     `{"type": "send_full_frame"}`, answered by one full-resolution JPEG on the same
     socket.
   - The node then pushes `{"type": "autoframe_result", "crop": ..., "r2": ...}` or
     `{"type": "autoframe_result", "error": "no_board"}`.
4. **UI (`/control`).** An "Auto-frame" button draws the proposed rectangle on the
   preview. The user accepts it, which sends the usual `configure`, or adjusts it.
   Nothing applies without the click.
5. **Phone.** Unchanged except for answering `send_full_frame`.

## Gaps (timeboxed, not measured)

- **Localisation on the phone's 480 px preview** (conditions `prev480-*` exist in
  `autocrop_run.py`, not run). It decides whether the phone needs a new message.
- **Margin sweep 5 / 15 %:** measured only as-is (per-square 98.9 / 99.1 / 99.1 % at
  5 / 10 / 15 % vs 99.2 hand). Take 10 % or more.
- **A frame with no board or with two boards**, and a real uncropped room frame:
  no recording exists. Record one full-frame still per setup.
- **1080p downscale** (the board at its real pixel count in a 1080p frame).
- **Continuous tracking** (re-crop after a bump) was dropped from scope. The node
  already reports `camera_moved` from the board direction (D11).
- **The classical times** were taken under 10-process load. They are an upper bound.

## Reproduce

```bash
S=<scratch>
uv run --project tools/chessq_export --frozen python tools/probes/autocrop_model.py \
  models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx $S/fp32-full.onnx      # all heads
uv run --project tools/chessq_export --frozen python tools/probes/autocrop_model.py \
  models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx $S/fp32-heads.onnx 1:0 2:0 3:0
uv run --project tools/chessq_export --frozen python tools/probes/autocrop_model.py \
  models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx $S/int8-heads.onnx 1:0 2:0 3:0
CUDA_VISIBLE_DEVICES= uv run python tools/probes/autocrop_label.py <takes>   # outlines
cd tools/probes
flock /tmp/claude-1000/br-gpu.lock uv run python autocrop_heads.py $S/fp32-full.onnx
CUDA_VISIBLE_DEVICES= uv run python autocrop_classical.py $S/out
flock /tmp/claude-1000/br-gpu.lock uv run python autocrop_run.py $S/fp32-heads.onnx $S/out chessq
flock /tmp/claude-1000/br-gpu.lock uv run python autocrop_run.py $S/fp32-heads.onnx $S/out classical
flock /tmp/claude-1000/br-gpu.lock uv run python autocrop_cost.py $S/fp32-heads.onnx $S/int8-heads.onnx
CUDA_VISIBLE_DEVICES= uv run python autocrop_analyze.py $S/out
```

The scratch directory is `autocrop_lib.SCRATCH`. Outlines and run outputs live there,
not in the repo.

## Follow-up: the phone's preview (2026-11-25)

**Question.** The phone keeps cropping. Can the node find the board, and notice a
camera move ("drift"), on the whole-frame previews the phone already sends: 480 px
long side, JPEG quality 0.6, every 1.5 s (`PREVIEW_*` in
`packages/viewer/src/lib/components/camera/state.ts`)?

**Answer: yes, for the real setup (board ≈ 47 % of the frame), with one read per
preview.**

- **Locate:** at a 50 % share the single pass finds 96 % of the boards with 1.7 %
  median corner error (1 bad of 51 found). Coarse to fine on the 480 px preview
  only upsamples, but it still helps: 1.2 % error and no bad grid at 50 %, and at
  35 % it raises the found rate from 66 % to 91 %. Below 35 % the preview is too
  small: at 25 % even coarse to fine finds 68 % with 3 bad grids. Auto-frame for a
  small board needs the full-resolution frame (the `send_full_frame` path above).
- **Drift rule (proposed):** compare each preview's board with the reference board
  (the board located when the crop was set). Count a preview only when its grid
  has **R² ≥ 0.97** (stricter than Auto-frame's 0.90). Report `camera_moved` when
  the **max corner displacement exceeds 5 % of the board width on 2 consecutive
  counted previews** (3 s). A preview below R² 0.97 (a hand over the board, a
  person in view) neither counts nor resets the streak.
  - On 421 still previews (11 takes, 11 min): **0 false fires**.
  - It catches moves whose true corner displacement is ≥ 6 %: rotation 5° 20/22,
    zoom ±10 % 38/44, shift 10 % 20/22, all within 3 previews (4.5 s). It misses most
    moves of 3–5 % (shift 5 %: 17/22 caught), which still sit inside the crop's
    10 % margin.
- **Cost:** 195 ms per fp32 GPU read with the attention output (the read is 644 px
  whatever the preview's size), 1,067 ms per int8 CPU read, plus 8–16 ms for the
  grid fit. At one preview per 1.5 s that is 40 reads per minute per camera:
  8 s of GPU time per minute (13 % of one GPU), or 43 s of one CPU core per minute
  on the int8 file.

### Locate on the preview

`autocrop_preview.py locate`: the 30-image subset of the canvases above plus every
labelled live-take image (53 images: 7 stills, 11 gate frames, 35 live frames), on
the room backdrop, then downscaled to 480 px and JPEG'd at quality 60. "Found" is
R² ≥ 0.90; "good" is corner error ≤ 5 % against the outlines. Coarse to fine cuts
pass 1's box grown by 50 % out of the 480 px preview and reads it again.

| board share | board px in preview | strategy | found | good of found | bad | corner err (median) | R² p10 / median |
|---|---|---|---|---|---|---|---|
| as-is (~80 %) | 328 | single pass | 96 % | 100 % | 0 | 1.3 % | 0.990 / 0.997 |
| | | coarse to fine | 96 % | 100 % | 0 | 1.2 % | 0.987 / 0.997 |
| 50 % | 240 | **single pass** | **96 %** | 98 % | 1 | 1.7 % | 0.971 / 0.995 |
| | | coarse to fine | 96 % | 100 % | 0 | 1.2 % | 0.980 / 0.997 |
| 35 % | 168 | single pass | 66 % | 97 % | 1 | 2.1 % | 0.784 / 0.959 |
| | | coarse to fine | 91 % | 96 % | 2 | 1.4 % | 0.915 / 0.997 |
| 25 % | 120 | single pass | 23 % | 58 % | 5 | 3.8 % | 0.483 / 0.763 |
| | | coarse to fine | 68 % | 92 % | 3 | 1.7 % | 0.622 / 0.980 |

- **The misses at 50 % and above** are the same two images in every row: a still
  (img_0568) and a dark live frame with a hand (004038 #285). The bad 50 % grid is a
  gate frame at 5.6 % corner error, just over the line.
- **What coarse to fine buys on the preview:** the second read sees the board at
  about half its width, so ChessQ's 46 patches cover it more finely, even though
  the pixels are upsampled. It doubles the cost. For drift checks on a 47 % board
  the single pass is enough; `locate_preview` (single pass) fits that.

### Drift

`autocrop_preview.py drift`: every take (7 gate, 4 live) as frames 1.5 s apart (at
most 40 per take), each placed on one fixed 50 % room canvas per take, so the camera
does not move while hands and pieces do. The **reference** is the median located
quad of a take's first three counted previews. The **truth** is the probe's
outline of each frame (the classical detector's ±20-row median). Previews whose
outline moved ≥ 1.5 % from the reference are left out of the still set: the 2026-08
tripods creep by ~2 % over a take, and several takes hold real bumps (up to 38 %).
"Moved" events: from two previews per take on, the whole canvas is shifted
(diagonally, in % of the board width), rotated (about the frame centre) or zoomed
(about the frame centre) before the downscale, for three previews in a row.

**Still** (single pass; displacement = max corner distance in % of the board width):

| counted when | previews | not counted | jitter, consecutive: med / p99 / max | vs reference: med / p99 / max |
|---|---|---|---|---|
| R² ≥ 0.90 | 421 | 1.7 % | 1.37 / 10.7 / 16.4 % | 1.80 / 9.8 / 14.9 % |
| R² ≥ 0.97 | 421 | 11.9 % | 1.25 / 7.1 / 16.4 % | 1.56 / 5.0 / 8.4 % |

- **The tail is single-preview spikes.** Almost every still jump over 5 % goes
  out and back in the next preview (A→B large, B→C large), and most have R²
  0.92–0.96: a grid pulled by a hand or a crowded rank. Hence the stricter R² and
  the 2-preview persistence. The persistence, not the threshold, gives the margin:
  the 5 % threshold sits at the still p99.

**Moved** (single pass, R² ≥ 0.90; 66 previews per row, 22 events):

| move | true displacement (median) | located p10 | located median |
|---|---|---|---|
| shift 3 % | 3.1 % | 3.3 % | 4.3 % |
| shift 5 % | 5.1 % | 4.9 % | 6.0 % |
| shift 10 % | 10.1 % | 10.1 % | 11.2 % |
| rotate 3° | 4.8 % | 3.8 % | 5.1 % |
| rotate 5° | 7.5 % | 6.3 % | 7.6 % |
| rotate 10° | 14.7 % | 11.8 % | 15.2 % |
| zoom −5 % | 4.6 % | 3.3 % | 4.8 % |
| zoom +5 % | 4.4 % | 3.8 % | 5.4 % |
| zoom −10 % | 8.6 % | 7.2 % | 8.7 % |
| zoom +10 % | 8.6 % | 6.7 % | 9.7 % |
| zoom −20 % | 16.1 % | 13.8 % | 17.4 % |
| zoom +20 % | 16.7 % | 12.4 % | 17.2 % |

The located displacement tracks the true one with about +1 % bias (the still median).

**Rule** (θ on 2 consecutive counted previews; false fires over the 421 still
previews, 11 min; misses = events not caught within their 3 moved previews):

| counted when | θ | false fires | miss shift 3 / 5 / 10 % | miss rotate 3 / 5 / 10° | miss zoom ±5 / ±10 / ±20 % |
|---|---|---|---|---|---|
| R² ≥ 0.90 | 4 % | 15 | 12 / 2 / 1 of 22 | 3 / 0 / 0 of 22 | 12 / 1 / 2 of 44 |
| | 5 % | 4 | 15 / 4 / 1 of 22 | 12 / 0 / 0 of 22 | 24 / 1 / 2 of 44 |
| R² ≥ 0.97 | 4 % | 6 | 11 / 3 / 2 of 22 | 4 / 2 / 2 of 22 | 14 / 6 / 5 of 44 |
| | **5 %** | **0** | 15 / 5 / 2 of 22 | 13 / 2 / 2 of 22 | 26 / 6 / 5 of 44 |
| | 6 % | 0 | 17 / 13 / 2 of 22 | 18 / 4 / 2 of 22 | 31 / 6 / 5 of 44 |

- **The misses of big moves at R² ≥ 0.97** (2 of 22 even at 10 %) are events where
  fewer than 2 of the 3 moved previews passed R² 0.97. A real move persists, so
  the rule fires on a later preview; the 3-preview window undercounts them.
- **Metric:** max corner displacement and 1 − IoU of the boxes separate still
  previews from moves of ≥ 5 % true displacement equally well (AUC 0.998 and 0.996
  at R² ≥ 0.97). Max corner displacement also sees a rotation that keeps the box,
  and it reads in the same unit as the crop margin, so take that.
- **Low R² alone never triggers.** In the still set 11.9 % of previews fall below
  R² 0.97 (1.7 % below 0.90) with the camera unmoved: hands, a dark frame. A board
  that stays untrusted for long (a person in view, or a camera knocked off the
  board entirely) is a different signal: report it as "board not visible" after,
  say, 10 previews (15 s), not as `camera_moved`.
- **Caveat: 11 minutes of still previews is thin.** Zero false fires in 421
  previews bounds the rate only to below ~1 per 140 previews (95 %). Before
  shipping, replay a long still take (an hour, with play) through the rule.
- **Caveat: the canvas is synthetic.** The phone frame around the board is the room
  backdrop, not a real uncropped frame; no uncropped recording exists (see Gaps).

### Cost

| | median |
|---|---|
| fp32 GPU read with `square_attention` (served `-attn` file) | 195 ms |
| int8 CPU read with `square_attention` | 1,067 ms |
| grid fit (Python) | 8–16 ms |

The GPU was shared with another job during the run; the CPU read ran beside the
GPU run. At 40 previews per minute per camera, one GPU serves about 7 cameras on
drift reads alone, before any position reads. Reading every second preview
(every 3 s, still 2 consecutive = 6 s to fire) halves that.

### Reproduce

```bash
cd tools/probes
export AUTOCROP_SCRATCH=<scratch>      # holds quads/ from autocrop_label.py
flock /tmp/claude-1000/br-gpu.lock uv run python autocrop_preview.py locate $AUTOCROP_SCRATCH/prev
flock /tmp/claude-1000/br-gpu.lock uv run python autocrop_preview.py drift $AUTOCROP_SCRATCH/prev
CUDA_VISIBLE_DEVICES= uv run python autocrop_preview.py cost $AUTOCROP_SCRATCH/prev
uv run python autocrop_preview_analyze.py $AUTOCROP_SCRATCH/prev   # R² gate: MIN_R2
```
