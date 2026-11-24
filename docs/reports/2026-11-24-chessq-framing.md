> Experiment code and raw results: `scratch/experiments/chessq-framing/` (gitignored). Plan step 13; decides D1 in
> [docs/plans/2026-11-24-simplify-around-chessq.md](../plans/2026-11-24-simplify-around-chessq.md).

# ChessQ Lite V6: does it need a board-dominant crop?

Model: `models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx`, CPU, squash-resize of the whole input to 644x644 (cq_bench `OnnxModel`).
Images: `stills()` + `recording_frames()` = 467 (11 heldout-0831, 10 curated, 446 recording frames in 7 takes). Each image has its board quad.

## Method
`run.py` places the image on a larger canvas at scale 1 (cv2.warpAffine) so the board bounding-box width is a fixed share of the canvas width, board centered, canvas aspect fixed. Content outside the image is padded; if the canvas is shorter than the image, it is cropped. Conditions:
- `as-is`: image unchanged (board about 83 % of width).
- `50pct-16x9-reflect`, `35pct-16x9-reflect`: landscape 16:9, board 50 % / 35 % of width, centered, BORDER_REFLECT_101.
- `35pct-16x9-left-reflect`: 35 %, board left edge at 2 % of width (it does not fit inside the left third, 35 % > 33 %), vertically centered.
- `80pct-9x16-reflect`: portrait 9:16, board 80 % of width.
- `50pct-16x9-gray`, `35pct-16x9-gray`: same as the centered ones, but padded with constant RGB 128.

Subsampling (on the coordinator's request): as-is, 50 % and 35 % (reflect) ran on all 467. Left-35 %, portrait ran on a 264-image subsample (all stills + first/middle/last of the saved frames per take-segment). Gray ran on every 2nd image of that subsample (132). Matched tables below compare like with like. `rot!=0` = count of images where a 90-degree rotation of the prediction scores strictly higher than rotation 0: zero in every condition.

Files: `run.py`, `analyze.py`, `res_*.json` (per image), `log.txt`.

## Results (per-square % / perfect boards %)
All images for as-is, 50 %, 35 % reflect (n=467); subsample for the rest (n shown).

| condition | heldout-0831 | curated | recordings | ALL |
|---|---|---|---|---|
| as-is | 100.00 / 100% (11) | 99.84 / 90% (10) | 99.40 / 81.4% (446) | 99.42 / 82.0% (467) |
| 50 % 16:9 reflect | 100.00 / 100% | 99.53 / 80% | 99.30 / 78.5% | 99.32 / 79.0% |
| 35 % 16:9 reflect | 97.59 / 72.7% | 95.16 / 50% | 91.13 / 2.2% | 91.37 / 4.9% |
| 35 % left reflect (sub) | 85.94 / 27.3% | 78.59 / 30% | 72.32 / 0% (243) | 73.12 / 2.3% (264) |
| 80 % 9:16 reflect (sub) | 99.57 / 81.8% | 99.38 / 60% | 98.31 / 55.6% (243) | 98.40 / 56.8% (264) |
| 50 % 16:9 gray (sub/2) | 100.00 / 100% (6) | 99.69 / 80% (5) | 99.34 / 77.7% (121) | 99.38 / 78.8% (132) |
| 35 % 16:9 gray (sub/2) | 100.00 / 100% (6) | 99.06 / 60% (5) | 99.01 / 67.8% (121) | 99.05 / 68.9% (132) |

Matched subsample (n=264), per-sq / perfect: as-is 99.38 / 80.7 %; 50 % reflect 99.24 / 77.7 %; 35 % reflect 91.03 / 7.2 %; 35 % left 73.12 / 2.3 %; 9:16 80 % 98.40 / 56.8 %.

Matched gray subset (n=132), per-sq / perfect: as-is 99.38 / 80.3 %; 50 % reflect 99.16 / 75.0 %; 50 % gray 99.38 / 78.8 %; 35 % reflect 91.17 / 6.8 %; 35 % gray 99.05 / 68.9 %.

Rotation errors: 0 in all conditions (no image where a rotation other than 0 wins).

Per-take numbers: run `uv run python scratch/experiments/chessq-framing/analyze.py`. Every take drops the same way; 35 % reflect has 0 % perfect boards in 6 of 7 takes.

## Verdict
- 50 % of frame width (16:9, centered): effectively fine. Per-square -0.1 (reflect) or 0.0 (gray) vs as-is; perfect boards -3 points with reflect (82.0 to 79.0 on the matched full set; 80.3 to 78.8 on the gray subset with gray padding). Under the 1-point per-square threshold; the perfect-board loss is small and within the noise of ~450 correlated frames.
- 35 % width: padding-dependent, and that is the main finding. With reflect padding per-square falls 8 points (99.4 to 91.4) and perfect boards collapse (82 % to 5 %). With flat gray padding the drop is 0.3 points per-square but perfect boards fall 80 to 69 %, so it still loses about 11 points of perfect boards. The reflect result is partly an artifact: reflection mirrors the board and its pieces into the padding and adds phantom boards the model may attend to. A real wide frame holds a table, not mirrored boards, so the truth lies between the two (reflect is pessimistic, flat gray is optimistic). Either way 35 % is past the limit: the threshold of more than 1 point per-square or a material loss of perfect boards is crossed between 50 % and 35 %.
- Off-center board at 35 % (reflect): 73 % per-square, 2 % perfect. Position matters, but this condition also puts the board at the frame edge with most of the reflected content on one side, so it is a worst case; not tested with gray padding.
- Portrait 9:16 at 80 %: -1.0 per-square, perfect boards 80.7 to 56.8 % (matched). Taller canvas = reflected content above and below, and the board is down-scaled to 644 px width less than at 50 % (board is 80 % of the width, but the squash resize stretches the tall frame, distorting the aspect ratio by 3.2x). Not safe as-is.
- Note the squash-resize: a 16:9 frame is squeezed to a square, so squares become tall and thin (anisotropic), in addition to losing pixels. This is the real behavior of the Lichess preprocessing and is included in all numbers; a letterbox or crop would change them (not tested).

Recommendation: the edge should send a crop with the board at 50 % of width or more (as-is 83 % is best); do not send the whole wide frame. Board at 35 % is not safe, especially with realistic padding content and off-center placement. If a wide frame is unavoidable, crop to the board with margin before sending.

Caveat: padding is synthetic (no real wide footage), so borders look unnatural; real wide frames may add distractors (people, other boards) that these tests do not capture.
