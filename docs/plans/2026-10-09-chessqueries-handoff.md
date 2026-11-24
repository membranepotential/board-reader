# Handoff — ChessQueries / Lichess "From image" as our estimator (2026-10-09)

**State: preliminary investigation done, benchmark written but never run** (the
user stopped it to free the CPU). No number in this file was measured on our data.

## The question

Lichess shipped board recognition from photos. Can it replace or seed our
recognizer, given the three things that stalled the project: hardware limits,
accuracy not good enough, and too little labeled data?

## What Lichess shipped

- Board editor → "From image…" (lila PR #21908, merged 2026-10-03, commit
  `ef8db2c`). It runs fully client-side: `onnxruntime-web`, int8 ONNX on CPU
  (`public/lifat/vision/chessq-lite-v4-int8.onnx`, 41 MB). Per the PR author,
  the int8 model outperforms the fp16 one.
- The model is **ChessQ Lite v4** (ViT-S) by Joël Seytre, a small version of
  **ChessQueries** (arXiv 2608.30762, repo `github.com/JSeytre/chessqueries`,
  weights `huggingface.co/joelseytre/chessqueries` and `.../chessq-lite`).
- Input contract, ported from `ui/editor/src/editor.vision.worker.ts`: squash-resize
  the **whole image** to 644×644, divide by 255, ImageNet mean/std, NCHW float32
  input `image`. Output `logits` (1, 64, 13): squares in FEN order a8…h1, classes
  `.PNBRQKpnbrqk`. No board detection, no corners, no rectification.

## Why it matters for us

| Our blocker | What ChessQueries offers |
|---|---|
| Hardware | Lite runs on CPU inside a browser tab. Our shipped read costs ~1.3 GPU-s. |
| Accuracy | Paper (full ViT-L, exact-board accuracy): ChessReD **99.5 %** (prior SOTA 15.3 %), SLCC broadcasts **87.1 %**, CVChess zero-shot **87.6 %**. |
| Data | SLCC labels come from aligning broadcast frames to Lichess relay PGNs (OCR of names and clocks picks the ply, a model ranks candidates, a human reviews each kept sample). That is our parked auto-labeling idea (docs/plans/2026-08-31-autolabel-engine.md), built and released. A LoRA on **10 frames** of a new domain gives a large gain (SLCC: ~24 % board accuracy with 10 frames vs 0 % zero-shot). That is a candidate replacement for `/calibrate` and its non-monotone bank writes. |

- **Contract fit.** A stateless per-frame estimator whose per-square softmax is a
  per-square confidence, which is exactly DESIGN D1 and the L3 protocol. Swapping
  it in touches only the inference node's estimator; nothing on the wire changes.
- **Consistent with our own finding.** The paper reports that a frozen encoder
  breaks the linear head and that full fine-tuning at 644 px wins. This matches
  docs' "frozen-encoder ceiling ~80 %" (the backbone bakeoff and the DINOv3 probe).

## Caveats

- **License.** Code and weights are PolyForm Noncommercial 1.0.0, and the SLCC
  annotations are CC BY-NC 4.0. That is fine for this project, but blocks any
  commercial use. Don't commit the weights to the public repo; download them at
  build time.
- **No published numbers for Lite.** All the paper's figures are for ViT-L. Lite v4
  is "further trained on proprietary data", and its HF validation JSON is unread.
- **Known failure modes (paper).** Upside-down reads, two white kings, pawn-direction
  confusion, heavy occlusion, small or low-resolution boards. Dropping the
  resolution 644→448 costs SLCC 87.1→81.6 %.
- **Full model cost.** ViT-L is 372 M params (743 MB fp16), 19 ms/board on a 4090
  and ~590 ms on an M3 Pro. On our GTX 1650 it is unmeasured.

## The benchmark (ready, never run)

Local only, `scratch/` is gitignored: `scratch/experiments/chessq-lite-probe/`

- `cq_bench.py <model.onnx> cpu|cuda <out.json>` runs the ONNX file through the
  workspace's own `onnxruntime` and our own preprocessing. No third-party code runs.
  Inputs:
  - the 10 curated stills (`data/eval/ground_truth.json`) and the 12 held-out
    08-31 stills (`data/eval/heldout-20260831/`);
  - up to 6 frames from the middle half of every truth segment of each
    `recordings/*/truth.json` take.

  Every input runs twice: `full` (whole frame) and `crop10` (axis-aligned bbox of
  our detected corners plus a 10 % margin). The script records per-square
  correctness, the best of 4 rotations (to catch upside-down reads), the errors,
  and confidences.
- `analyze.py <out.json>` prints a table per set × mode, a per-segment majority
  vote (a crude stand-in for the settled stream), the top confusions, and a
  confidence-separation check.
- `weights/chessq-lite-v4-{int8,fp16}.onnx` were fetched from
  `media.githubusercontent.com/media/lichess-org/lila/ef8db2c/public/lifat/vision/`.
  The int8 SHA-256 is `89d75e34…32ae1f`, which matches lila's LFS pointer.
- **CPU-only in practice.** The CUDA provider silently fell back to CPU (the GPU
  stayed at 167 MiB). Expect ~10 min of CPU for the int8 run; run it when the
  machine is free.

## Bar to beat (shipped stack)

| Set | Shipped stack | Note |
|---|---|---|
| held-out 08-31 stills | 96.6 % per-square, 1/11 perfect | cold for both |
| gate takes, raw per-read (`tkr-shipped` replays) | 08-13 96.0 % · 08-27-1755 92.2 % · 08-27-1818 88.5 % · 08-28 91.6 %; 0–2 perfect per take | our bank has seen these sessions; ChessQ is cold |
| gate takes, settled | 22/31 · 5/12 · 3/43 · 5/23 perfect | compare to the majority vote only loosely |

## Next steps, in order

1. Run `cq_bench.py` with int8 on CPU, then `analyze.py`. Decide on the held-out
   stills and the raw gate numbers. Check whether `crop10` beats `full`; our
   frames are wide phone shots, and the model was trained on board-dominant images.
2. **If Lite matches or beats the shipped stack cold:** prototype it as an estimator
   behind the inference node, then run the standard replay gate against `tkr-shipped`.
3. **The full ViT-L needs the user's permission to install and run the
   `chessqueries` Python package.** The auto-mode classifier blocked this as
   "Code from External" on 2026-10-09. The alternative is to write our own loader
   for the safetensors weights, using their `minimal_reproduction/model.py` as the
   reference.
4. If the full or Lite model is close but short: try a 10-frame LoRA on our own
   recordings (their `scripts/lora_fewshot.py`). That is the calibrate replacement
   test.
5. Data: evaluate their SLCC producer (`chessqueries-annotate`) as our
   auto-labeling engine instead of building ours.

## Results (run 2026-11-24): GO — all three models beat the shipped stack cold

`cq_bench.py` on CPU, whole frame (`full`); `crop10` is within ±0.5 points
everywhere, so cropping does not matter. `cq_bench.py` now also loads `.tflite`
(ChessCam), run from its own env `scratch/experiments/chessq-lite-probe/litert-env`
(`ai-edge-litert`). Raw outputs are in `out/*.json`.

| Model | ms/frame (CPU) | held-out 08-31 stills | curated stills | gate-take frames, per-sq | perfect frames | segment vote, perfect |
|---|---|---|---|---|---|---|
| **Shipped stack** | ~1.3 GPU-s | 96.6 %, 1/11 | — | 88.5–96.0 % | — | — |
| ChessCam v2 (MobileNetV4, 320 px, fp16 tflite) | **45** | 100 %, 11/11 | 98.3 %, 6/10 | 98.3 % | 283/413 | 54/73 |
| ChessQ Lite v4 (int8) | 1133 | 100 %, 11/11 | 99.5 %, 7/10 | 98.7 % | 291/413 | 55/73 |
| **ChessQ Lite V6** (int8, HF `joelseytre/chessq-lite`, SHA-256 checked) | 1076 | 100 %, 11/11 | **99.8 %, 9/10** | **99.2 %** | **324/413** | **61/73** |

- No model ever read a board upside down (`rot!=0` = 0 everywhere).
- **The remaining errors are mostly hands.** Errors cluster in whole truth segments and
  in one board region. For example, 08-28 frame #48 has a hand over a1–d2, which is read
  as 7 empty squares. The sampler takes the middle half of each segment, which does not
  exclude hands. The errors are confident: mean p(argmax) on wrong squares is 0.95 for
  ChessCam. So a hand gate (our masking, or the stabilizer) is still needed. Board
  min-confidence does separate perfect from imperfect reads (0.99 vs 0.81 for ChessCam).
- Latencies for v4 and V6 were measured while other CPU jobs ran, so they read high.
- The segment vote is a crude stand-in for the settled stream. The real comparison is
  the replay gate (next step).

### Next
1. Wrap V6 (accuracy) and ChessCam v2 (speed) as estimators in the inference node, then
   run the replay gate against `tkr-shipped`.
2. Test whether ChessCam v2's square-centre output (`(1,64,3)`: x, y, sigma) can replace our
   corner detection. Compare its homography with the recorded corners.
