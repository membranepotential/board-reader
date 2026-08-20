# Zero-shot piece recognition: campaign report (2026-07-17/18)

**Goal.** Make the board reader work out of the box on a chess set it has never
seen ("zero-shot"), instead of requiring per-set calibration.

**Outcome.** A multi-style-trained ResNet18 reads the operator's set — never seen in
training — at **92.5% per-square accuracy (occupancy 99.7%)**, matching the
per-set *calibrated* exemplar bank (91.9%) with zero calibration. Every
training-free route was tested first and measured dead (~80% ceiling, all
gated by piece-type confusion). The model is deployed as the app's default
estimator (see §5).

---

## 1. The problem, quantified

Recognition = per-square 13-class classification (empty + 6 types × 2 colors)
over crops cut from the photo (square base + 0.15 headroom). Reference points
on the 5-photo / 320-square labeled eval of the operator's set:

| Baseline | Accuracy |
|---|---|
| Same-style calibrated kNN exemplar bank (LOO) | **294/320 = 91.9%** |
| Any cross-style (different physical set) approach | ~80% ceiling (below) |

The failure signature is always the same: **occupancy transfers across piece
styles (94–99%), piece *type* does not** — rooks/knights/bishops/queens/kings
collapse to 0–40% recall across the style gap; empties and pawns survive.

## 2. Training-free routes: seven attempts, seven negatives

| # | Approach | Cross-style /320 | Verdict |
|---|---|---|---|
| 1 | kNN exemplar transfer (ChessReD bank) | 255 | style-entangled features |
| 2 | Linear probe trained on ChessReD embeddings | 254 | ties kNN → gap is in features, not classifier (probe hits 90.7% within-style) |
| 3 | SAM2-segmented crops (mask out clutter) | 255 (bbox) / 238 (in-place) | no fix; masking *hurts* same-style (291→280/261) |
| 4 | Classical embedding alignment (~50 configs: CORAL, ZCA/PCA whitening, z-score, top-PC removal) | ≤252 | all within noise; style is not a linear offset nor in dominant PCs |
| 5 | Published pretrained models | LC2FEN 256, chesscog 151 | nobody else solved it either |
| 6 | Bigger/different frozen backbones (DINOv2-B, SigLIP2, OpenCLIP, ImageNet-ResNet) | ≤250 | scale flat (S→B: +3) |
| 7 | Shape-biased SIN-ResNet50 (Stylized-ImageNet) | 219 | *refuted instructively*: best same-style (296), worst cross-style — shape bias over-specializes |

One transferable nugget from #6: **DINOv2-S central-column patch pooling**
(mean of middle 8/16 token columns) beats CLS by +9 (256/320) with
near-perfect occupancy — queued as a cheap upgrade to the existing kNN path.

Sub-findings worth keeping: per-class MAX cosine beats class-means by ~40
squares (validated; keep it); CORAL in PCA mode is pathological (use ZCA);
fuller/taller crop boxes lose to short consistent ones (3D boxes: 283 vs 294).

## 3. What worked: multi-style supervised training

**Thesis** (confirmed): piece-type style-invariance must be *learned* from
many piece styles; no single-style data or frozen feature trick gets there.

**Data assembled** (per-style dirs, common `labels_normalized.json` schema):

| Source | Boards | Styles | Notes |
|---|---|---|---|
| ChessReD (real, CC BY-NC-SA) | 2,078 | 1 set | corner-annotated subset |
| ChessRender360 (synthetic, CC BY) | 5,992 | 1 geometry × 3 materials | corners derived from instance masks |
| ChessR Blender renders (ours) | 890 | 4 geometries × 6 material families | material re-rolled per render; dim scenes included; CGTrader-derived geometry (license murky → segregated, droppable) |
| chesscog transfer set (real) | 22 | 1 set | held-out validation style |
| Zenodo piece crops (real, CC BY) | 3,200 crops | 4 sets | top-down framing; aux only |
| + 8 external CC0/CC-BY 3D sets | render-ready | 8 geometries | wave 2, import-validated |

**Recipe** (run "a"): ResNet18 (ImageNet init), 224² crops cut by the *deployed*
crop code, heavy shape-bias augmentation (channel shuffle, grayscale,
posterize/solarize, noise, AdaIN-style color-stat randomization, dim gamma),
fp32 (see §6), class-balanced sampling, 565,888 crops across 5 training styles,
early-stopped at epoch 6.

**Results on the operator's set (never in training):**

| Metric | Frozen best | Calibrated bank | **Trained (zero-shot)** |
|---|---|---|---|
| 5-photo eval accuracy | 80.0% | 91.9% | **92.5%** |
| Occupancy | 96–99% | 97.8% | **99.7%** (0 missed, 1 halluc) |
| Dim-evening uploads | ~poor | good (after dim calibration) | 84.4% zero-shot |
| Grayscale robustness | — | — | −0.4pt (fully color-invariant) |

The remaining errors are 23 piece-type confusions — the same weakness class,
now at a quarter of its former size, with waves 2 styles and candidates (b)/(c)
still untried.

**Validation caveat:** the chesscog-transfer holdout collapsed (50%) while
real-photo performance soared — the holdout instrument itself is suspect
(extreme top-down camera; investigation open). Tuning decisions were not made
against the operator's photos (report-only).

## 4. App improvements shipped alongside (all tested, mobile-verified)

- **Start-position auto-calibration**: occupancy pattern of a starting position
  (style-robust) triggers silent per-set calibration; any rotation; White's
  side by brightness; agreeing-squares mask prevents pollution. Persisted
  opt-in via `BR_SESSION_BANK`.
- **Calibrated-bank self-refresh** on low-confidence start sightings (lighting
  drift tracking; validated by the dim-room result: ~50%→98.8% via appends).
- **Occupancy-only mode + hint** for unknown sets (pre-training stopgap).
- **Mobile UI overhaul**: zero-tap photo→position, camera-roll fix (iOS
  `capture=` removed), scroll-safe corner handles (touch-action discipline),
  low-light warning, and a **distribution correction panel** (privileged Empty
  row, 2×6 color-symmetric piece grid with probability bars, stays-open
  multi-fix, live Lichess links).
- **Upload retention** (`BR_SAVE_UPLOADS`): every phone photo becomes labeled
  eval data (this is what made the dim-light forensics and bank rebuilds possible).

## 5. Deployment (verified live 2026-07-18 ~11:40)

- Model: `models/piece_classifier_multistyle_v1.onnx` (ResNet18, 13 logits in
  protocol CLASSES order; input RGB/[0,1]/ImageNet-normalized NCHW 224²).
- Integration: new `TrainedEstimator` (`board_reader_inference.trained`) behind
  the existing estimator seam; server default via `BR_MODEL`; precedence
  `BR_CENTROIDS` > `BR_MODEL` > session-restore > occupancy/stub. `/calibrate`
  still hot-swaps a per-set bank over it (optional booster).
- Verification chain: trainer fixture reproduced bit-for-bit (max logit
  deviation 4.5e-5); full eval through the deployed estimator = **296/320
  (92.5%)**, identical to the training-side eval; live server read 62/64 at
  conf 0.92 in **0.6 s** end-to-end.
- Serve: `BR_MODEL=models/piece_classifier_multistyle_v1.onnx
  BR_SAVE_UPLOADS=data/uploads BR_SESSION_BANK=models/exemplars_session.npz`.
  The occupancy-bank/auto-calibration path is no longer loaded by default (the
  trained model IS the zero-shot path); dim-light remains the trained model's
  weakest regime (~84%) where per-set calibration still beats it — the
  calibrate button covers exactly that gap.
- **License**: the model was trained on ChessReD (CC BY-NC-SA) + murky-license
  ChessR geometry — fine for personal use, NOT redistributable/commercial
  as-is. A clean-data retrain is the path if that ever changes.

## 6. Incidents & durable lessons (also in `.learnings.ndjson`)

1. **Session-bank clobbering, twice.** Server-side cwd-relative persistence let
   ordinary test runs overwrite the real calibration bank; the second time a
   restart loaded the clobbered file and served garbage. Structural fix:
   persistence is opt-in (`BR_SESSION_BANK`), guarded by a no-write test.
   Ops rule: verify a restart with a known-photo read, not just healthz.
2. **Never SIGSTOP/SIGCONT CUDA training.** Resumed process became a zombie-
   spinner: GPU 100%, ~30× throughput collapse, invisible to liveness checks.
   Pause = checkpoint + kill + resume. Monitors must track *throughput*.
3. **GTX 1650 precision facts (measured):** fp32 2.63 steps/s > bf16 0.99
   (emulated) > fp16 0.78 + forward-pass NaN (GradScaler can't fix forward
   overflow). Use fp32.
4. **Fresh-context reviewer over long-context author** caught real bugs twice
   (touch-action regression; detached-DOM `contains()` bug → use
   `composedPath()`). Retire agents whose context balloons.
5. **Licensing ledger:** ChessReD CC BY-NC-SA (non-commercial; nothing
   committed), ChessRender360/Chesscog/Zenodo CC BY, ChessR geometry murky
   (segregated + droppable), 8 scouted sets CC0/CC-BY (catalogued).

## 7. Open items

- Replace/repair the chesscog-transfer validation holdout (top-down domain).
- ~~Candidate (b) SIN-init~~ (ran: best val 0.665 on the vlab holdout — loses,
  consistent with the SIN frozen negative; dead); candidate (c) DINOv2
  fine-tune deprioritized in favor of the §8 crop-geometry fix.
- Wave-2 renders → retrain; expect further piece-type gains (§8 refocuses the
  target regime).
- Central-column patch pooling for the kNN/calibration path.
- ~~Calibration-as-booster on top of the trained model (ensemble)~~ — shipped,
  see §8.
- Nothing committed to git yet — the whole campaign is one working tree.

## 8. Field session addendum (2026-07-18 afternoon)

The operator photographed live games (same wooden set, steeper ~25–45° camera, warm
dim room light) and the deployed model disappointed. Root-caused and partly
fixed the same day.

**Diagnosis — the crops decapitate.** The whole pipeline (training cropcache
and serving alike) had settled on `headroom=0.15` square-widths — near-base-only
crops, against the design intent of ~1.6 ("whole standing piece"). At steep
elevations a tall piece's head lies a full square above its base, so the
classifier only ever sees lower bodies: on the hard board
(`up_20260718_0001`) the model scored 57/64 with **every major piece wrong** —
q→b, K→Q, R→P, B→P — plus phantom pawns hallucinated from cut-off heads one
square up. A second shot taken from Black's side additionally came back
180°-rotated (auto-detected corners carry no orientation; nothing corrected it).
Two of today's shots are now labeled eval boards (`data/eval/ground_truth.json`,
7 boards / 448 squares).

**Shipped the same day (all tested, 195/195 green, deployed + live-verified):**

| Change | Measured effect |
|---|---|
| `EnsembleEstimator`: trained model ⊕ session bank, 50/50 distribution blend; `/calibrate`, auto-calibration, self-refresh and boot-restore all wrap rather than replace the trained model | 439/448 vs 415 (trained) / 430 (bank); live re-verify: img_0566 56→62, up_0000 62→64, up_0001 57 (unchanged — needs the retrain) |
| Trained-path **bootstrap auto-calibration**: the model's own start-position sighting builds its booster bank, zero taps | new capability (previously the trained path never auto-calibrated) |
| **Orientation auto-correction** (`autorotate=1`, sent by the UI unless the user hand-placed/relabeled corners): fewest pawns on ranks 1/8, tie-break White-nearer-rank-1 | fixes the 180°-flipped field shot; board-printed coordinates/square parity deliberately unused (players ignore printed coords; non-FIDE boards exist — both seen today) |
| Upload sidecars now record the predicted `placement` + `autorotated` | every field shot becomes eval data without forensics |

**Second incident, same afternoon (bank corruption #3).** With autorotate live,
The 16:23 trial exposed a trap the fix itself created: auto-detected corners
were 180° off, autorotate silently corrected the *reads* (so everything looked
fine), but his **Calibrate** tap still trusted the corner orientation — pushing
color-swapped exemplars into the persisted bank. Reads degraded to ~58/64 with
White/Black flips on the back ranks. Fixes, shipped + live-verified: (1)
`/calibrate` now probes the photo with the stateless trained member and rotates
the claimed labels into agreement (`label_rotation` in the response); (2)
`tools/rebuild_session_bank.py` rebuilt the bank from all 11 retained calibrate
uploads with the same verification — it flagged exactly the one poisoned
calibrate (rotation agreement [32,19,**56**,20]) and salvaged it as
correctly-labeled bright-room data. After rebuild: the correct-corner start
photo reads **64/64** live; the flipped-corner one 62/64 (residual errors are
the decapitation class the tall-crop retrain targets). Meta-lesson recorded:
an affordance that silently corrects an error (autorotate) masks that same
error from every flow that still trusts it (calibrate).

**Third trial, 17:00 — the operating envelope, and incident #4.** The operator's next
shot was taken seated across the table, phone barely raised: projected board
aspect (quad short/long side) **0.56**, versus 0.73–0.88 in every previous
photo. There the model's output goes near-random — pieces occlude whole ranks
and each 0.15-headroom crop mostly contains the *next* rank's piece. Corners
and orientation were fine; the regime is simply outside anything trained or
evaluated. He then tapped Calibrate on that (mid-game) board, which labeled
mid-game crops as the start position — bank poisoning #4, caught in the same
forensics. Shipped, tested, live-verified:

- `/estimate` warns `very_low_angle` below aspect 0.62 and the UI tells the
  user to raise the phone (like the low-light hint).
- `/calibrate` now **rejects** (HTTP 422, reason shown in the UI) when fewer
  than 60% of the *claimed piece squares* agree with a stateless probe of the
  photo. Piece-square agreement is the right metric — whole-board agreement is
  inflated to ~40/64 by the empty middle that mid-game and start positions
  share, which is exactly how this poisoning slipped past intuition.
- `tools/rebuild_session_bank.py` applies the same gate; the re-rebuild
  rejected precisely the one poisoned calibrate (10/32 piece agreement) and
  the clean bank is redeployed (64/64 and 62/64 on the two start photos).
- Wave-2 render sampling extended to cover aspect 0.5–0.9 (mode ~0.75) so the
  retrain learns the flat-angle regime instead of only warning about it.

**Tall-crop sweep result (evening) — d16 DEPLOYED.** The headroom sweep
confirmed the decapitation diagnosis quantitatively: same recipe/data, only the
crop geometry changed. vlab holdout: 0.665 (SIN run b) → **0.835** (d16,
hr1.6); user photos: 415 → **430**/448 alone. hr1.8/sp0.25 (d18) ties on the
holdout but loses on real photos (409) — 1.6 is the deployed geometry, and
geometry now ships with the model (`<model>.onnx.meta.json` sidecar read by
`TrainedEstimator`; a train/serve geometry mismatch silently decapitates
crops). The eval grew to 10 boards / 640 squares (three of today's
start-position uploads have by-construction GT):

| config | /640 |
|---|---|
| v1 alone | 579 |
| d16 alone | 605 |
| v1 ⊕ bank 50/50 (previous deploy) | 626 |
| **d16 ⊕ bank, maturity-weighted (deployed)** | **633** |

Blend weight now depends on bank maturity (≥1024 exemplars → 30/70
trained/bank, else 50/50): a mature bank has earned the trust, a one-photo
bootstrap hasn't. Live-verified: both bright start photos 64/64 and 63/64,
img_0566 64/64; the steep+dim board 57 → 59/64 (wave-2's target).

**In flight (agents):** dataset-bank is bulk-rendering wave-2 (6 wooden-Staunton
styles × 300, coordinate-border boards, warm dim light, camera aspect 0.5–0.9
including the near-level tail) → run (e) convergence retrain on ~11 styles.
Ops note for the pipeline: two GPU-idle stalls today (a notification-gated
trainer phase, and this session's own server squatting 2.1GB via its DINOv2
CUDA session) — fixed by a self-driving sweep script and CPU-masked serving.
