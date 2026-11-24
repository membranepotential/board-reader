# Plan — simplify the pipeline around ChessQ (2026-11-24)

**Decision (user, 2026-11-24):** ChessQ Lite V6 becomes the core recognizer, and the
recognition pipeline is simplified around it; then the settle window is tuned.
Evidence: [docs/reports/2026-11-24-chessq-v6-replay-gate.md](../reports/2026-11-24-chessq-v6-replay-gate.md)
(4-take gate: 92/101 raw, 84/101 settled perfect vs the shipped stack's 5 and 35).

## Key facts (verified in code)

- On the ChessQ path **none of the old recognition stack changes the output**: corners,
  crops, bank, attractor demotion, autorotate and debug geometry are skipped or
  ignored (`chessq.py` `estimate_photo` → `estimate`). Its only borrowings are
  `trained.preprocess/softmax/LogitsFn`, `routing.RECTIFIED_GEOMETRY` (debug only),
  `squares.CropGeometry`, `onnxrt.cuda_providers`.
- **Regression gate for every deletion step:** replaying the four gate takes must
  reproduce `cq-v6`'s placements, prior and settled boards (evidence may differ by
  float noise only).
- **Frames:** gate truths, `data/eval/ground_truth.json` and `heldout-20260831` are
  chess-frame in practice (the cold probe found no rotation anywhere).
  `data/uploads/*.label.json` (27) are genuinely corner-frame and need migrating.
  Recordings 163352/170333/172736 have no ChessQ replay; confirm their frame first.
- **Crop vs whole frame is unmeasured on wide framing.** Every gate frame is the
  /camera bbox crop (board ≈ 83 % of width); full-vs-crop10 was measured on those.
- `stabilize.py` imports `prior.tally/violation`, so `prior.py` stays.
- `replay.config_digest` hashes only set env vars, so keeping the name `BR_CHESSQ`
  keeps digests continuous with `cq-v6`.
- The e2e "real pixels" tests use `CentroidEstimator` + `LiteEmbedder` on `core.sim`
  renders; they need a test-side `SimRecognizer` before `centroid.py`/`embed.py` go.
- Replay cannot measure edge changes (crop and stability are baked into recordings);
  edge changes need a live take.

## Target architecture

1. **Edge** (Pi CLI or /camera): capture → `MotionMeter` stability →
   `AdaptiveForwarder` → JPEG (whole frame or a plain ROI box, per D1), with `seq`,
   `capture_ts`, `stability`, `sharpness`. No detection, rectification, corner labels
   or calibration.
2. **Wire** unchanged; `corners` is legacy: still parsed (old recordings replay), never
   sent, ignored.
3. **Inference:** `FrameIngest` (latest-wins inbox, `prefer_frame`, `BurstTriggered`,
   backstop) → `recognizer.read(image)` → `PositionEvent` → `prior.decode` (advisory)
   → `BoardStabilizer` (retuned) → hub + recorder.
4. **One seam,** `recognizer.py`: `Recognizer` protocol (`name`, `read(image) -> (8,8,13)`
   chess-frame dist), `StubRecognizer`, `event_from_dist`, `build_from_env()`.
   ChessCam v2 later = one more file.
5. **HTTP:** `POST /estimate` (photo only), `/labels`, `/uploads`, `/boards/*`, `/healthz`.
   Gone: `/calibrate`, `?corners`, `?autorotate`, `?debug`, and `/detect` (per D1).
6. **The chess frame is the only convention** for events, labels, truth, eval, replay and the viewer.
   `orientation` stays as a constant.
7. **Measurement:** `replay` (+ settle flags, `make gate`), `eval-session` (+ truth-frame
   diagnostic), `eval` (stills via the recognizer).
8. **Viewer:** live board, /camera (viewfinder, stream, health), /snapshot
   (photo → board → correct → label, 4 stages).
9. **No banks, no in-house models, no training harness.** Git keeps the code;
   `docs/reports` keeps the research.

## Commit sequence

Each step keeps `make test` green and `make dev` demoable; from step 3 on, also the
replay invariant above.

**Phase A: ChessQ becomes the core, nothing deleted yet**
1. `docs(design)`: ADR "ChessQ is the core recognizer"; amend DESIGN (list below).
2. `refactor(inference)`: `recognizer.py` seam; `chessq.py` self-contained (inline
   preprocess/softmax), `frame = "chess"`; `test_read_seam` and `test_chessq` updated.
3. `feat(inference)`: chess frame for labels and eval:
   - `/estimate` ignores `autorotate` under a chess-frame recognizer;
   - `/labels` needs no corners and writes `frame: "chess"`;
   - `labels.py` gains `frame` and optional corners;
   - `evaluate.py` reads via the recognizer, skips non-chess labels, and adds a rotation diagnostic;
   - `trial.py` adds a truth-frame check;
   - add `make gate` and `tools/gate_table.py`.
4. `feat(tools)`: `tools/migrate_labels_to_chess_frame.py`:
   - choose the rotation only, with a margin, cross-checked against `placement_shown`;
   - back up the old files, and eye-check every disagreement;
   - mark the eval stills `frame: "chess"`;
   - replay ChessQ on the 3 non-gate takes.
5. `feat(deploy)`: ChessQ is the default:
   - compose and Makefile set `BR_CHESSQ`;
   - `make models` fetches the pinned HF revision and checks the SHA-256;
   - missing weights degrade to the stub, loudly (D2);
   - update `test_deploy_parity`, README and THIRD-PARTY;
   - measure ChessQ on `sim` renders (D3).

**Phase B: delete the old stack, UI first**
6. `refactor(ui)`: remove the calibration surfaces (/camera calibrate and auto-calibrate,
   `BoardCard` calibrate, `calibration.ts`, /snapshot calibrate and occupancy,
   `api.calibrate`).
7. `refactor(inference)`: remove calibration and the bank:
   - `/calibrate`, `Uncalibrated` and occupancy;
   - `session.py`, `autocalib.py`, `rebuild.py`, `cli calibrate`/`rebuild-bank`;
   - `BR_SESSION_BANK`/`BR_OCCUPANCY_BANK`/`BR_CENTROIDS`;
   - the bank tools.
8. `refactor(inference)`: remove the crop-classifier stack:
   - `trained.py`, `centroid.py`, `embed.py`, `estimator.py` (Ensemble, Centroid, `demote_attractor`, `read_event`);
   - `BR_MODEL*`, `BR_CASCADE_TAU`, `trial.AttractorGuard`, the `evaluate` composites, the `CONFIG_ENV` trim;
   - add `tests/_sim_recognizer.py` for the e2e and ingest tests.
9. `refactor(ui)`: /snapshot becomes 4 stages (capture → read → repair → output); drop
   `CornerPicker`, `framing.ts`, detect/rectify/sample/orient, debug images and rotate labels.
10. `refactor(inference)`: one read path. Delete `routing.py`, `squares.py`
   (`BR_CROP_MASK`), `diagnostics.py`, `?debug`, `very_low_angle` and `X-Corners`.
11. `chore`: delete `tools/train/**` (and its pyproject testpaths),
   `export_dinov2_onnx.py`, `repair_upload_orientation.py`, `corner_picker.html` and
   `trial.export_crops`; delete or keep RunPod per D7.
12. `docs`: vault rewrite:
   - `recognition-pipeline.md` rewritten;
   - `calibration-and-the-bank.md` and `training-and-data.md` become historical stubs;
   - architecture, measuring-accuracy, operating-envelope, runbook, glossary, TESTING,
     CLAUDE.md, aspects.yaml and README updated.

**Phase C: edge and corners (needs a live take)**
13. `exp`: does ROI cropping matter? Pad the gate crops to a 16:9 field (board at ~50 %
    and ~35 % of the width), and portrait. Report only.
14. `refactor(ui,edge)`: framing per D1:
    - /camera drops the tracker, homography, corners, rotate labels, re-detect and the portrait block;
    - the edge drops `rectify.py`, `quadcheck.py` and `--rectify/--corners/--forward-original`;
    - validate with one live take.
15. Remove `/detect`, `core.boarddetect` (move `quad_aspect` into `tools/video`) and `core.corners`.

**Phase D: settle tuning (any time after step 5)**
16. `--settle-seconds/--settle-min-reads/--stale-windows` on replay (manifest, slug,
    digest) and `BR_SETTLE_*` on serve.
17. `exp` sweep: settle 0.5–2.5 s × min_reads 1–3 × `burst_end_reads`/`spacing_s`/`keepalive_s`.
    - Primary metric: settled perfect /101.
    - Secondary metric: median move-to-settle time.
    - Hard constraint: no hand frame settles (08-13 stays 31/31).
    - Optional variant: a count-drop guard.
18. Ship the new defaults with a gate row.

## DESIGN changes (step 1)

- **Changes:**
  - #3: no calibration round.
  - #6 / D4: the edge sends the frame or an ROI; no detection or rectification.
  - #7, D6, §4.6, #16: ChessQ behind a `Recognizer` seam; PolyForm NC accepted.
  - #15, §4.4.1, §9: the chess frame, resolved per frame by the recognizer.
  - D8, §4.4: hands are handled by burst-end scheduling and the debounce, not by confidence.
  - §4.2, §4.5: `corners`, `/detect` and `/calibrate` go.
  - §5: `orientation` semantics and the settle values.
  - §7: M5 (calibration) replaced by an optional offline LoRA.
  - §8: WP-A shrinks, WP-B is replaced, WP-E is removed.
- **Unchanged:** D1 (stateless; prior and stabilizer advisory), D2, D3, D5, D7, D9, the
  edge/inference separation, and user-testable, fully tested milestones.

## Decisions

| # | Decision | Resolution |
|---|---|---|
| D1 | Edge framing: quad tracker / operator rectangle / whole frame | **Measured 2026-11-24 ([report](../reports/2026-11-24-chessq-framing.md)): keep a board-dominant crop (board ≥ ~50 % of frame width; 35 % collapses perfect boards, portrait costs too). → B: a plain rectangle crop with margin; no corner labels, no quad tracker. Whole frame is out.** |
| D2 | Missing weights on a fresh clone | Degrade to the stub, loudly; `make models` |
| D3 | What the compose demo reads | Measure ChessQ on sim renders in step 5 |
| D4 | Prior repair | Keep (cheap safety net; the stabilizer needs `tally`/`violation`) |
| D5 | Calibration | Remove; any adaptation is an offline LoRA → new ONNX |
| D6 | Old model assets | Archive to the HDD, outside the repo |
| D7 | Training + RunPod tooling | **User 2026-11-24: delete both** (`tools/train`, `tools/runpod`, the runpod skill) |
| D8 | Protocol `orientation`, statuses | Keep, documented as reserved |
| D9 | Env naming | Keep `BR_CHESSQ` |
| D10 | Model `direction` output | Ignore for now |
| D11 | PolyForm NC licence as the core | **User 2026-11-24: accepted** |
| D12 | Settle gate metric | Settled perfect + latency + a hand-leak constraint |

## Risks

- **One third-party, noncommercial model.** Pin the revision, verify the SHA-256,
  keep a local copy. ChessCam v2 is the hedge, also for Pi/Jetson latency.
- **Hands read as confident empties** (p ≥ 0.96). A shorter settle window raises
  leakage, so 08-13 must stay 31/31.
- **Wide framing and small boards are unmeasured**, and replay cannot test edge changes.
- **The label migration uses the model to choose the rotation:** rotation only, with a
  margin and a cross-check, never squares.
- **Orientation is now inferred by the model** (DESIGN #15 reversal). Rotation
  diagnostics in eval and eval-session are the guard.
- **`tkr-shipped` can only be rescored, never re-replayed,** once the old code is gone.
  Keep the replay dirs and keep `trial.py` compatible with old event shapes.
- **Re-fixture shared tests** (`test_server`, `test_prior_serving`, `test_evaluate`) on
  the ChessQ fake rather than deleting them.
- **Old recordings carry `corners`:** `wire.Frame` must keep parsing them.
- **The gate is four takes on one physical set.** Record more varied sessions before
  trusting the settle defaults broadly.
