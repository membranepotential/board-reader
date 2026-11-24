# Camera-Based Chess Board Reader — Design Document

> A distributed system that reads live, over-the-board (OTB) chess **positions**
> with a camera, as a low-cost alternative to DGT electronic boards that needs no
> special equipment. It streams **timestamped board positions with per-square
> confidences**. It does **not** compute moves, legality or game history: that is
> the job of the application layer, downstream and out of scope.

**Status:** v0.6 · **Date:** 2026-11-24 · **Author:** Felix Peppert (with research synthesis)

This document is the decision record of the system **as it is today**. How the system
behaves in detail lives in [index.md](index.md) and the pages it links. Decisions that
no longer hold are listed once, in [Superseded decisions](#superseded-decisions), with
the commit that still holds their full text.

### Core philosophy (read this first)

**The CV system is a per-frame position estimator, not a chess engine.** For each
input frame it answers one question: *"which pieces are on which squares right now,
and how sure am I?"* It keeps no game state. Specifically, the CV layer:

- **streams positions, not moves.** An illegal arrangement looks exactly like a legal
  one; we report what we see.
- **never applies chess rules.** No legality, no move resolution, no FEN completion,
  no takebacks, no PGN. The application layer owns all of that and may apply
  temporal and legality smoothing to our stream.
- **outputs piece placement only** (FEN field 1, or an 8×8 grid). Side to move,
  castling rights, en passant and move counters cannot be known from one image.
- **emits per-square distributions** and an advisory **stability** signal, so the
  application layer can decide when a position is at rest and reconstruct the move
  sequence itself, including two pieces in motion at once.

Two advisory blocks ride beside the raw read: `prior`, a repair to the nearest
*possible* board, and `settled`, a display debounce (§5). Neither replaces the raw
read, and neither is chess logic.

### Locked decisions

The numbers are the original ones, because code and notes cite them (#14, #15).
Where a decision changed, its text below is today's, and the old one is in
[Superseded decisions](#superseded-decisions).

1. **The CV layer streams positions only.** No moves, legality, game state or PGN.
2. **The output is piece placement plus confidences**, not a full FEN and not a move.
3. **No setup position and no calibration.** The system works when dropped into any
   mid-game position. Adapting to a new set is an offline fine-tune that ships as a new
   ONNX file (D10), never a runtime calibration round.
4. **Stability is advisory, not a gate.** The edge streams a stability score; the
   application layer owns move segmentation. The node's read scheduler uses it as a
   hint, under a keepalive floor (§4.3).
5. **Camera placement:** optimize for the **angled, broadcast-style view** (~45–60°), D8.
6. **The edge is a dumb camera** (D11). It does no board detection and no
   rectification. It crops to a rectangle set from the laptop's `/control` page, and
   the board must fill at least about half of the sent image's width.
7. **Recognition is ChessQ Lite V6** behind the `Recognizer` seam (D10). Its weights
   are PolyForm Noncommercial 1.0.0; the user accepted that licence.
8. **Clock in v1: which side is running**, streamed as sensor state (D7). Time OCR is
   deferred. Not built yet (M6).
9. **Edges in the sim:** the phone browser page `/camera` (TypeScript, no OpenCV) and
   the Python edge CLI. The edge interface stays small enough to port to a real Pi.
10. **No Lichess in v1.** The project ships its own viewer; PGN or Lichess export is an
    optional later adapter.
11. **Transport is WebSocket** for frames in and position events out.
12. **The board is the logical unit;** one edge may host several boards. The POC runs
    one board.
13. **The inference node is Python** (FastAPI, asyncio, ONNX Runtime).
14. **The canonical output is the full 64×13 per-square distribution** (`square_dist`);
    `placement` and `confidence` are derived from it.
15. **The chess frame is the only convention** (reversed 2026-11-24, D10). The
    recognizer finds the board and its orientation on every frame; events, labels,
    truth, eval, replay and the viewer all use the chess frame (rank 8 first, file a
    first, as in FEN).
16. *(Superseded: the DINOv2-S backbone. There is no in-house backbone; see #7.)*

---

## 1. Background

### 1.1 The problem domain

Tournament chess broadcasting relies almost entirely on **DGT electronic boards**:

- **Hardware cost and lock-in.** A DGT board with its mandatory electronic pieces costs
  **~€500–€750 per board** before clock and cabling. Every piece holds a passive LC
  circuit tuned to its own frequency, so ordinary sets do not work.
- **Cabling that scales linearly.** Boards chain on an RS-232/USB bus; a hall needs a
  connection set per ~12 boards, bus boxes and runs back to a host PC.
- **Move-detection edge cases.** Fast moves can be bundled or caught mid-air, and
  transient illegal positions are emitted on every physical change.

### 1.2 What DGT emits, and how our split mirrors it

A DGT board does **not** emit moves either. It emits **per-square occupancy changes**,
and host software (DGT LiveChess) reconstructs the move, checks legality and writes
PGN. We build the sensor, not LiveChess. The two things DGT gets for free and a camera
does not, **per-square identity** and an **unambiguous move-completion signal** (the
clock lever), we answer with per-square distributions plus dense streaming, and with
the which-side-running clock signal.

### 1.3 State of computer-vision chess reading

| Project | Approach | Real-world result |
|---|---|---|
| **Chesscog** (Wölflein & Arandjelović, 2021) | localization → occupancy CNN → piece CNN | 93.9 % whole-board on synthetic, **2.3 %** on real photos |
| **LiveChess2FEN** (Mallasén, 2020) | localize once → 64-square CNN | ~92 % per piece; <1 s/board on a Jetson Nano |
| **End-to-End / ChessReD** (Masouris & van Gemert, 2024) | one ResNeXt predicts the whole board | **15.3 %** perfect boards on real angled photos |
| **ChessQueries** (Seytre, 2026; ChessQ Lite runs Lichess's "From image") | end-to-end ViT, whole image in, 64×13 out | paper: 99.5 % exact boards on ChessReD (ViT-L); ours, below |
| **ChessCam** (Pbatch) | phone in the browser, its own ChessQueries-architecture model | 39/43 held-out blitz games perfect |

For most of the literature, per-frame whole-board recognition on real angled boards was
the weak link. ChessQueries changed that: on our own four-take replay gate ChessQ Lite
V6 reads 92/101 raw boards perfectly, cold, where our eight-month in-house stack read
5/101 ([reports/2026-11-24-chessq-v6-replay-gate.md](reports/2026-11-24-chessq-v6-replay-gate.md)).
Sources: [plans/2026-10-09-chessqueries-handoff.md](plans/2026-10-09-chessqueries-handoff.md),
[plans/2026-11-24-chesscam-camerachess.md](plans/2026-11-24-chesscam-camerachess.md).

**Our split stays as it was:** the CV layer streams per-frame placements with
distributions, and the consumer applies temporal and legality smoothing.

---

## 2. Aim

### 2.1 Goal

**One camera films one board.** A small **edge** crops the frame to the board and
forwards frames at an adaptive rate. A central **inference node** estimates, for each
frame it reads, the board's piece placement with per-square distributions, for **many
boards at once**. The output is a **stream of timestamped positions** that any
downstream application can use to reconstruct moves, legality, PGN or broadcasts.

### 2.2 Objectives

- **Positions, not moves.** A documented stream of timestamped placements, per-square
  distributions and advisory stability. No move, legality or PGN logic.
- **Many boards per node.** The edge forwards adaptively (a low keepalive at rest,
  bursts during motion), and the node reads only what is worth reading (burst ends plus
  a keepalive floor), so compute scales to many boards.
- **Honest output.** We never block on a clean frame; we stream what we read. Raw reads
  stay byte-identical on the wire beside the advisory `prior` and `settled` blocks.
- **Accurate timestamps.** Stamp at **capture time on the edge**, not at server
  receive. Target: tens of ms.
- **No hardware purchase yet.** A local simulation: a phone as the camera and edge, the
  laptop as the inference node, plus a viewer web app.

### 2.3 Non-goals (the application layer's job)

- Move detection or segmentation, including ordering when several pieces move.
- Legality, illegal-position handling, takebacks, game state, PGN, broadcast.
- FEN completion beyond piece placement.
- Emulating the DGT serial/USB protocol.
- Bughouse or Chess960 first.

---

## 3. Key design decisions

### D1 — The CV layer is a stateless per-frame position estimator

Frame in → 8×8 placement + per-square distribution + advisory stability out. No game
state, no legality, no move tracking. This is the load-bearing decision.

Two pieces of node state exist and are allowed because they never change the read:
the prior repair (a pure function of one read) and the `settled` debounce (state over
*published* reads, served beside them). `square_dist` and `placement` on every event
are exactly what the recognizer produced.

### D2 — Output is piece placement plus a full per-square distribution

The canonical output is a **64×13 distribution** over {empty + 12 piece classes}. A
single scalar per square would hide *which* alternatives a square was confused with
(`0.4 N / 0.35 B` and `0.4 empty / 0.4 p` both read as `0.4`), and that structure is
what downstream probabilistic move inference needs. `placement` (FEN field 1) is the
per-square argmax, and `confidence` the mean per-square maximum. We do not fabricate
side to move, castling, en passant or counters.

### D3 — Stability is an advisory signal; the consumer segments moves

Each frame carries `stability ∈ [0,1]` (1.0 = at rest), measured cheaply from
inter-frame motion on the edge, and each event carries it on. It is a **hint**: the
application layer decides when a move completed. Inside the system two parts use it:
the edge's adaptive forwarder and the node's read scheduler. The scheduler reads on a
keepalive floor whatever the meter claims, because a stuck meter once read "moving" all
day (the `board_reader_inference.schedule` module docstring).

### D4 — Adaptive frame forwarding on the edge

The edge forwards a low **keepalive** at rest (~1 fps) and a **burst** during motion
(~12 fps, with hysteresis), so the consumer sees whole transitions without a full
30 fps stream per board. This keeps the many-boards-per-node goal. Each frame carries
its capture time and `seq` (D5). D4's other half, board detection and rectification on
the edge, is superseded by D11 (see [Superseded decisions](#superseded-decisions)).

### D5 — Authoritative timestamp at capture; sync via chrony/NTP

The edge stamps each frame with its capture time (`capture_ts`) and a per-board
monotonic `seq`. Real hardware syncs nodes with **chrony against a LAN NTP server**
(tens of µs; PTP is unavailable on Pi 4 and Jetson Nano hardware). `seq` is the
authoritative order and survives clock steps; the scheduler and forwarder re-anchor
rather than stall when capture time steps backwards.

### D7 — Clock (v1): which side is running, streamed as sensor state

Temporal differencing of two clock regions over ~1.5–2 s gives `running ∈
{white, black, none}`, streamed as a sensor reading. The protocol carries the field
(`clock.running`); no edge measures it yet, so it is always `none` (M6).

### D8 — Optimize for the angled broadcast view

The target is a tripod view at ~45–60°, so back-rank occlusion is expected. ChessQ
reads a square under a hand or arm as a **confident** empty (p ≥ 0.96), so confidence
cannot flag occlusion. Hands are handled in time instead: the node reads at the end of
a motion burst, when the hand is gone (§4.3), and `settled` adopts a change only after
it held across a window (§5). Evidence:
[reports/2026-11-24-settle-tuning.md](reports/2026-11-24-settle-tuning.md) (no hand
frame settles on the gate).

### D9 — Target Jetson Orin Nano; simulate on x86; models in ONNX

The original Jetson Nano is end-of-life, so the reference target is the **Jetson Orin
Nano (Super)**; a **Raspberry Pi 5 + Hailo-8L** is the value alternative. No
DeepStream at this scale. Models are **ONNX**; TensorRT engines would be built only on
the device. The sim is plain `linux/amd64` with ONNX Runtime: the GPU (CUDA EP) on the
laptop, the CPU in `docker compose`. It tests logic and integration, not Jetson speed.
ChessQ V6 costs ~205 ms per read as fp32 on CUDA and ~660 ms as int8 on CPU
([reports/2026-11-24-chessq-gpu-formats.md](reports/2026-11-24-chessq-gpu-formats.md));
Pi or Jetson latency is unmeasured (§9).

### D10 — ChessQ is the core recognizer (ADR, 2026-11-24)

**Status:** accepted (user, 2026-11-24). Plan:
[plans/2026-11-24-simplify-around-chessq.md](plans/2026-11-24-simplify-around-chessq.md).

**Context.** Eight months of in-house recognition (DINOv2 features, exemplar banks,
calibration, multi-style crop classifiers, the black-bishop demotion) never beat
~80 % cross-style piece type zero-shot. On the four-take replay gate the shipped stack
reached 5/101 raw and 35/101 settled perfect boards. ChessQ Lite V6 (ChessQueries'
small model, which Lichess runs in the browser: the whole image squash-resized to
644×644 in, 64×13 logits out) read the same frames cold at 92/101 raw and 84/101
settled perfect, with no crop of ours, no warp, no bank and no calibration
([reports/2026-11-24-chessq-v6-replay-gate.md](reports/2026-11-24-chessq-v6-replay-gate.md)).

**Decision.**
- **One recognizer seam.** `board_reader_inference.recognizer` defines `Recognizer`
  (`name`, `frame`, `model_path`, `read(image) -> (8,8,13)` in the chess frame),
  `StubRecognizer`, `event_from_dist` and `build_from_env()`. ChessQ
  (`board_reader_inference.chessq`) is the implementation `BR_CHESSQ` selects. A
  second model (ChessCam v2, the hedge for licence and Pi/Jetson latency) is one more
  file behind the same seam.
- **The chess frame is the only convention** (locked decision #15).
- **No calibration, no banks, no in-house models, no training harness.** Adapting to a
  new set is an offline LoRA fine-tune that produces a new ONNX file.
- **Licence accepted.** The weights are PolyForm Noncommercial 1.0.0. `make models`
  fetches them at a pinned Hugging Face revision, checks their SHA-256, and exports the
  fp32 file from the official safetensors (`tools/chessq_export`). They are never
  committed. Without them the node serves the stub recognizer, loudly: a startup
  banner and a `fallback` reason under `recognizer` on `/healthz`.
- **One release, two files, chosen by hardware.** The GPU serves our fp32 export
  (`make dev`, `make gate`); the CPU serves the author's int8 file (`make dev-cpu`,
  `compose.yaml`). Both read every gate frame identically
  ([reports/2026-11-24-chessq-gpu-formats.md](reports/2026-11-24-chessq-gpu-formats.md));
  `tests/test_deploy_parity.py` holds the pairing.

### D11 — The edge is a dumb camera; the control UI owns its configuration (2026-11-24)

**Status:** accepted (user, 2026-11-24). Plan: the "Phase C becomes dumb camera"
section of [plans/2026-11-24-simplify-around-chessq.md](plans/2026-11-24-simplify-around-chessq.md).
(Not the plan's own decision D11, which is the licence.)

**Context.** ChessQ finds the board itself, so corner tracking and labelled corners
have nothing left to feed. Framing still matters: the board must fill at least about
half of the sent image's width (at 35 % perfect boards collapse), and an upright
(portrait) frame costs about a quarter of perfect boards
([reports/2026-11-24-chessq-framing.md](reports/2026-11-24-chessq-framing.md)). A phone
held over a board is the worst place to set that up.

**Decision.**
- **The phone only films and streams.** `/camera` is a viewfinder with a status line
  and no controls.
- **The laptop configures it** from `/control`: the crop rectangle drawn on the phone's
  full-frame preview, the lens, the capture resolution, and streaming on or off.
- **The config is pushed to the edge and applied there.** The inference node holds it
  per board, in memory. The phone posts a report every 500 ms
  (`POST /boards/{id}/camera/report`; the reply is the config) and a whole-frame
  preview every 1.5 s. The contract is `board_reader_core.control`.
- **The edge crops, meters and forwards.** It crops to the rectangle, measures motion
  (`stability`) and sharpness on the crop, and forwards adaptively (D4). The L1→L2
  wire (§4.5) is unchanged; frames arrive already cropped.
- **Portrait is reported, not blocked.** `/control` warns.
- **The Python edge** takes a static `--roi x,y,w,h` (the same `CropRect`). Wiring it
  to the control channel is open work (§8).

**Rejected.** Cropping on the node from whole frames: it changes the wire contract and
takes motion measurement off the crop. An operator rectangle on the phone: the phone is
the wrong place for setup. Sending the whole wide frame: ruled out by the framing
measurement.

---

## Superseded decisions

Each entry: what it was, what replaced it, the evidence, and where its full text lives.
`7f56f58` is the last commit before the 2026-11-24 rebuild; `53c1468` is the last
commit whose DESIGN.md carried every section with its dated amendments.

| Was | Replaced by | Evidence | Full text |
|---|---|---|---|
| **D4's detection half / old locked #6 — the edge detects and rectifies the board** (Canny→Hough→RANSAC homography or a 4-corner tap; rectified crops; `/detect`; `core.boarddetect`; the edge rectifier with `--rectify/--corners/--forward-original`) | D11: the edge crops to a rectangle set on `/control`, `--roi` on the Python edge. The detector survives only as the offline QC tool `tools/video/boarddetect.py`. Adaptive forwarding, D4's other half, stays (D4). | [framing report](reports/2026-11-24-chessq-framing.md) | `git show 7f56f58:docs/DESIGN.md` (D4, §4.2, §4.5) |
| **Old locked #15 / §4.4.1 — fixed orientation set at setup; camera-movement handling** (bump vs reposition, the `/camera` plane tracker, re-detection after sustained loss, auto-orientation from printed coordinates, statuses `recalibrating`/`needs_setup`) | Locked decision #15 as rewritten: the recognizer resolves orientation per frame, chess frame everywhere. A moved camera needs only its rectangle redrawn on `/control`. The two statuses stay reserved in the protocol and are never emitted. | [V6 gate](reports/2026-11-24-chessq-v6-replay-gate.md); [framing report](reports/2026-11-24-chessq-framing.md) (zero rotation errors) | `git show 7f56f58:docs/DESIGN.md` (§4.4.1) |
| **D6 / locked #7, #16 / §4.6 — DINOv2-S embeddings + an offline-trained head; DINOv2 over DINOv3 on licence** | D10: ChessQ Lite V6, end to end, behind the `Recognizer` seam. | [V6 gate](reports/2026-11-24-chessq-v6-replay-gate.md); [backbone ceiling and DINOv3 probe](reports/2026-08-31-dinov3-probe.md) | `git show 7f56f58:docs/DESIGN.md` (D6, §4.6) |
| **Locked #3 / M5 / WP-E — optional calibration round** (exemplar bank, centroids, `/calibrate`, start-position auto-calibration, the occupancy bank, `BR_SESSION_BANK`/`BR_OCCUPANCY_BANK`/`BR_CENTROIDS`) | No calibration (locked decision #3); adaptation is an offline LoRA → new ONNX. Bank writes were measured non-monotone. | [calibrate before/after](reports/2026-08-31-calibrate-before-after.md); [bank rematch](reports/2026-08-29-bank-rematch.md) | `git show 7f56f58:docs/DESIGN.md` (§4.6), `git show 7f56f58:docs/calibration-and-the-bank.md` |
| **The multi-style crop classifier and its composites** (`BR_MODEL`, `BR_MODEL_B` cascade, ensembles with the bank, `tools/train`, RunPod) | D10: no in-house models, no training harness (plan decision D7: deleted). | [retrain program](reports/2026-08-28-retrain-program.md); [V6 gate](reports/2026-11-24-chessq-v6-replay-gate.md) | `git show 7f56f58:docs/recognition-pipeline.md`, `git show 7f56f58:docs/training-and-data.md` |
| **Square cropping and occlusion masking** (rectified tiles or perspective full-piece boxes with ~1.6 square-widths headroom; `BR_CROP_MASK=gray`) | The recognizer reads the whole sent image; nothing of ours crops squares. | [V6 gate](reports/2026-11-24-chessq-v6-replay-gate.md) | `git show 7f56f58:docs/recognition-pipeline.md` ("Cropping is where the accuracy is") |
| **Black-bishop (attractor) demotion** (`demote_attractor`, `trial.AttractorGuard`) | ChessQ's softmax is served unchanged. | [evidence rerank](reports/2026-08-27-evidence-rerank.md) (why it existed) | `git show 7f56f58:docs/recognition-pipeline.md` ("The black-bishop demotion") |
| **Autorotate by chess plausibility** (`?autorotate=1`: fewest pawns on ranks 1/8) | The recognizer resolves orientation (locked decision #15). | [V6 gate](reports/2026-11-24-chessq-v6-replay-gate.md) | `git show 7f56f58:docs/DESIGN.md` (§4.6) |
| **D8's "occluded squares get low confidence"; the optional burst voter** (§4.2, §4.4; the voter was never built) | D8 as rewritten: burst-end reads plus the `settled` debounce. | [settle tuning](reports/2026-11-24-settle-tuning.md) | `git show 7f56f58:docs/DESIGN.md` (D8, §4.4) |
| **Corner-carrying frames** (`corners` on every wire frame; `POST /estimate?corners=`) | `corners` is a legacy wire field: still parsed so old recordings replay, never sent, ignored. `/estimate` takes the photo alone. | plan, Phase B–C | `git show 7f56f58:docs/DESIGN.md` (§4.5) |
| **The ~2.5 s / 2-read settle window and the one-read `burst-end` schedule** | 1.0 s / 2 reads / 2 stale windows, with `burst-end:burst_end_reads=3,spacing_s=0.6`. | [settle tuning](reports/2026-11-24-settle-tuning.md) | `git show 53c1468:docs/DESIGN.md` (§5) |
| **Old §7 milestones M1/M5 and §8 work packages WP-A, WP-B, WP-E as written for the DINOv2 pipeline** | §7 and §8 below, with a status per item. | — | `git show 7f56f58:docs/DESIGN.md` (§7, §8) |

---

## 4. Architecture

### 4.1 Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L0  CAPTURE        one camera per board (sim: a phone)                     │
├──────────────────────────────────────────────────────────────────────────┤
│ L1  EDGE           phone /camera page, or the Python edge CLI              │
│     • crop to the configured rectangle                                     │
│     • motion → stability; sharpness                                        │
│     • adaptive forwarding (keepalive + bursts)                             │
│     • capture timestamp + seq; JPEG                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ TRANSPORT          WebSocket: frames in │ position events out               │
│                    HTTP: camera control channel, snapshot, labels           │
├──────────────────────────────────────────────────────────────────────────┤
│ L2  INFERENCE      ingest → read scheduler → Recognizer (ChessQ)           │
│     (stateless     → prior repair (advisory) → settled debounce (advisory) │
│      estimator)    → hub (latest per board) + recorder                     │
├──────────────────────────────────────────────────────────────────────────┤
│ L3  POSITION STREAM  timestamped {placement, square_dist, stability, …}    │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  out of scope — downstream consumers
  APPLICATION LAYER: move reconstruction · legality · game state · PGN · broadcast.
  The POC ships only a viewer that draws the latest board.
```

### 4.2 Packages (deployment boundaries)

| Package | Role | Main modules |
|---|---|---|
| `board-reader-core` | the shared contract | `protocol` (L3 events, §5), `wire` (L1→L2 frames, §4.5), `control` (camera config and report), `recording` (session directories), `sim` (synthetic renders, §6) |
| `board-reader-edge` | the Python edge ("Pi") | `source` (camera, video, images, synthetic game, recording), `motion` (stability), `forwarder` (adaptive rate), `client` (WebSocket), `adapter` (pacing), `cli` (`feed`) |
| `board-reader-inference` | the inference node ("Jetson") | `server` (FastAPI gateway), `ingest` (inbox + workers), `schedule` (read policy), `recognizer` + `chessq` (the read), `prior` (repair), `stabilize` (debounce), `hub`, `camera_control`, `replay` / `evaluate` / `trial` (measurement), `labels`, `calibration` (temperature scaling for eval only) |
| `board-reader-viewer` | serves the web UI and relays the node's API under `/api` and its sockets under `/ws` | `server`, `proxy`, `tls`; pages `/`, `/camera`, `/control`, `/snapshot` |

**Separation rule.** `edge` and `inference` never import each other; they meet only on
the wire. `core` imports neither. `tests/test_layer_separation.py` enforces it.

### 4.3 Data flow for one frame

1. The edge captures a frame, crops it to the configured rectangle, and measures
   `stability` and `sharpness` on the crop.
2. The forwarder decides whether to send (always during a burst, at the keepalive rate
   at rest), JPEG-encodes the crop and stamps `capture_ts` and `seq`.
3. The node's `/ingest` handler acks the frame ("received", the edge's flow control),
   lets the read scheduler observe its header, and puts it in the board's
   **latest-wins inbox**. Of two waiting frames the inbox keeps the sharper one.
4. The board's worker takes the frame and asks the scheduler whether it is due. The
   default policy is `burst-end:burst_end_reads=3,spacing_s=0.6`: three reads 0.6 s
   apart at the end of a motion burst, and a read whenever 3 s of capture time passed
   without one (the keepalive floor). A wall-clock backstop covers a frozen
   `capture_ts`. `BR_SCHEDULE` overrides the policy.
5. A due frame goes to `Recognizer.read` in a worker thread, giving a chess-frame
   `square_dist`, then a `PositionEvent`.
6. `prior.decode` attaches the repaired board (`prior`); `BoardStabilizer` attaches the
   debounced board (`settled`).
7. The hub publishes the event to `/viewer` subscribers and keeps the latest per board;
   the recorder writes every ingested frame to a session directory for replay.

There is deliberately **no** "find the legal move" or "commit a move" step.

### 4.4 Failure behavior

- **Hand over the board.** Covered squares read as confident empties. The scheduler
  skips the burst and reads at its end; `settled` holds the previous board until a
  change holds for its window. The raw stream still carries every read.
- **Camera moved.** Nothing on the edge follows the board. The operator redraws the
  rectangle on `/control`; reads in between are whatever the recognizer sees.
- **Edge offline.** The stream goes quiet; the viewer marks the board stale on its side.
  The protocol's `stale` status is reserved, not emitted.
- **Low confidence.** Emitted verbatim; the consumer decides.
- **Missing weights.** The stub recognizer serves, loudly (D10).
- **Impossible board** (two white kings, a pawn on rank 8). Reported as read, status
  `ok`; `prior` carries the nearest possible board beside it.

### 4.5 The inference node's interfaces (gateway, wire, hub)

**Ingress (WebSocket `/ingest`).** One message per frame: a JSON header (`board_id`,
`seq`, `capture_ts`, `stability`, optional `sharpness`, `clock`, `burst_id`) followed by
the JPEG (`board_reader_core.wire`). The header may carry the legacy `corners`, which
is parsed and ignored. The `control` message kind is reserved; nothing uses it.

**Egress (WebSocket `/viewer`).** Position events as JSON (§5), the latest per board
on connect.

**HTTP.**

| Route | Purpose |
|---|---|
| `GET /healthz` | liveness, ingest counters, settle values, recording stats, the serving recognizer (and `fallback` when the stub serves) |
| `GET /boards/{id}` · `GET /boards/{id}/frame` | latest event · latest ingested frame |
| `POST /estimate` | one photo (JPEG body) → its event, synchronously; also published to the hub. No `settled` block. |
| `POST /labels/{upload}` · `GET /uploads` · `GET /uploads/{upload}/photo` | confirm a retained photo's position as ground truth; list and fetch retained photos (`BR_SAVE_UPLOADS`) |
| `GET`/`PUT /boards/{id}/camera` | the camera control state and config (D11) |
| `POST /boards/{id}/camera/report` | the phone's report; the reply is its config |
| `POST`/`GET /boards/{id}/camera/preview` | the phone's whole-frame preview for `/control` |

---

## 5. Position stream protocol (L3 output)

Contract: `board_reader_core.protocol`. One event per read:

```json
{
  "board_id": "A1",
  "utc": "2026-06-15T14:22:09.470Z",
  "seq": 12831,
  "orientation": "a1_bottom_left",
  "classes": ["empty","P","N","B","R","Q","K","p","n","b","r","q","k"],
  "placement": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
  "square_dist": [[[0.01,0,0,0,0.99,0,0,0,0,0,0,0,0], "…8 squares a8..h8…"], "…8 rows…"],
  "confidence": 0.93,
  "stability": 0.12,
  "clock": { "running": "none" },
  "status": "ok",
  "prior":   { "placement": "…", "flips": [], "residual": {} },
  "settled": { "placement": "…", "pending": [] }
}
```

- `utc` — the edge **capture** time, RFC 3339 with milliseconds.
- `seq` — per-board monotonic; authoritative for ordering.
- `orientation` — always `a1_bottom_left`, reserved. The placement is in the chess
  frame (rank 8 first, file a first, as in FEN), resolved per frame by the recognizer.
- `classes` — the 13-class index order of every per-square vector; uppercase = White,
  `empty` is index 0.
- `square_dist` — **canonical output**: 8×8 13-vectors, rank 8 → rank 1, file a → h.
  Everything else is derived from it.
- `placement` — FEN field 1, the per-square argmax.
- `confidence` — the mean per-square maximum probability.
- `stability` — `[0,1]`, 1 = at rest; advisory.
- `clock.running` — `white|black|none`; always `none` until M6.
- `status` — `ok|recalibrating|needs_setup|stale`. Only `ok` is emitted; the other three
  are reserved. There is **no `illegal`**: an illegal-looking board is `ok`.
- `prior` (optional) — the hard-constraint decoder's repair, served **beside** the raw
  fields: `{placement, flips: [{square, from, to, cost, constraint}], residual}`. The
  constraints are necessary conditions on the piece multiset (≤1 king per side, ≤16
  pieces per side, the promotion budget, no back-rank pawns when the frame is passed
  in). They are not legality. The node currently passes no frame, so the back-rank
  constraint is skipped. A consumer that smooths the raw distribution ignores this
  block and cannot double-count it.
- `settled` (optional, streamed frames only) — a display debounce of the repaired reads:
  `{placement, pending: [squares reading differently, not yet adopted]}`. A square
  changes only after it held the same class on every read across **1.0 s** and **≥2
  reads**. A settlement is committed only if it does not make the board less possible
  than it already is; a square blocked that way gives up after **2 windows** and the
  contested set falls through to the current read. The window is time, not a read
  count, because the read rate varies with load. Knobs: `BR_SETTLE_SECONDS`,
  `BR_SETTLE_MIN_READS`, `BR_SETTLE_STALE_WINDOWS`; `/healthz` reports them. Evidence:
  [reports/2026-11-24-settle-tuning.md](reports/2026-11-24-settle-tuning.md) (gate:
  104/119 settled perfect, 110/119 raw, median move-to-settle 3.0 s). This is not the
  downstream smoother (D3): it uses no move model and no legality. The live viewer
  draws `settled.placement`, and `prior.placement` where there is no `settled`.

Nothing has been built on the far side of this socket yet: the viewer and the headless
`board-reader-inference positions` tail render or capture the stream, and no
move-aware temporal consistency exists anywhere ([architecture.md](architecture.md),
"Temporal consistency").

---

## 6. Local simulation

| Real component | Simulated as |
|---|---|
| Camera + Pi edge | a **phone** on `/camera` (https, so `getUserMedia` works), or the **Python edge** `board-reader-edge feed` from a video, an image folder, a recording or a synthetic game |
| Jetson | the laptop (`make dev`, CUDA) or a `linux/amd64` container (`docker compose up`, CPU) |
| Transport | WebSocket on localhost/LAN; the viewer relays everything, so one origin serves all pages |
| Viewer | `/` (live board), `/control` (camera setup), `/snapshot` (one photo → board → correct → label) |
| Clock | not simulated yet (M6) |

**What the synthetic game is for now.** `board_reader_core.sim` renders letter boards
and a synthetic game with hand-occluded transitions. ChessQ cannot read these renders
(0/14 perfect, 79 % per square, measured 2026-11-24), so `docker compose up` shows the
pipe working (forwarding, bursts, scheduling, the stream reaching the viewer), not
recognition. The tests read sim renders with a test-side template matcher
(`tests/_sim_recognizer.py`) so the end-to-end tests check real pixels without the
model. Recognition is demoed and measured on real frames: the phone, recordings
(`replay`, `make gate`) and labeled stills (`eval`).

### 6.1 Bring-up and test data

1. **File-based edge first.** `board-reader-edge feed` streams a video, an image
   folder, a recording or the synthetic game, so the whole pipe runs without a camera.
2. **Then a real board:** a phone on `/camera`, configured from `/control`, with the
   laptop as the inference node.

Own recordings are the source of truth (`recordings/<take>/` with a
`truth.json`, scored by `eval-session`); [youtube-streams.md](youtube-streams.md) lists
curated public streams for extra footage.

---

## 7. Milestones

Build order (CLAUDE.md principle 3). Status as of 2026-11-24.

| Milestone | Scope | Status |
|---|---|---|
| **M0 — Walking skeleton** | gateway, protocol, compose, file-based edge, stub recognizer, viewer | done |
| **M1 — Single-frame estimation** | one photo → placement + distributions; accuracy on real photos | done: `/estimate`, `/snapshot`, `eval` (ChessQ) |
| **M2 — Live stream over a full game** ★ | adaptive forwarding, stability, a continuous stream through hands and transitions | done on recorded takes (the replay gate). The live take with the dumb-camera phone is next. |
| **M3 — Transport and timestamps** | capture stamps, `seq`, retained latest, recorder, status events, time sync | mostly done: stamps, `seq`, retained latest, recorder. Not done: emitted status events, chrony/NTP. |
| **M4 — Multi-board on one node** | N edges → one node; boards per node | partly: one worker per board sharing one recognizer. Not measured. |
| **M5 — Offline adaptation** (optional) | a LoRA fine-tune of the recognizer on one set's labeled frames → new ONNX, gated on the replay gate | not started. Replaces the old calibration milestone. |
| **M6 — Clock** | which side is running, streamed as sensor state | not started; the field is carried as `none` |
| **M7 — Hardening and hardware port** | robustness; ONNX→TensorRT on an Orin Nano or Pi 5 + Hailo; throughput | not started |

---

## 8. Work packages and roadmap

### 8.1 Work packages

A decomposition by component, not a sequence (CLAUDE.md principle 3). The IDs are cited
in code and tests, so they stay; the status is as of 2026-11-24.

| WP | Package | Status |
|---|---|---|
| **A** Edge framing | A1–A3 (board detection, rectification, re-detection on drift) | **obsolete** (D11). The edge crops to a configured rectangle; the detector survives only in `tools/video/boarddetect.py`. |
| **B** Recognition | B1 square cropping, B2 DINOv2 + head | **obsolete**, replaced by the `Recognizer` seam and ChessQ (D10) |
| | B3 per-square + position confidence | done (`confidence`); "honest occlusion flags" obsolete (D8) |
| | B4 burst voting; eval harness | burst voting never built (burst-end reads + `settled` instead); eval done (`eval`, `eval-session`, `make gate`) |
| **C** Edge capture and forwarding | C1 capture (getUserMedia, video, images, recording), C2 motion → stability, C3 adaptive forwarder, C4 capture stamps + JPEG + publisher | done, in the Python edge and in `/camera` |
| **D** Transport and stream | D1 WebSocket ingress with backpressure, D2 egress with retained latest, D3 event schema + validation | done (WebSocket only, no SSE; no schema versioning) |
| | D4 time sync (chrony/NTP) | not started |
| **E** Calibration | E1–E3 | **removed** (D10) |
| **F** Clock | F1 clock region, F2 which side runs, F3 digit OCR (later) | not started (M6) |
| **G** Node orchestration | G1 gateway + per-board inbox + pre-filter, G2 shared estimator worker, G3 Docker image | done (latest-wins inbox with sharpness preference and the read scheduler; one worker per board on one recognizer; amd64 image, CPU in compose) |
| | G4 scaling tests (boards per node, latency budget) | not started (M4) |
| **H** Consumers | H1 viewer, H2 recorder + replay | done |
| | H3 reference move reconstructor + PGN/Lichess export (optional, application layer) | not built; `/snapshot` assembles Lichess links in the page |
| **I** Cross-cutting | I1 sim harness, I3 metrics (`/healthz`, eval, gate), I4 docs and runbook | done, maintained |
| | I2 datasets (own recordings with truth, [youtube-streams.md](youtube-streams.md)) | ongoing; the binding gap (§9) |

### 8.2 Roadmap

What is open now, in rough order:

1. **The live take** with the dumb-camera phone, configured from `/control`
   ([TESTING.md](TESTING.md), Option J). It validates D11; replay cannot, because crop
   and stability are baked into recordings.
2. **More varied recordings:** a second physical set, a second camera, other lighting,
   mid-game positions. The gate is four takes of one set (§9).
3. **The Python edge on the control channel:** today it takes a static `--roi`.
4. **The ChessCam v2 hedge:** a second recognizer behind the seam, for the licence risk
   and for Pi/Jetson latency.
5. **Fully on-phone app** (idea): run recognition in the phone's browser. See
   [ideas.md](ideas.md).
6. Then the milestones in order: M4 measurement, M6 clock, M5 offline adaptation, M7
   hardware port.

---

## 9. Risks and open questions

**Risks**

- **One third-party, noncommercial model (D10).** The weights are PolyForm
  Noncommercial 1.0.0 from one upstream. Mitigation: pinned revision, SHA-256 check,
  a local copy; ChessCam v2 behind the same seam as the hedge. No commercial use.
- **Hands read as confident empties (D8).** ChessQ gives p ≥ 0.96 to an empty under an
  arm, so confidence cannot flag occlusion. The burst-end scheduler and the `settled`
  debounce absorb it on the gate (no hand frame settles). A shorter settle window or a
  stuck motion meter would let hands through; the 08-13 take settling all its boards is
  the guard.
- **Framing (D11).** The board must fill ≥ ~50 % of the sent width; 35 % collapses
  perfect boards and portrait costs about a quarter of them. Wide framing was measured
  with synthetic padding only, and real distractors (people, other boards) are untested.
- **One physical set in the gate.** All four gate takes film the same set and camera,
  so the settle defaults and the accuracy numbers may not carry over.
- **Orientation is inferred by the model.** A misread rotation would show as a whole
  board wrong at once. None was seen (zero rotation errors in the framing run); the
  rotation diagnostics in `eval` and `eval-session` are the guard.
- **Jetson/Pi latency is unmeasured.** ~205 ms per read on a GTX 1650 (fp32, CUDA) and
  ~660 ms on the laptop CPU (int8). A Pi CPU would be far slower; ChessCam's smaller
  model is the fallback.

**Open questions**

- **Confidence calibration.** Downstream smoothing wants calibrated per-square
  probabilities. On the 47 labeled stills ChessQ's per-square reliability is close
  (ECE 0.002), but nearly every square claims ≥ 0.9, and the mean board confidence is
  1.000 on perfect and imperfect boards alike, so confidence does not single out a bad
  board. `calibration.py` fits a temperature for eval reports only; whether a
  calibrated distribution goes on the wire is undecided.
- **Does `stability` earn its keep for consumers?** Internally it drives the forwarder
  and the scheduler.
- **The back-rank-pawn constraint.** The chess frame is now always known, so `prior`
  could pass it; today it does not.
- **Clock as its own topic?** `clock.running` rides inside each event; a separate topic
  at the clock's ~1–2 Hz may fit better in v2.
