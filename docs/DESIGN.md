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

Each event carries one board, `placement`: the settled board, a display debounce over
the reads (§5), guarded so that it never becomes more impossible than it is (D14). The
read itself stays on the event as its distribution, so the raw board is always
recoverable (D15). The debounce is not legality.

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
the weak link. ChessQueries changed that: on our own replay gate, then four takes,
ChessQ Lite V6 read 92/101 raw boards perfectly, cold, where our eight-month in-house
stack read 5/101 ([reports/2026-11-24-chessq-v6-replay-gate.md](reports/2026-11-24-chessq-v6-replay-gate.md)). The gate now has seven takes
(base `guard-only`: 341/361 raw and 314/361 settled perfect).
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
  stay byte-identical on the wire as each event's `square_dist`, beside the settled
  `placement`.
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

One piece of node state exists and is allowed because it never changes the read:
the settle debounce (state over *published* reads). It decides the event's
`placement`; `square_dist` and `confidence` on every event are exactly what the
recognizer produced, so the read's own board is the argmax of `square_dist` (D15).

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
day (the `board_reader_inference.stream.schedule` module docstring).

### D4 — Adaptive frame forwarding on the edge

The edge forwards a low **keepalive** at rest (~1 fps) and a **burst** during motion
(~12 fps, with hysteresis), so the consumer sees whole transitions without a full
30 fps stream per board. This keeps the many-boards-per-node goal. After a burst ends
both forwarders (the Python edge and the phone's `motion.ts`) keep the burst rate for
a **tail** of `tail_s` = 1 s, as rest frames (`burst_id` null), because the node reads
the first calm frames after a move and would otherwise wait ~1 s for each keepalive
frame ([latency round 2](reports/2026-11-24-latency-round2.md)). Each frame carries
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
a motion burst, when the hand is gone (§4.3), and the settled `placement` adopts a
change only after it held across a window (§5). Evidence:
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
~80 % cross-style piece type zero-shot. On the replay gate, then four takes, the shipped stack
reached 5/101 raw and 35/101 settled perfect boards. ChessQ Lite V6 (ChessQueries'
small model, which Lichess runs in the browser: the whole image squash-resized to
644×644 in, 64×13 logits out) read the same frames cold at 92/101 raw and 84/101
settled perfect, with no crop of ours, no warp, no bank and no calibration
([reports/2026-11-24-chessq-v6-replay-gate.md](reports/2026-11-24-chessq-v6-replay-gate.md)).

**Decision.**
- **One recognizer seam.** `board_reader_inference.recognition.recognizer` defines `Recognizer`
  (`name`, `model_path`, `read(image) -> Recognition`: the (8,8,13) distribution and the board's direction in the chess frame),
  `StubRecognizer`, `event_from_dist` and `build_from_env()`. ChessQ
  (`board_reader_inference.recognition.chessq`) is the implementation `BR_CHESSQ` selects. A
  second model (ChessCam v2, the hedge for licence and Pi/Jetson latency) is one more
  file behind the same seam.
- **The chess frame is the only convention** (locked decision #15).
- **The board's direction rides beside each read, and guards the stream**
  (2026-11-24). The release's second output, `direction` (1,2), is the unit vector
  from e1 toward e8 in the image as (cos, sin), cos up, sin right.
  `Recognizer.read` returns a `Recognition`: the distribution and that bearing in
  degrees clockwise from up (0 = White at the bottom, 90 = White on the left; the stub
  reports none). It never enters the `PositionEvent`. On every gate take it read
  87–91°, within ±2° inside a take. The **orientation guard**
  (`ingest.OrientationGuard`) keeps each board's circular median of its last 9
  published directions and drops a read more than **45°** off it, before the
  debounce: such a read is almost certainly the board read turned. A
  seq that does not advance starts the guard afresh; 5 drops in a row re-anchor it on
  the new direction (the camera really turned). It runs in `ingest._read_and_settle`,
  which the live worker and replay share, so the gate counts the same drops
  (`/healthz` `ingest.misoriented`, replay `frames_misoriented`, a WARNING per drop).
  Not on `POST /estimate`: a snapshot may be taken from any side. The gate is
  unchanged with it (0 reads dropped, every event line identical to `sp03-settle05`).
  Each live published read's direction goes to the camera registry, which tells
  `/control` which side White sits on (D11).
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
  per board and is its single source of truth. It persists each board's config to
  `BR_CAMERA_CONFIG` (`serve` sets `data/camera-config.json`; empty keeps it in memory
  only), so a restart keeps the framing. The contract is `board_reader_core.control`.
- **The channel is two WebSockets per board, and nothing polls** (2026-11-24).
  `WS /boards/{id}/camera` is the phone's: the node pushes the config on connect and
  on every change, and says `previews` on or off; the phone sends its report on
  connect and whenever it changes, and a whole-frame preview JPEG (binary) every
  1.5 s, only while previews are on. `WS /boards/{id}/control` is each control page's:
  the node pushes the state (config, last report, phone connected) on connect and on
  every change, the latest preview and received frame, and a health tick once a
  second, which carries the board's direction (D10) as the side White sits on and
  `camera_moved` when the settled direction moved more than 15°; the page sends `configure` with the whole config. Previews are on exactly
  while at least one control page is connected. Each socket's outbox keeps only the
  latest message of each type, so a slow client can make the node buffer nothing.
- **The edge crops, meters and forwards.** It crops to the rectangle, measures motion
  (`stability`) and sharpness on the crop, and forwards adaptively (D4). The L1→L2
  wire (§4.5) is unchanged; frames arrive already cropped.
- **Portrait is reported, not blocked.** `/control` warns.
- **The Python edge** takes a static `--roi x,y,w,h` (the same `CropRect`). Wiring it
  to the control channel is open work (§8).

**Rejected.** Cropping on the node from whole frames: it changes the wire contract and
takes motion measurement off the crop. An operator rectangle on the phone: the phone is
the wrong place for setup. Sending the whole wide frame: ruled out by the framing
measurement. HTTP polling, the first transport (the phone posted its report every
500 ms and a preview every 1.5 s; `/control` fetched the state, `/healthz` and the
preview every second): about eight requests a second, each logged twice (viewer and
node), with up to half a second before a change was heard, and previews uploaded with
nobody watching. Carrying the control messages on `/ingest`: `/ingest` is not
board-bound (each header names its board), it is open only while streaming, its
reader pairs every text header with the next binary JPEG, and recording and replay sit
on it; a control message there would change the framing every edge and every
recording relies on, for one socket fewer on the phone.

### D12 — The viewer is its own deployment unit, in Node (2026-11-24)

**Status:** accepted (user, 2026-11-24).

**Decision.** The viewer is its own deployment unit in Node/TypeScript (SvelteKit on
adapter-node). SvelteKit has no WebSocket upgrade, so a short custom server
(`server/main.ts`) relays the sockets and serves TLS; all HTTP stays in SvelteKit
routes. A static build (adapter-static, `strict: false`) stays possible for the
on-phone app.

**Rejected.** Serving the UI from the inference node: it would tie the UI to the
Jetson.

### D13 — The wire contract is written once, in Python; TypeScript is generated (2026-11-24)

**Status:** accepted (user, 2026-11-24).

**Context.** Every JSON message between the node and the browser was written twice:
as Python dataclasses with hand parsers in `board_reader_core`, and as TypeScript
types with hand parsers in the viewer. The two drifted in what they accepted (the
TypeScript parsers defaulted fields the Python ones required, and the other way round).

**Decision.**
- **Python is the source of truth.** Every message that crosses to the browser is a
  strict pydantic dataclass in `board_reader_core` (`protocol`, `control`, `wire`,
  built with `wiremodel`): strict scalar types, finite numbers, unknown keys refused,
  ranges as field constraints. Validation lives in the models; the hand parsers are
  gone. The models stay dataclasses because the nodes build them positionally and
  compare them.
- **The JSON Schema is exported** by `board_reader_core.schema` (`python -m
  board_reader_core.schema`): one document, a `$defs` entry per message and per union
  each socket carries, in serialization mode with defaults required (what the node
  sends, every key present).
- **TypeScript is generated from it**, types and runtime validators, and a test fails
  when the generated file is stale.
- **A rule no JSON Schema keyword can state** is a validator in the model plus an
  `x-check` keyword naming it (`isPlacement`, `cropInside`); the TypeScript side
  implements the named check, and a property test runs both on the same inputs.

**Rejected.** Writing the TypeScript first or keeping two hand-written sides: drift is
then found at runtime. A pydantic `BaseModel`: it takes no positional arguments, which
the nodes use throughout.

### D14 — No per-read repair; the settle guard plays standard chess unless a board says `free` (2026-11-24)

**Status:** accepted (user, 2026-11-24).

**Context.** The hard-constraint decoder (`stream/prior.py`) repaired an impossible
read to the nearest possible one, rode the wire as `PositionEvent.prior` and fed the
debounce. Under ChessQ it repaired 2 of 446 gate reads and changed one settled board
in 361; the debounce's guard alone kept every impossible board off the settled
stream ([report](reports/2026-11-24-prior-backrank.md)).

**Decision.**
- **The decoder and the `prior` block are gone.** The stabilizer settles the raw read.
- **The settle guard stays** (`stabilize._excess`, conditions in `stream/rules.py`): a
  commit may not make the settled board more impossible. The conditions hold in any
  position of a standard game, whoever is to move: ≤1 king per colour (0 is fine, a
  hand hides kings), ≤16 pieces per colour, surplus officers plus pawns ≤8 per colour.
  Extra checks (both kings in check, three or more checkers, a pawn on rank 1/8) were
  measured and left out: no ChessQ read or settle on the gate takes breaks one.
- **`rules: "standard" | "free"` per board** in the camera config, set on `/control`,
  persisted with the rest (`BR_CAMERA_CONFIG`; an older file loads as `standard`).
  `free` turns the guard off for a board with extra material or a variant. The node
  reads the setting before every read; each `events.ndjson` line records it, and
  `replay --rules` reproduces it.
- **python-chess** (GPL-3.0-or-later) builds the board the conditions are counted on;
  see [THIRD-PARTY.md](../THIRD-PARTY.md) §5.

**Rejected.** Keeping the decoder for `/estimate`: a snapshot then shows a board the
recognizer did not read, and the snapshot page's own material table already flags an
impossible count. `chess.Board.status()` alone for the counts: its `TOO_MANY_KINGS`
lets two white kings pass and it has no promotion budget.

### D15 — An event carries one board: the settled `placement` (2026-11-24)

**Status:** accepted (user, 2026-11-24).

**Context.** An event carried the read as `placement` and the debounced board in an
advisory `settled` block beside it, so every consumer had to choose a board and the
viewer drew `settled.placement ?? placement`. The read is already on the event as its
distribution.

**Decision.** `placement` is the settled board; the `settled` block and its model
(`SettledBoard`) are gone. A top-level `pending` names the squares where this read's
argmax disagrees with it. `square_dist` and `confidence` stay this read's raw
evidence, so the raw board is `argmax(square_dist)` (`PositionEvent.raw_placement`)
and nothing is lost. `PositionEvent.from_square_dist` takes the settled board as an
optional `placement` and `with_placement` swaps one in; both derive `pending`, and
`validate()` checks it, so `placement`, `pending` and `square_dist` cannot drift.
`/estimate` publishes the read itself with no pending squares. `changed_utc` says
when the settled board last changed (`BoardStabilizer.changed_ts`, back-dated by
pixels over the frames the node received but did not read, `stream/backdate.py`), so
move times are recoverable from the stream. `events.ndjson` keeps
the read as the top row of each square's `evidence`, and `eval-session` still reads
the older lines that carry `settled`.

**Rejected.** Keeping both boards on the wire: two placements invite a consumer to
pick the wrong one, and the raw one is redundant with `square_dist`.

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
| **D8's "occluded squares get low confidence"; the optional burst voter** (§4.2, §4.4; the voter was never built) | D8 as rewritten: burst-end reads plus the settle debounce. | [settle tuning](reports/2026-11-24-settle-tuning.md) | `git show 7f56f58:docs/DESIGN.md` (D8, §4.4) |
| **Corner-carrying frames** (`corners` on every wire frame; `POST /estimate?corners=`) | `corners` is a legacy wire field: still parsed so old recordings replay, never sent, ignored. `/estimate` takes the photo alone. | plan, Phase B–C | `git show 7f56f58:docs/DESIGN.md` (§4.5) |
| **The ~2.5 s / 2-read settle window and the one-read `burst-end` schedule; then 1.0 s with three reads 0.6 s apart** | 0.5 s / 2 reads / 2 stale windows, with `burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3` and the forwarder tail (D4). | [settle tuning](reports/2026-11-24-settle-tuning.md); [live latency](reports/2026-11-24-live-latency.md); [latency round 2](reports/2026-11-24-latency-round2.md) | `git show 53c1468:docs/DESIGN.md` (§5) |
| **The hard-constraint prior decoder** (`stream/prior.py`, `PositionEvent.prior`, `/estimate`'s repair, `eval --no-prior`, the trial's `prior` stream) | D14: the settle guard alone, with a per-board `rules` switch. | [prior report](reports/2026-11-24-prior-backrank.md) | `git show f5cc37e:packages/inference/src/board_reader_inference/stream/prior.py` |
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
│     (stateless     → settled debounce + its guard (advisory)               │
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
| `board-reader-core` | the shared contract | `protocol` (L3 events, §5), `wire` (L1→L2 frames, §4.5), `control` (camera config and report), `wiremodel` (how those are modelled) and `schema` (their JSON Schema, D13), `recording` (session directories), `sim` (synthetic renders, §6) |
| `board-reader-edge` | the Python edge ("Pi") | `source` (camera, video, images, synthetic game, recording), `motion` (stability), `forwarder` (adaptive rate), `client` (WebSocket), `adapter` (pacing), `cli` (`feed`) |
| `board-reader-inference` | the inference node ("Jetson") | at the root `server` (FastAPI gateway), `cli`, `positions` (headless `/viewer` client); `recognition/`: `recognizer` + `chessq` + `onnxrt` (the read); `stream/`: `ingest` (inbox + workers), `schedule` (read policy), `stabilize` (debounce), `rules` (its guard's conditions), `hub`; `control/`: `camera` (the camera control channel); `measurement/`: `replay` / `evaluate` / `trial`, `labels`, `calibration` (temperature scaling for eval only) |
| `packages/viewer` (Node) | the viewer: a SvelteKit app whose pages are prerendered, with `/api/*` and `/api/client-log` as routes, served by `server/main.ts`, which adds the `/ws/*` relay and the TLS port (D12) | `src/routes/api`, `server/{main,ws,tls}.ts`; pages `/`, `/camera`, `/control`, `/snapshot` |

**Separation rule.** `edge` and `inference` never import each other; they meet only on
the wire. `core` imports neither. `tests/test_layer_separation.py` enforces it.

### 4.3 Data flow for one frame

1. The edge captures a frame, crops it to the configured rectangle, and measures
   `stability` and `sharpness` on the crop.
2. The forwarder decides whether to send (always during a burst and for the 1 s tail
   after it, at the keepalive rate at rest), JPEG-encodes the crop and stamps
   `capture_ts` and `seq`.
3. The node's `/ingest` handler acks the frame ("received", the edge's flow control),
   lets the read scheduler observe its header, sends a motion update to `/viewer`
   (§5; at most 5/s per board, always on a change), and puts the frame in the board's
   **latest-wins inbox**. Of two waiting frames the inbox keeps the sharper one.
4. The board's worker takes the frame and asks the scheduler whether it is due. The
   default policy is `burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3`: three
   reads 0.3 s apart at the end of a motion burst, and a read whenever 2 s of capture
   time passed without one (the keepalive floor; 3 s until the
   [live latency report](reports/2026-11-24-live-latency.md), 1 s let hands settle). A wall-clock backstop covers a frozen
   `capture_ts`. `BR_SCHEDULE` overrides the policy.
5. A due frame goes to `Recognizer.read` in a worker thread, giving a chess-frame
   `square_dist` and the board's direction, then a `PositionEvent`.
6. The orientation guard drops the read if the board points more than 45° from its
   recent reads (D10). Otherwise `BoardStabilizer` settles the read, under the board's
   `rules` setting (D14), and the event goes out with the settled board as its
   `placement` (`PositionEvent.with_placement`, which derives `pending`).
7. The hub publishes the event to `/viewer` subscribers and keeps the latest per board;
   the recorder writes every ingested frame to a session directory for replay, and the
   read's direction goes to the camera registry (D11).

There is deliberately **no** "find the legal move" or "commit a move" step.

### 4.4 Failure behavior

- **Hand over the board.** Covered squares read as confident empties. The scheduler
  skips the burst and reads at its end; `placement` holds the previous board until a
  change holds for its window. Every event still carries its read in `square_dist`.
- **Camera moved.** Nothing on the edge follows the board. The operator redraws the
  rectangle on `/control`; reads in between are whatever the recognizer sees.
- **Edge offline.** The stream goes quiet; the viewer marks the board stale on its side.
  The protocol's `stale` status is reserved, not emitted.
- **Low confidence.** Emitted verbatim; the consumer decides.
- **Missing weights.** The stub recognizer serves, loudly (D10).
- **Impossible board** (two white kings, a pawn on rank 8). Reported as read, status
  `ok`; under `standard` rules the settled `placement` does not adopt it (D14).

### 4.5 The inference node's interfaces (gateway, wire, hub)

**Ingress (WebSocket `/ingest`).** Two messages per frame: the JSON header as a text
message (`wire.FrameHeader`: `type`, `board_id`, `seq`, `capture_ts`, `stability`,
`sharpness` or null, `clock`, `burst_id` or null), then its JPEG as the next, binary
message. The node reads the two in lockstep, so nothing may be sent between them; the
JPEG is outside the JSON Schema (D13). The header may carry the legacy `corners`,
which is parsed and ignored, and which the JSON Schema leaves out. The node parses
every header strictly through `FrameHeader`: `"seq": "4"` or an unknown key refuses
the frame (an error reply, nothing recorded), because a string where the wire says
number is an edge bug the node must not hide. The `control` message kind is reserved; nothing uses it.

**Egress (WebSocket `/viewer`).** Position events as JSON (§5), the latest per board
on connect, interleaved with motion updates (§5, display only, never retained).

**Camera control (WebSockets `/boards/{id}/camera` and `/boards/{id}/control`).** The
phone's and the control pages' sockets for one board's camera (D11); the messages are
`board_reader_core.control`'s. Every text message names itself with `type`, and each
socket's direction is one discriminated union in the JSON Schema (`ReportMessage` and
`PhoneDown` on the phone's, `ConfigureMessage` and `ControlDown` on a control page's).

**HTTP.**

| Route | Purpose |
|---|---|
| `GET /healthz` | liveness, ingest counters, settle values, recording stats, the serving recognizer (and `fallback` when the stub serves) |
| `GET /boards/{id}` · `GET /boards/{id}/frame` | latest event · latest ingested frame |
| `POST /estimate` | one photo (JPEG body) → its event, synchronously; also published to the hub. Nothing to settle over: `placement` is the read, `pending` empty. |
| `POST /labels/{upload}` · `GET /uploads` · `GET /uploads/{upload}/photo` | confirm a retained photo's position as ground truth; list and fetch retained photos (`BR_SAVE_UPLOADS`) |

---

## 5. Position stream protocol (L3 output)

Contract: `board_reader_core.protocol` (pydantic models, D13; the JSON Schema of
every message in this section is `board_reader_core.schema`'s). One event per read:

```json
{
  "board_id": "A1",
  "utc": "2026-06-15T14:22:09.470Z",
  "changed_utc": "2026-06-15T14:22:08.120Z",
  "seq": 12831,
  "orientation": "a1_bottom_left",
  "classes": ["empty","P","N","B","R","Q","K","p","n","b","r","q","k"],
  "placement": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
  "pending": [],
  "square_dist": [[[0.01,0,0,0,0.99,0,0,0,0,0,0,0,0], "…8 squares a8..h8…"], "…8 rows…"],
  "confidence": 0.93,
  "stability": 0.12,
  "clock": { "running": "none" },
  "status": "ok"
}
```

- `utc` — the edge **capture** time of this read, RFC 3339 UTC with milliseconds and a `Z`.
- `changed_utc` — when `placement` last changed, same format and clock: the capture
  time of the earliest frame that showed the board as it now stands. The stabilizer
  gives the earliest *read* (the start of the run of reads that settled it; for a
  commit of several squares, the latest of their run starts); the node then
  back-dates it over the frames it received but did not read (`stream/backdate.py`):
  to the start of the unbroken run of frames, ending at that read, whose pixels on
  the changed squares match that read and not the last read of the old board. A hand
  over the square matches neither, so it stops the search. Pixels only, no extra
  recognizer read; it runs inside the read, so the event that first carries the new
  board already has the back-dated time. It stays put while the board holds, and
  equals `utc` on `/estimate`. Timestamp a move by the `changed_utc` of the first
  event that carries the new board: on the gate takes it sits a median 1.35 s after
  the truth boundary (p90 2.74 s; 1.46 s before back-dating), on the live take with
  the edge's burst tail 0.57 s (0.94 s before), against 3.12 s (p90 4.30 s) for that
  event's `utc`. What remains is mostly a hand still on the moved piece. The motion
  burst's calm start was worse in the tail and is not used
  ([report](reports/2026-11-24-changed-utc.md)). A replay derives it the same way.
- `seq` — per-board monotonic, a non-negative integer; authoritative for ordering.
- `orientation` — always `a1_bottom_left`, reserved. The placement is in the chess
  frame (rank 8 first, file a first, as in FEN), resolved per frame by the recognizer.
- `classes` — the 13-class index order of every per-square vector; uppercase = White,
  `empty` is index 0.
- `square_dist` — **canonical output**: this read's 8×8 13-vectors, rank 8 → rank 1,
  file a → h. The read's own board is its per-square argmax.
- `placement` — FEN field 1 (eight ranks of eight squares, a digit being a run of 1
  to 8 empty squares: the `isPlacement` check): the **settled board** on a stream, the
  read itself on a one-shot `/estimate`. See the settle rules below.
- `pending` — the squares, board order, where this read's argmax differs from
  `placement`: a move in flight, or a misread still being debounced. Empty when they
  agree, and always on `/estimate`.
- `confidence` — the mean per-square maximum probability of this read.
- `stability` — `[0,1]`, 1 = at rest; advisory.
- `clock.running` — `white|black|none`; always `none` until M6.
- `status` — `ok|recalibrating|needs_setup|stale`. Only `ok` is emitted; the other three
  are reserved. There is **no `illegal`**: an illegal-looking board is `ok`.
- Every field above is always sent; `square_dist` entries lie in [0, 1], and
  `confidence` and `stability` too. Unknown keys are refused. That each square sums
  to 1 and that `pending` is exactly where `placement` differs from the argmax is
  checked by `PositionEvent.validate()`, not by parsing: a distribution rounded to
  four decimals can tie two classes.
- **The settle rules** (streamed frames only) — the settled `placement` is a display
  debounce of the reads. A square
  changes only after it held the same class on every read across **0.5 s** and **≥2
  reads**. Under the board's `rules: standard` (D14) a settlement is committed only if
  it does not make the board less possible than it already is (≤1 king per colour,
  ≤16 pieces per colour, the promotion budget); a square blocked that way gives up after **2 windows** and the
  contested set falls through to the current read. The window is time, not a read
  count, because the read rate varies with load. Knobs: `BR_SETTLE_SECONDS`,
  `BR_SETTLE_MIN_READS`, `BR_SETTLE_STALE_WINDOWS`; `/healthz` reports them. Evidence:
  [reports/2026-11-24-settle-tuning.md](reports/2026-11-24-settle-tuning.md) (gate:
  104/119 settled perfect, 110/119 raw, median move-to-settle 3.0 s; with the 2 s
  keepalive 138/154 and 2.0 s, [live latency](reports/2026-11-24-live-latency.md); with
  three reads 0.3 s apart and the 0.5 s window 140/156 and ~0.6 s from burst end to
  settled, [latency round 2](reports/2026-11-24-latency-round2.md)). The window must
  stay strictly between the burst-end spacing and the run's span (0.3 < 0.5 ≤ 0.6):
  in the sweep a window that two reads span let a hand settle in 19 of 30 configs.
  This is not the downstream smoother (D3): it uses no move model and no legality; a
  consumer running its own smoother takes the reads from `square_dist`. The live
  viewer draws `placement`; it marks `pending` squares with dashed amber rings and
  shows no confidence or stability bars. Recordings made before 2026-11-24 carry the
  older shape (`placement` the read, a `settled` block beside it); `eval-session`
  reads both (`trial.read_events`).

**Motion updates (display only).** Between the sparse position events, `/viewer` also
carries one message per ingested frame, at most 5 a second per board but always when
`moving` changes:

```json
{"type": "motion", "board_id": "A1", "moving": true, "stability": 0.42, "ts": 1795551874.25}
```

`ts` is the frame's `capture_ts`; `moving` is `stability` below the read scheduler's
burst threshold (`schedule.ENTER_BURST`, 0.7, which the UI mirrors as `MOVING_BELOW`).
A position event has no `type`, so only a consumer that reads every message as one has
to change (`protocol.parse_viewer_message`). It is not a read: nothing records it,
the hub does not retain it, and replay does not produce it. The live viewer's motion
chip is driven by it.

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
   mid-game positions. The gate is seven takes of one set (§9).
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
  debounce absorb it on the gate (no hand frame settles). A settle window at or below
  the burst-end spacing lets hands through (19 of 30 swept configs leaked one; the
  shipped 3 × 0.3 s with 0.5 s leaked none), and so would a stuck motion meter; the
  08-13 take settling all its boards is the guard. A hand resting still over the board
  reads as rest and can settle under any window
  ([latency round 2](reports/2026-11-24-latency-round2.md)).
- **Framing (D11).** The board must fill ≥ ~50 % of the sent width; 35 % collapses
  perfect boards and portrait costs about a quarter of them. Wide framing was measured
  with synthetic padding only, and real distractors (people, other boards) are untested.
- **One physical set in the gate.** All seven gate takes film the same set and camera,
  so the settle defaults and the accuracy numbers may not carry over.
- **Orientation is inferred by the model.** A misread rotation would show as a whole
  board wrong at once. None was seen (zero rotation errors in the framing run). On the
  stream the orientation guard drops a read whose direction turns more than 45° (D10);
  offline the rotation diagnostics in `eval` and `eval-session` remain.
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
- **Clock as its own topic?** `clock.running` rides inside each event; a separate topic
  at the clock's ~1–2 Hz may fit better in v2.
