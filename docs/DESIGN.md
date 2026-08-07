# Camera-Based Chess Board Reader — Design Document

> A distributed system that reads live, over-the-board (OTB) chess **positions**
> using cameras and computer vision, as a lower-cost, no-special-equipment
> alternative to DGT electronic boards. The system streams **timestamped board
> positions with confidences** (plus clock state). It does **not** compute moves,
> legality, or game history — that is the application layer's job, downstream and
> out of scope.

**Status:** Draft v0.4 · **Date:** 2026-06-16 · **Author:** Felix (with research synthesis)

### Core philosophy (read this first)

**The CV system is a per-frame position estimator, not a chess engine.** For each
input frame it answers one question: *"what pieces are on which squares right
now, and how sure am I?"* It is essentially **stateless** with respect to the
game. Specifically, the CV layer:

- **streams positions, not moves.** An illegal arrangement looks exactly like a
  legal one — we report what we see.
- **never applies chess rules.** No legality, no move-resolution, no FEN
  completion, no takeback handling, no PGN. The application layer (a downstream
  consumer) owns all of that and is free to apply temporal/legality smoothing to
  our stream.
- **outputs piece placement only** (the first FEN field / an 8×8 grid), because
  side-to-move, castling rights, en-passant, and move counters are *not knowable
  from a single image* — they are history-derived.
- **emits per-square confidences** and an advisory **stability** signal, so the
  application layer can decide when a position is at rest and reconstruct the
  move sequence itself — including the case of **two pieces in motion at once**.

This deliberately pulls responsibility out of the CV layer. The research below
shows per-frame whole-board recognition is hard on real boards; we do *not* try
to hide that with in-CV rules. Instead we expose rich confidences and stream the
full transition (dense frames while things move), and let the consumer apply the
constraints that recover accuracy.

### Locked decisions (from review)
1. **CV layer streams positions only** — no moves, legality, game state, or PGN
   (application-layer concerns, out of scope). See *Core philosophy*.
2. **Output is piece-placement + confidences**, not a full FEN and not a move.
3. **No required calibration from the start position.** The system must work when
   dropped into an arbitrary mid-game position. A short **calibration round is
   optional** and may use *any* known labeled position, not specifically the start.
4. **Stability is advisory, not a gate.** We stream a per-frame stability/change
   score; the application layer owns move segmentation. The CV layer does not
   decide "a move happened."
5. **Camera placement:** optimize for the **angled / broadcast view** (~45–60°).
6. **Board detection runs on the edge** (it's cheap and the camera/board geometry
   is local). The edge sends **rectified board crops**; the central node need not
   re-localize.
7. **Recognition model:** **DINOv2 embeddings** + a lightweight classifier head;
   default works with no per-board training, optionally refined by calibration.
8. **Clock in v1:** **which-side-is-running only**, streamed as sensor state (the
   application layer interprets it). Time OCR deferred.
9. **Edge runtime in the sim:** **in the browser** (JS/WASM OpenCV); keep the edge
   interface abstract enough to later port to a real Pi process.
10. **No Lichess in v1.** The POC ships its own **viewer web app** rendering live
    positions; PGN/Lichess export is an optional later adapter.
11. **Sim transport = WebSocket** for both frame ingress and position egress.
12. **Board is the logical unit; an edge may host several boards** (one camera can
    film multiple boards as a cost option). POC = one board.
13. **Language = Python** for the central node (FastAPI + asyncio + ONNX Runtime).
14. **Canonical output = full 64×13 per-square distribution** (placement string +
    scalar confidences are derived from it).
15. **Fixed canonical board orientation** set at setup, reported as metadata. The CV
    layer never infers orientation from the *pieces*; after a reposition it may
    auto-recover the frame from *board-intrinsic* cues (printed edge coordinates,
    clock side) and otherwise falls back to manual re-setup (§4.4.1).
16. **Backbone = DINOv2-S (Apache-2.0)**, kept behind a swappable interface; DINOv3
    is a candidate, not the default (licensing).

---

## 1. Background

### 1.1 The problem domain

Tournament chess broadcasting today relies almost entirely on **DGT electronic
boards**. These work well but have real drawbacks:

- **Hardware cost & lock-in.** A working DGT board + the *mandatory* electronic
  pieces costs **~€500–€750 per board** before clock and cabling. Ordinary sets
  do not work — every piece contains a passive LC resonant circuit tuned to a
  unique frequency (~90–350 kHz) that encodes both type and colour, sensed by
  printed coil loops under the board (patent US6168158).
- **Cabling logistics that scale linearly.** Boards chain on an RS-232/USB bus
  (9600 baud, 8N1). A tournament hall needs a "connection set" per ~12 boards,
  bus boxes, patch cables, and runs back to a host PC.
- **Move-detection edge cases.** Fast blitz moves can be bundled or detected
  mid-air; castling and captures have timing requirements; transient illegal
  positions are emitted on every physical change and must be filtered.

### 1.2 What DGT actually emits (and how our split differs)

A DGT board does **not** emit moves either. It emits **per-square occupancy
transitions** (`DGT_MSG_FIELD_UPDATE`, field 0–63 + piece code), and host
software (DGT LiveChess) reconstructs the move, validates legality, and produces
PGN. Crucially, DGT knows **piece identity per square** with ~100% reliability;
a camera must *infer* it under occlusion, perspective, and lighting.

Our architecture mirrors DGT's *split of concerns*: the sensor (DGT board / our
CV system) reports **board state**; a separate software layer (DGT LiveChess /
the application consuming our stream) turns state into moves, legality, and PGN.
We are building the sensor, not LiveChess. The two things DGT gives for free that
are genuinely hard for a camera — **per-square piece identity** and an
**unambiguous move-completion signal** (the clock lever) — we address with
confidences + dense streaming and with the which-side-running clock signal,
respectively, but we do *not* try to fully replace them inside the CV layer.

### 1.3 Why now — state of computer-vision chess reading

| Project | Approach | Real-world result |
|---|---|---|
| **Chesscog** (Wölflein & Arandjelović, 2021) | RANSAC localization → occupancy CNN → piece CNN → FEN | 93.9% whole-board on **synthetic**; **2.3%** on real photos; **99.8% per-square after few-shot adaptation from 2 photos** of the set |
| **LiveChess2FEN** (Mallasén, 2020) | Localize once → 64-square CNN → FEN + temporal change-detection | ~92% piece acc; **<1 s/board on a Jetson Nano** with TensorRT |
| **End-to-End / ChessReD** (Masouris & van Gemert, 2024) | Single ResNeXt predicts whole board; **10,800 real photos** benchmark | **15.3% perfect boards** on real angled photos |
| **CVChess** (2025) | Homography → residual CNN | 98.9% per-square but ~64% perfect FEN in-domain, ~30% out-of-domain |
| **ChessCam** (Pbatch) | Phone on tripod → Lichess broadcast, no special board | Closest existing prior art |

**The key finding:** per-frame whole-board classification is hard on real angled
boards (one wrong square ruins a whole-board read; Chesscog's per-square error
goes 0.23% → 73.6% synthetic→real). The literature recovers accuracy with
**temporal consistency + move legality** (track changes, snap to legal moves).

**Our design choice given this:** we keep that recovery mechanism but **relocate
it to the application layer**. The CV layer streams per-frame placements with
per-square confidences and streams *through* transitions (dense frames while
pieces move); the consumer applies temporal/legality smoothing. Within the CV
layer we still raise raw per-frame quality via (a) optional per-set calibration,
(b) multi-frame burst voting on settled frames, and (c) honest per-square
confidences (occluded/uncertain squares are flagged, not guessed-and-hidden).

---

## 2. Aim

### 2.1 Goal statement

Build a system where **one camera films one chess board**, a small **edge node**
detects and rectifies the board and forwards relevant frames, and a **central
vision node** estimates, **for each frame, the board's piece placement with
confidences**, for **many boards at once**. The output is a **stream of
timestamped positions** (plus clock state) that any downstream application can
consume to reconstruct moves, legality, PGN, broadcasts, analysis, etc.

### 2.2 Concrete objectives

- **Positions, not moves.** Emit a documented stream of timestamped board
  placements + per-square confidences + advisory stability. No move/legality/PGN
  logic in the system.
- **Many boards per central node.** The edge forwards frames at an **adaptive
  rate** (low baseline at rest, dense bursts during change) so the central node's
  scarce resource — compute — scales to many boards while still capturing
  transitions.
- **Honest under obstruction.** We do not block on a clean frame; we stream what
  we see and **flag occluded/uncertain squares with low confidence**, letting the
  consumer decide.
- **Accurate timestamps.** Timestamp at **capture time on the edge** (sensor
  timestamp), not server-receive time. Target ≤ tens of ms.
- **Clock (v1): which side is running**, streamed as sensor state.
- **No hardware purchase yet.** Deliver a **local simulation**: smartphone camera,
  a **web app as the simulated Raspberry Pi edge node**, and a **Docker container
  as the simulated central (Jetson) node**, plus a **viewer web app**.

### 2.3 Explicit non-goals (out of scope — these belong to the application layer)

- **Move detection / segmentation**, including ordering when multiple pieces move.
- **Legality, illegal-position handling, takebacks, game state, PGN, broadcast.**
- **FEN completion** beyond piece placement (side-to-move, castling, en-passant,
  counters are history-derived).
- Not emulating the DGT serial/USB byte protocol.
- Not bughouse/Chess960-first.
- Not requiring calibration from the start position (calibration is optional and
  position-agnostic).

---

## 3. Key design decisions (and the research behind them)

### D1 — The CV layer is a stateless per-frame position estimator
Frame in → 8×8 piece-placement estimate + per-square confidence + advisory
stability out. No game state, no legality, no move-tracking. The temporal +
legality intelligence the literature relies on lives in the **application layer**,
fed by our rich per-frame stream. This is the load-bearing decision; everything
else follows.

### D2 — Output is piece placement + a full per-square distribution, not a FEN or a move
We emit the **piece-placement field** (FEN field 1, e.g.
`rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR`) as a human-readable convenience,
backed by the **canonical output: a full per-square probability distribution
(64×13) over {empty + 12 piece classes}**, plus a derived position-level
confidence. We emit the full distribution rather than a single scalar per square
because a scalar collapses *which alternatives a square was confused with* (e.g.
`0.4 white-knight / 0.35 white-bishop` vs `0.4 empty / 0.4 black-pawn` both read
as `0.4`), and that ambiguity structure is exactly what downstream probabilistic
move inference needs. At fp16 the grid is ≈1.6 KB — negligible beside a JPEG crop.
The 8×8 scalar grid and `placement` string are both derivable from it. We do
**not** fabricate side-to-move, castling, en-passant, or counters — they cannot be
derived from one image. A consumer that wants a full FEN assembles it from the
move history it reconstructs.

### D3 — Stability is an advisory signal; the consumer segments moves
Each position event carries `stability ∈ [0,1]` (1.0 = at rest; equivalently the
complement `p_change = 1 − stability`), computed cheaply from inter-frame motion
on the edge. It is a **hint**, not a decision: the application layer decides when
a move completed and in what order pieces moved, possibly ignoring our hint
entirely. Because we stream densely through transitions, the consumer can observe
**two pieces moving simultaneously** and resolve ordering itself. (Open: whether
`stability` earns its keep at all, or whether the consumer is better served by raw
positions alone — we will keep it cheap and optional.)

### D4 — Edge does board detection + rectification; adaptive frame forwarding
Board localization is cheap and local, so it runs on the edge (Canny→Hough→
RANSAC homography, or a 4-corner tap during setup), and the edge sends **rectified
board crops**. The edge forwards frames at an **adaptive rate**:
- **at rest:** a low keepalive rate (e.g. ~1–2 fps) with high stability;
- **during motion:** a **burst** (e.g. 10–15 fps) with low stability, so the
  consumer sees the whole transition.

This replaces the old "only send one settled frame" gate — we no longer hide
transitions, but we still avoid streaming full 30 fps continuously, preserving
the many-boards-per-Jetson goal. Each frame is stamped with capture time + seq.

### D5 — Authoritative timestamp at capture, sync via chrony/NTP
Read the sensor capture timestamp on the edge and convert to wall-clock via a
one-time `CLOCK_MONOTONIC↔CLOCK_REALTIME` offset. Sync nodes with **chrony
against a local LAN NTP server over Ethernet** (tens of µs, well past our target).
PTP is overkill *and* unavailable on Pi 4 / Jetson Nano hardware. Every event
carries wall-clock time **and** a per-board monotonic `seq` (authoritative
ordering, survives clock steps).

### D6 — Recognition: DINOv2 embeddings, calibration optional & position-agnostic
Default path needs **no per-board training**: a frozen **DINOv2 ViT** embeds each
square crop, and a **classifier head trained once (offline, by us) on public
chess datasets** (ChessReD real + synthetic) maps embeddings → {empty + 12 piece
classes}. Optionally, a **calibration round** refines accuracy for a specific set
by building per-set class centroids from **any known labeled position** the
operator supplies (not necessarily the start). See §4.6.

### D7 — Clock (v1): which-side-running, streamed as sensor state
Temporal differencing of two clock ROIs over a ~1.5–2 s window → `running ∈
{white, black, none}`. Streamed as a sensor reading; the application layer
decides what it means for move timing. Time OCR (digit CNN / `ssocr`) is deferred.

### D8 — Optimize for the angled broadcast view
Target a tripod view at ~45–60°. Implication: **back-rank occlusion** is expected;
occluded squares are reported with **low confidence** rather than guessed. We do
not lean on legality to fill them in (that's the consumer's option).

### D9 — Target Jetson Orin Nano; simulate on x86; models in ONNX
The original Jetson Nano is EOL (JetPack 4.6 / CUDA 10.2, EOL Jan 2027). Reference
target = **Jetson Orin Nano (Super)** (~$249, 67 INT8 TOPS, JetPack 6.x); note
**Raspberry Pi 5 + Hailo-8L** (~$130–150) as a value alternative. **Skip
DeepStream** at this scale; OpenCV + TensorRT (FP16) suffices. **Author models in
ONNX** (portable); build TensorRT engines *only on-device*. Sim = plain
`linux/amd64` Docker + ONNX Runtime (CPU or host CUDA EP) — logic/integration
fidelity, not a Jetson benchmark.

---

## 4. Architecture

### 4.1 Layered overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L0  CAPTURE        Camera per board (sim: smartphone) + clock in view       │
├──────────────────────────────────────────────────────────────────────────┤
│ L1  EDGE NODE      (sim: browser web app "virtual Pi")                       │
│     • board detection + rectification (homography, cached)                  │
│     • inter-frame motion → stability score                                  │
│     • adaptive frame forwarding (keepalive + bursts)                        │
│     • capture-time timestamp; JPEG (rectified) crop encode                  │
│     • clock ROI motion (which side runs)                                    │
├──────────────────────────────────────────────────────────────────────────┤
│ TRANSPORT          WebSocket: frames+control in │ position stream out        │
├──────────────────────────────────────────────────────────────────────────┤
│ L2  CENTRAL VISION (sim: Docker container "virtual Jetson") — STATELESS      │
│     • per-frame board-placement estimator (DINOv2 + head, ONNX)             │
│     • per-square + position confidences; optional burst voting              │
│     • clock running-side passthrough                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ L3  POSITION STREAM   timestamped {placement, confidences, stability, clock} │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  (out of scope — downstream consumers)
┌──────────────────────────────────────────────────────────────────────────┐
│ APPLICATION LAYER  move reconstruction · legality · game state · PGN ·       │
│ (consumers)        broadcast.   POC ships a thin VIEWER that just renders    │
│                    the latest placement — no move logic required.            │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module responsibilities

**L1 Edge node** (the "Raspberry Pi"; sim = browser web app, `getUserMedia`):
- `localizer` — detects board, computes/caches homography; rectifies to a
  canonical top-down crop. Re-runs only on ROI drift (camera bumped).
- `motion-meter` — low-res inter-frame differencing → `stability` score.
- `frame-forwarder` — adaptive rate: keepalive at rest + burst during motion.
- `clock-watcher` — two-ROI differencing → `running ∈ {white,black,none}`.
- `publisher` — JPEG-encodes the rectified crop, stamps capture time + seq +
  stability, sends over the WebSocket; sends control messages on setup.

**L2 Central vision** (the "Jetson"; Docker container) — **stateless per board**
except for cached model + optional calibration centroids:
- `frame-router` — receives crops per board id, dispatches to the worker pool.
- `estimator` — DINOv2 embed each of 64 squares → classifier head → per-square
  distribution over {empty + 12 classes}; assembles placement + confidences.
- `burst-voter` (optional) — if the edge tagged a burst, confidence-weighted vote
  per square across the burst for a cleaner estimate; otherwise pass per-frame.
- `clock-passthrough` — forwards running-side (v1); time OCR later.

**L3 Position stream** — canonical output (see §5).

**Application layer (out of scope)** — independent consumers reconstruct moves,
apply legality, maintain game state, build PGN, drive broadcasts. The POC ships
only a **viewer** that renders the latest placement directly.

### 4.3 Data flow for one frame

1. Edge captures frame, rectifies via cached homography, computes `stability`.
2. Edge decides to forward (always during a burst; at keepalive rate otherwise),
   JPEG-encodes the rectified crop, stamps `capture_ts` + `seq` + `stability`.
3. Central `frame-router` → `estimator`: DINOv2 + head over 64 squares.
4. (Optional) `burst-voter` refines across a tagged burst.
5. L3 emits a **position event**: placement + per-square confidences + position
   confidence + stability + clock + `capture_ts` + `seq`.
6. The viewer renders it; other consumers reconstruct moves/PGN as they wish.

(There is deliberately **no** "find the legal move" or "commit a move" step.)

### 4.4 Failure & robustness behavior

- **Occlusion (hand over board).** Affected squares get **low per-square
  confidence**; the position is still emitted. Burst voting recovers the estimate
  once the hand leaves. We never block the stream.
- **Camera moved.** Handled in detail in §4.4.1 below — the case splits into a
  recoverable *bump/drift* and an orientation-invalidating *reposition*.
- **Edge offline.** Egress marks the board stale (no frames / status event).
- **Low overall confidence.** Emitted verbatim with low `confidence`; the consumer
  decides whether to trust, smooth, or ignore it.

### 4.4.1 Camera movement

"The camera moved" is really **two different events** with different handling. The
distinction matters because the CV layer fixes its board orientation at setup
(decision #15) and refuses to re-derive it from the pieces — so geometry can be
re-detected automatically, but *which corner is `a1`* cannot.

**(a) Bump / drift — camera shifts but still frames the board.** Recoverable
entirely on the edge. Because board localization and the cached homography live on
the edge (decision #6), a bump never reaches the central node as bad data — it is a
local re-rectification:
- `localizer` detects ROI drift (the cached corner/homography no longer fits).
- It re-runs detection (Canny→Hough→RANSAC), recomputes the homography, and checks
  the new corner correspondence against the cached orientation reference.
- If the corners map **consistently** with the cached orientation → auto-recover.
  The board is flagged `status: recalibrating` and placements are **suppressed**
  (see "during the transient" below) only until the new homography stabilizes,
  then it returns to `ok` and resumes streaming. The central node sees nothing but
  the status flag — it keeps receiving canonical top-down crops throughout.

Because there is no required start-position calibration (decision #3), recovery
does **not** need the board reset — it re-locks on whatever mid-game position is
present.

**(b) Reposition — moved far enough to invalidate the orientation reference.** If
the camera is carried to a new vantage (e.g. the other side of the table), the
board can still be *located*, but the original corner-tap frame is now meaningless
and the board's 4-fold rotational ambiguity must be re-resolved. The edge
**attempts an automatic orientation guess first**, and only falls back to manual
re-setup if that fails:

1. **Auto-orientation from board-intrinsic cues.** The edge tries to recover the
   frame from the *board's own markings*, never from the piece arrangement (which
   would breach decision #15 and the stateless principle):
   - **printed edge coordinates** (`a`–`h` files / `1`–`8` ranks lettered on the
     board border) read by a small OCR pass on the rectified crop — the strongest
     cue when present;
   - **clock side** as a tiebreaker — the clock's known position relative to the
     board (a configured setup fact) constrains which edge faces the players.
   If a cue resolves the rotation with sufficient confidence, the edge re-locks the
   orientation automatically, flags `status: recalibrating` during the transient,
   then returns to `ok`. No operator action needed.

2. **Manual re-setup fallback.** If no intrinsic cue is readable (no edge
   coordinates, clock not in view / ambiguous) or the guess is low-confidence, the
   board is flagged `status: needs_setup` and **stops emitting placements** until
   the operator re-declares the frame (new corner-tap / orientation), after which it
   returns to `ok`.

Note the auto-guess reads the *board*, not the *game* — it is orientation recovery,
not position inference, so it does not make the CV layer stateful or rule-aware.

**Threshold between (a) and (b).** Geometry re-detected **and** corner mapping
consistent with the cached orientation → treat as a bump, auto-recover. Geometry
not locatable, or corner mapping ambiguous/rotated → treat as a reposition, which
then attempts auto-orientation (b.1) before falling back to manual re-setup (b.2).
The orientation-consistency check (not just "did we find a board?") is what
separates the two.

**How the browser edge implements (a): track, do not re-detect.** *(added 2026-08-07,
shipped in `/camera`.)* The recovery above is written as "re-run detection and check the
new corners against the cached orientation reference". The browser edge does something
strictly stronger and cheaper: it **tracks the board plane continuously**, frame to
frame — Lucas-Kanade over the board's own lattice of square corners, a RANSAC homography
over the points that agree, the four corners pushed through it.

The reason is decision #15. A fresh detection returns an *unlabelled* quad, so the
orientation reference has to be re-checked and can be lost; a tracked quad carries its
labelling with it by construction, because the corners are never re-derived, only moved.
That collapses the (a)/(b) distinction for any movement that keeps the board in view —
including a full walk around the table, which is a §4.4.1(b) *reposition* by the
threshold above, yet needs no orientation recovery at all when tracked. This is what
makes the multi-viewpoint calibration walk (§4.6) possible.

The suppression rule below is unchanged and is what the tracker does on failure: losing
the board **freezes** the last quad that fitted and says so, rather than reporting a
half-fitted one.

**During the transient (both cases).** We **suppress placements** (or floor
`confidence` to ~0) rather than stream through, and surface the status
(`recalibrating` / `needs_setup`) instead. Rationale: a mis-localized camera
produces *geometrically* wrong crops, so the per-square distributions would be
**confidently wrong** — worse for downstream temporal smoothing than an honest gap,
since a consumer cannot distinguish a confident-but-wrong placement from a real one.
A clean gap is recoverable; a confident lie corrupts the consumer's state.

### 4.5 Central node ("Jetson") software structure

```
 Edge ─WS(frame|control)─▶ Gateway (FastAPI WS) ─▶ per-board inbox queue
   (bounded; frames latest-wins or burst-grouped; control msgs prioritized)
        ─▶ Pre-filter  (dedup by hash · sharpness floor · cheap "blank?" reject)
        ─▶ inference job queue
        ─▶ Estimator worker pool (shared GPU)  DINOv2 → head → placement+conf
        ─▶ (optional) burst-voter
        ─▶ Position-stream hub (pub/sub, retains latest per board)
        ─WS/SSE▶ Viewer + any consumers ; ─▶ recorder (NDJSON)
```

- **Ingress (WebSocket).** Two message kinds from the edge: `frame` (binary JPEG
  + JSON header: `board_id`, `capture_ts`, `seq`, `stability`, optional
  `burst_id`, optional `corners`) and `control` (`hello`/register, `calibrate`,
  `clock_config` — still reserved, none implemented). Server replies with acks.
  A frame without `corners` is a rectified board crop (the original contract); a
  frame **with** `corners` (a8,h8,h1,a1 in its own pixels) is an original camera
  image, routed like the snapshot path below — photo-mode estimators cut
  full-piece boxes from it, others get a server-side rectification. Corners ride
  on **every** such frame rather than in a `set_corners` control message
  (superseding the earlier sketch): the gateway stays stateless per frame, so a
  reconnect, a restart or a mid-stream re-crop needs no handshake. This is also
  how the §6 browser edge (`/camera`) streams — a browser cannot cheaply
  rectify, and tall-piece headroom means the un-warped photo is the *better*
  input anyway (§4.6).
- **Snapshot ingress (HTTP, convenience).** `POST /estimate` accepts a single
  frame and returns the position event synchronously (also published to the hub):
  either an already-rectified crop (JPEG body), or — preferred for real photos —
  the **original photo plus the four surface corners** (`?corners=`), from which
  the estimator crops **perspective-aware full-piece boxes** (see §4.6). Used by
  the phone **snapshot page** (a WP-H3-style consumer served by the viewer;
  Lichess analysis/editor links are assembled client-side from the placement).
  `POST /detect` proposes the surface corners automatically (geometry only —
  orientation stays an operator decision, per decision #15).
- **Detection code placement.** Board detection conceptually runs on the edge
  (D4); the algorithm lives in `board_reader_core.boarddetect` so the browser-edge
  snapshot path (which cannot run OpenCV cheaply) can delegate it to the
  inference node's `/detect`, while the real Pi edge imports the same module
  locally. Core still imports neither node.
- **Per-board inbox queue** — bounded; frames latest-wins at rest, burst-grouped
  during motion; control messages prioritized. The backpressure valve.
- **Pre-filter** — cheap server-side hygiene only (dedup, sharpness floor, reject
  blank/garbage frames). **No "is it a new position?" logic** — that would be
  state/move reasoning, which we deliberately exclude.
- **Estimator worker pool** — *shared*, not per-board; the resource one Jetson
  divides across edges. asyncio for I/O + `run_in_executor` for inference.
- **Position-stream hub** — pub/sub; **retains the latest position per board** for
  late subscribers; feeds the recorder. Stateless beyond "last value".
- The central node also **serves the static web apps** (edge + viewer) in the sim.

Note there is **no move-resolver / game-state module** — by design.

### 4.6 Recognition model (DINOv2, calibration optional)

**Default (no per-board training):**
- A frozen **DINOv2 ViT-S/14** (~21M params; ViT-B for headroom) embeds
  each rectified square crop → 384/768-d vector. DINOv2 is chosen over DINOv3
  deliberately: it is **Apache-2.0** (no gated download, no "Built with DINOv3"
  attribution clause, no custom-license legal review) with a mature ONNX-export
  ecosystem. DINOv3 has stronger frozen *dense* features (Gram-anchoring) and is a
  candidate if per-square accuracy proves backbone-limited, so the embedding
  backbone stays behind an interface — but it is **not** the default. (See risks.)
- ~~A **classifier head we train once, offline**, on public datasets maps
  embeddings → 13 classes and works on an unseen set immediately.~~ **Measured
  false (2026-07-18):** piece *type* does not transfer across physical piece
  styles on frozen DINOv2 features. A ChessReD-built exemplar bank scores
  255/320 zero-shot on our labeled photos and a ChessReD-trained linear probe
  ties it at 254/320 (vs 294/320 same-style calibrated; the probe reaches 90.7%
  *within* ChessReD, so the gap is the features, not the classifier). Mixing
  styles into one bank actively hurts (272 vs 291 user-only). What *does*
  transfer is **occupancy** (empty vs occupied): 94–96% zero-shot, style-robust.

**Optional calibration (position-agnostic):**
- The operator supplies **any known position** (start position, or a typed-in
  current position) for one rectified frame. We build **per-set class centroids**
  from those labeled square embeddings and switch to (or blend with) a
  nearest-centroid classifier. This is the Chesscog "few-shot from a couple of
  photos → 99.8% per-square" trick, generalized to any labeled position so we are
  **not tied to the start position**.
- **kNN exemplar bank, validated stronger.** The few-shot classifier is now a kNN
  exemplar bank rather than a per-class mean: every calibration embedding is kept
  as its own exemplar and a query is scored by its **nearest exemplar per class**,
  with **per-board transductive centering** (bank and query batch each re-centered
  on their own mean before the cosine step) absorbing per-photo lighting/backbone
  shift. Leave-one-photo-out validation on labeled real photos: 294/320 vs 261/320
  for centroids, and empty→piece hallucinations cut 28→6. Centroid files remain
  fully loadable (the classifier is picked by file kind); calibrating from a new
  photo now *adds* its exemplars to the running bank instead of replacing it.

**Multi-viewpoint calibration — the walk (2026-08-07, shipped in `/camera`).** Because
calibration *adds* exemplars rather than replacing them, the number of calibration
photos is a free parameter, and the evidence says more is better along the axes that
actually break: a single-frame bank transfers poorly across lighting (~70–75%) where a
two-lighting bank reaches ~90–95%, and dim-room accuracy recovered ~50%→98.8% purely
from same-conditions appends. **Viewing angle is the same kind of axis**, and it was the
one still calibrated from exactly one sample.

So the browser edge offers a **calibration walk**: the operator sets the board to the
starting position and carries the phone around it while the page collects one
full-resolution frame per *genuinely new vantage* and posts them one at a time.
Consequences worth stating as decisions:

- **Novelty is measured on the quad's shape**, translation and scale divided out — that
  is the viewing angle and nothing else, so standing closer does not count as a new
  viewpoint and a step around the table does.
- **It depends on tracking, not on re-declaration** (§4.4.1): the corners follow the
  board, so every vantage arrives already labelled and the operator never restates which
  corner is `a8`. Without tracking this feature is not buildable.
- **Forwarding is suppressed for the duration** — a camera in transit is precisely the
  mis-localised camera §4.4.1 says never to stream through.
- **A refused viewpoint is not fatal.** The `/calibrate` anti-poisoning guard is per
  photo and stays exactly as it is; a vantage it cannot read as the start position is
  reported and skipped, and the others still calibrate.

This does not weaken the "no calibration required" default (decision #3) or the
zero-setup path below — it is the optional path made worth taking.

**Zero-setup via start-position auto-calibration (2026-07-18).** Given the
occupancy/type split above, "works out of the box" is engineered rather than
trained: the gateway can hold a style-agnostic **occupancy bank** (built from
ChessReD, `BR_OCCUPANCY_BANK`). While no piece-type calibration is active,
`/estimate` answers with occupancy only (HTTP-response-only, never published to
the hub — the hub carries real position events exclusively). The occupancy
pattern of the chess **starting position** is unmistakable, so the moment a
photo matches it (≥60/64 squares, any of the 4 rotations — this also
self-corrects rotated corner detection; White's side resolved by piece
brightness), the server silently calibrates piece types *from that photo* via
the exemplar-bank machinery, using only squares whose occupancy agrees with the
pattern (so a photo taken a move or two in never contributes wrong labels).
The session bank persists/restores only when ``BR_SESSION_BANK=<path>`` opts in
(unset → calibration is in-memory only and nothing is ever written; a
cwd-relative default let ordinary test runs clobber the real bank). The *calibrated* path self-refreshes
too: a start-position sighting read at sub-threshold confidence (lighting
drift) appends label-confirmed exemplars from that photo (only squares whose
prediction matches the rotated start placement, ≥55/64 required, bank capped) —
so photographing the board before each game keeps adapting the bank to new
lighting with zero interaction. Validated motivation: dim-room accuracy went
~50%→98.8% purely from same-conditions appends. UX: photograph the board before the
game starts once per new set — no buttons, no labeling. Note: ChessReD is
**CC BY-NC-SA 4.0** (non-commercial); the occupancy bank derived from it must
not be redistributed with the repo.

**Multi-style trained classifier + calibration-as-booster (2026-07-18 PM).**
The campaign's answer to the zero-shot gap is a **multi-style supervised crop
classifier** (ResNet18, 5 piece styles, shape-bias augmentation) served via
``BR_MODEL`` (`TrainedEstimator`): 92.5%/320 zero-shot on the held-out user set
vs 91.9% calibrated kNN. Field testing the same day sharpened three decisions:

- **Crop headroom is load-bearing.** The pipeline had drifted to
  ``headroom=0.15`` (near-base-only crops) against this document's ~1.6
  intent. At steep camera elevations (~25–45°) that *decapitates* tall pieces —
  the classifier sees only lower bodies, so q→b, K→Q, R→P, B→P confusions
  dominate and cut-off heads hallucinate pawns one square up (measured 57/64,
  every major piece wrong, on a steep dim shot). Training and serving must use
  tall crops (~1.6–1.8 square-widths, image-up); the geometry ships with the
  model, and retraining with tall crops is the standing fix for the steep-angle
  regime.
- **Calibration boosts, never replaces.** The exemplar bank is near-perfect
  under conditions it has seen (64/64 on all bank-adjacent eval boards) but
  brittle outside them (46/64 on the steep dim shot); the trained model is the
  reverse. ``EnsembleEstimator`` averages the two square distributions 50/50
  and tracks the better member square-by-square: 439/448 total vs 415 (trained)
  and 430 (bank). ``/calibrate``, start-position auto-calibration, and the
  self-refresh path all wrap the trained model rather than displacing it, and
  ``BR_MODEL`` + a persisted ``BR_SESSION_BANK`` restore the ensemble at boot.
  The trained path also **bootstraps** its own booster: its first start-position
  sighting builds the bank from that photo (labeled by its own 13-class reading,
  agreement-masked) with zero user action.
- **Orientation is resolved by chess plausibility, not by the board.** Corner
  geometry cannot say which corner is a8 (decision #15), and auto-detected
  corners are arbitrarily oriented. With ``autorotate=1`` (sent by the snapshot
  UI whenever the user hasn't hand-placed/relabeled corners) the gateway picks
  the grid rotation with (1) fewest pawns on ranks 1/8, tie-broken by (2)
  White's men nearer rank 1. Square-color parity and printed border coordinates
  are deliberately **not** used: players set boards up ignoring printed
  coordinates, and non-FIDE-printed boards exist (both observed in the field on
  2026-07-18).

**Ranked alternatives:**
1. **DINOv2 embeddings + head/centroid** — chosen. ⭐
2. **MobileNetV2/SqueezeNet fine-tune** — LiveChess2FEN route, <1 s/board on a
   Nano; needs more training data than the embedding approach.
3. **Template matching** from calibration crops — cheap occupancy/fallback;
   brittle to shadows.
4. **CLIP zero-shot / Grounding DINO / OWLv2** — unreliable for piece *type* at an
   angle; skip.
5. **YOLOv8/v11 detection** — strong but needs labeled training; not few-shot.

**Honest accuracy note.** Because we deliberately keep legality/temporal logic out
of the CV layer, **raw per-frame placement accuracy on real angled boards is the
binding quality constraint** — there is no in-CV crutch. We mitigate with
calibration, burst voting, and per-square confidences, and we rely on the
application layer to apply temporal/legality smoothing across the stream. This is
an intentional division of labor, and the per-square confidences are precisely
what make the downstream smoothing tractable.

**Square cropping (WP-B1) has two modes.** From a rectified top-down crop
(streaming path), squares are equal tiles — cheap, but tall pieces are smeared
by the warp and truncated by the tile, which caps recognition quality. For
single-photo estimation (snapshot path) the estimator instead crops
**perspective-aware full-piece boxes from the original photo**: each square's
projected base quad extended upward by ~1.6 square-widths, so the backbone sees
whole pieces (the Chesscog-style trick). Calibration must use the same crop
mode as inference; the centroid file records it.

**Multiple frames:**
- **Settled burst → classification.** Confidence-weighted per-square vote over the
  sharp frames the edge sent during quiet. (Motion-blurred frames are bad to
  classify.)
- **Motion frames → streamed verbatim.** We pass them through with low stability;
  the consumer uses the transition to reason about ordering.

---

## 5. Position stream protocol (L3 output)

**A position event = capture timestamp + piece placement + confidences +
stability (+ clock).** Sim transport is a **WebSocket/SSE** pub-sub stream; real
hardware can add MQTT (retained latest position).

```json
{
  "board_id": "A1",
  "utc": "2026-06-15T14:22:09.470Z",
  "seq": 12831,
  "orientation": "a1_bottom_left",
  "classes": ["empty","P","N","B","R","Q","K","p","n","b","r","q","k"],
  "placement": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
  "square_dist": [
    [[0.01,0,0,0,0.99,0,0,0,0,0,0,0,0], "... 8 squares a8..h8, each a 13-vector ..."],
    "... 8 rows, rank 8 → rank 1; within a row file a → h ..."
  ],
  "confidence": 0.93,
  "stability": 0.12,
  "clock": { "running": "black" },
  "status": "ok"
}
```

Field notes:
- `utc` = edge **capture (sensor) time**, not server-receive time.
- `seq` = per-board monotonic counter; authoritative for ordering.
- `orientation` = the **fixed canonical frame** the placement is reported in,
  established at setup (corner-tap / rectification) — e.g. `a1_bottom_left`. The CV
  layer does **not** infer "White's view" from the pieces (unreliable from one
  image, and it would breach the stateless principle); it reports the deterministic
  frame and lets the consumer rotate if it wants.
- `classes` = the 13-class index order for every per-square vector. Uppercase =
  white, lowercase = black; `empty` is index 0.
- `square_dist` = **canonical output**: an 8×8 grid of 13-vectors, each a
  probability distribution over `classes` (rank 8 → rank 1, file a → h). This is
  the full information the estimator produces; everything below is derived from it.
- `placement` = **FEN field 1 only**, the argmax of `square_dist` per square, as a
  human-readable convenience. No side-to-move, castling, en-passant, or counters —
  those are not knowable from one image.
- `confidence` = overall position confidence (e.g. min/mean of the per-square max
  probability). A per-square scalar grid, if a consumer wants one, is likewise the
  per-square max of `square_dist`.
- `stability` ∈ [0,1] (1 = at rest); advisory only — the consumer segments moves.
- `clock.running` ∈ `white|black|none` (v1; embedded as a field per board event;
  times added later).
- `status` ∈ `ok|recalibrating|needs_setup|stale`. `recalibrating` = transient
  re-rectification after a bump (placements suppressed until stable);
  `needs_setup` = camera repositioned, orientation reference invalidated, awaiting
  operator re-setup (see §4.4.1). Note: **no `illegal`** — legality is not ours to
  judge; an illegal-looking placement is reported as `ok`.

The application layer (a consumer) is responsible for turning this stream into
moves, full FENs, legality decisions, and PGN.

> This socket is the seam, and nothing has been built on the far side of it yet —
> the consumers that exist (the viewer UI, the headless `board-reader-inference
> positions` tail) render or capture the stream and smooth nothing, so no
> temporal consistency exists anywhere. Not a change to D1/D3; a statement of
> what is unbuilt. See `docs/architecture.md`, "Temporal consistency".

---

## 6. Local simulation plan (no hardware)

| Real component | Simulated as |
|---|---|
| Camera (one per board) | **Smartphone browser** (`getUserMedia`) on a real board, or a looped recorded video |
| Raspberry Pi edge node | **Browser web app** ("virtual Pi"): board detect + rectify + motion/stability + adaptive forward (WASM/JS OpenCV) |
| Network transport | **WebSocket** (frames in, position stream out) on `localhost`/LAN |
| Jetson central node | **`linux/amd64` Docker container**, ONNX Runtime (CPU or host CUDA EP) — logic fidelity, not perf |
| Multiple boards | Browser tabs or looped videos via a **file-based edge adapter** |
| Viewer | **Viewer web app** rendering the latest placement per board (e.g. chessground) — no move logic, no Lichess |
| Clock | Phone filming a real/app digital clock in-frame (running-side only) |

The sim validates **architecture, protocol, and CV logic**. TensorRT engine
builds and true multi-stream throughput are deferred to hardware. Keep models in
ONNX so the same artifact runs in sim and (after on-device engine build) on the
Jetson.

### 6.1 Bring-up / test-data plan
1. **YouTube OTB footage first.** `yt-dlp` static-camera, continuous board-cams
   (club/tournament livestreams — *not* produced broadcasts with cuts/overlays),
   fed through a **file-based edge adapter** to develop detection + per-frame
   estimation offline. Curated example streams are listed in **`youtube-streams.md`**
   (growing list); own-recorded footage is added as it becomes available (see §6.1
   step 2).
2. **Real setup next.** Smartphone browser as edge filming a physical board; **this
   computer** runs the Jetson-sim container *and* serves both web apps.
3. **Two web apps, no Lichess.** Edge (capture + rectify + forward) and viewer
   (live placement rendering).

---

## 7. Milestones

- **M0 — Walking skeleton.** WebSocket gateway, position-stream schema, Docker
  compose for edge/central/viewer, a file-based edge adapter replaying a clip
  through a stub estimator (e.g. returns the rectified crop's naive guess) → the
  viewer renders a live placement. Proves the end-to-end pipe.
- **M1 — Single-frame position estimation.** Board detect + rectify on the edge;
  DINOv2 + offline-trained head on the central node → piece placement + per-square
  confidences for a still frame. Evaluate per-square / per-board accuracy on real
  photos.
- **M2 — Live position stream over a full game (★ headline demo).** Adaptive
  frame forwarding + stability; central streams positions continuously over a
  **full recorded game**, angled view; positions stay sensible through hands and
  transitions (low confidence where occluded). **No move logic** — the viewer just
  shows the evolving placement.
- **M3 — Transport & timestamps.** Capture-time stamps, seq, retained latest
  position, recorder, status events; chrony/NTP (or simulated offset).
- **M4 — Multi-board on one central node.** N simulated edges → one container;
  shared estimator pool; measure boards-per-container.
- **M5 — Calibration round (optional path).** Position-agnostic few-shot centroids
  from any labeled position; measure the accuracy lift vs the default head.
- **M6 — Clock (which-side-running).** Two-ROI differencing streamed as sensor
  state.
- **M7 — Hardening & (optional) hardware port.** Robustness, ONNX→TensorRT on a
  real Orin Nano / Pi 5 + Hailo; throughput benchmarks.

---

## 8. Work packages

**WP-A · Edge: board detection & rectification**
- A1 board detection (Canny→Hough→cluster→RANSAC homography) on the edge
- A2 rectify to canonical crop; cache homography; 4-corner manual fallback
- A3 ROI-drift / camera-bump detection & re-rectify

**WP-B · Central: per-frame recognition**
- B1 64-square cropping from the rectified board
- B2 DINOv2 embedding + classifier head (ONNX); offline training on public data
- B3 per-square + position confidence; honest occlusion flags
- B4 burst voting (optional); eval harness on real photos; accuracy dashboard

**WP-C · Edge: capture & forwarding (virtual Pi)**
- C1 capture + camera abstraction (browser getUserMedia / video file)
- C2 motion-meter → stability score (low-res differencing)
- C3 adaptive frame-forwarder (keepalive + burst)
- C4 capture-time timestamping + rectified-crop JPEG encode + publisher

**WP-D · Transport & stream**
- D1 WebSocket frame ingress (binary + header) with backpressure
- D2 position-stream egress (WebSocket/SSE), retained latest per board
- D3 position-event schema + versioning + validation
- D4 time sync (chrony + local NTP; simulated offset in sim)

**WP-E · Calibration (optional path)**
- E1 calibration UI: operator supplies any known labeled position
- E2 per-set centroid builder from labeled square embeddings
- E3 blend/switch logic (default head ↔ per-set centroids); accuracy eval

**WP-F · Clock**
- F1 clock ROI localization (within board view or separate)
- F2 which-side-running via temporal differencing → sensor event
- F3 *(later)* digital 7-seg OCR (perspective-correct → digit CNN / ssocr)

**WP-G · Central node orchestration**
- G1 gateway + per-board inbox queue + pre-filter
- G2 shared estimator worker pool; async I/O
- G3 Docker image (amd64, ONNX Runtime; optional CUDA EP)
- G4 scaling tests: boards-per-container; latency budget

**WP-H · Consumers (POC scope)**
- H1 **viewer web app**: live placement rendering per board (chessground),
  confidence/occlusion/stability/clock indicators
- H2 recorder (append-only NDJSON; replay tool)
- H3 *(optional, later, application-layer)* reference move-reconstructor +
  PGN/Lichess export — explicitly demonstrating the split, not part of the CV system

**WP-I · Cross-cutting**
- I1 sim harness: file-based edge adapter (`yt-dlp` clips → frames), browser edge
  app, multi-board orchestration
- I2 datasets: curate static-camera OTB clips (tracked in `youtube-streams.md`);
  collect own real-set footage; labeled calibration positions
- I3 metrics & observability (latency, per-square/per-board accuracy, confidence
  calibration, frame rates)
- I4 docs & runbook

---

## 9. Risks & open questions

**Top risks**
- **Per-frame recognition accuracy is now the binding constraint** (no in-CV
  legality/temporal crutch). Mitigated by calibration + burst voting + confidences;
  must be validated early on real footage (M1/M2). The bet is that good per-square
  confidences + dense streaming let the application layer recover game-level
  accuracy.
- **Confidence calibration.** The whole design leans on per-square confidences
  being *meaningful* (well-calibrated), so downstream smoothing can trust them.
  Needs explicit evaluation (reliability diagrams), not just accuracy.
- **Does `stability` earn its keep?** It may be redundant with the position stream
  itself. Keep it cheap and optional; revisit after M2.

**Resolved (2026-06-16)**
- **Placement encoding → full 64×13 distribution.** The canonical output is a full
  per-square class distribution; the 8×8 scalar grid and `placement` string are
  derived from it. Rationale in D2/§5.
- **Square indexing/orientation → fixed canonical + metadata.** Rectify to a
  deterministic frame set at setup, report it in an `orientation` field; the CV
  layer never infers "White's view." See §5.
- **Clock → embedded field (v1).** `clock.running` rides inside each position event
  as drafted. A separate clock topic (decoupling the ~1–2 Hz clock cadence from
  frame cadence) is noted as a possible v2 refinement.
- **Backbone → DINOv2-S (Apache).** DINOv3 rejected as default on licensing grounds
  (custom gated license + attribution clause) despite stronger dense features; kept
  behind an interface as a candidate if accuracy proves backbone-limited. See D6/§4.6.
