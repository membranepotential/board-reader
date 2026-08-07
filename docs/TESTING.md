# Testing board-reader

Two layers, both required by the project's "user-testable from the start" principle
(see [../CLAUDE.md](../CLAUDE.md)):

1. **Automated tests** — fast, run on every change.
2. **The live demo** — a human runs the whole pipe and *sees* it work.

---

## 1. Automated tests

```bash
uv sync                # once, to set up the workspace venv
make test              # the whole suite across all cores (should be green), ~10s
uv run pytest          # the same suite, serially (~23s)
uv run pytest -q       # quiet
uv run pytest packages/inference          # just one package
uv run pytest tests/test_end_to_end.py    # the real-network walking-skeleton test
```

What's covered today (M0 + M1 rectification):

| Area | File | Proves |
|---|---|---|
| L3 protocol | `packages/core/tests/test_protocol.py` | placement⇄distribution, validation, JSON roundtrip |
| Wire format | `packages/core/tests/test_wire.py` | JPEG + header frame roundtrip |
| Sim renderer + warp | `packages/core/tests/test_sim.py` | board images render; perspective warp + ground-truth quad |
| Frame sources | `packages/edge/tests/test_source.py` | video/dir/synthetic/repeat sources |
| Edge adapter | `packages/edge/tests/test_adapter.py` | frames → wire envelopes, throttle, **rectify + suppress** |
| **Rectification** | `packages/edge/tests/test_rectify.py` | **detect angled board, un-warp round-trip, manual corners, bump→recalibrating, blank→needs_setup** |
| **Square cropping** | `packages/inference/tests/test_squares.py` | **64 crops in protocol order, inset/resize** |
| **Embedder** | `packages/inference/tests/test_embed.py` | **lightweight embedder shape, L2-norm, distinct glyphs** |
| **Centroids** | `packages/inference/tests/test_centroid.py` | **few-shot calibration, self-consistency, generalisation, absent classes, save/load** |
| Estimator | `packages/inference/tests/test_estimator.py` | stub emits valid events; **CentroidEstimator reads pixels** |
| Hub | `packages/inference/tests/test_hub.py` | retained-latest, fan-out, overflow |
| Gateway | `packages/inference/tests/test_server.py` | ingest → viewer over WS (in-process) |
| Viewer | `packages/viewer/tests/test_viewer.py` | UI served, WS URL injected |
| **Layer split** | `tests/test_layer_separation.py` | edge/inference never import each other |
| **End-to-end** | `tests/test_end_to_end.py` | real uvicorn + edge client + viewer WS |

---

## 2. The live demo (what a user runs)

### Option A — Docker (canonical, one command)

```bash
docker compose up --build
```

Then open **http://localhost:8080**. You should see:

- the connection chip turn **“live”** (green),
- a board card for **A1** rendering the **start position**,
- **confidence ≈ 0.95** and a **stability** bar,
- `seq` ticking up about **twice a second** (the synthetic edge feeds at 2 fps).

This runs three containers: `edge` (synthetic feed) → `inference` (estimator +
gateway) → `viewer` (UI). The browser loads the UI from `viewer` but connects for
*data* straight to `inference` on `localhost:8000`.

Stop it with `docker compose down`.

### Option B — Local processes (no Docker)

Three terminals, all `uv run`:

```bash
uv run board-reader-inference serve --port 8000     # 1: gateway
uv run board-reader-viewer --port 8080              # 2: UI  -> open http://localhost:8080
uv run board-reader-edge feed                       # 3: synthetic feed (30 frames)
```

`feed` exits after 30 frames by default; use `--synthetic 0` to stream forever, or
`--throttle --fps 2` to pace it like the Docker demo.

### Option C — Feed real footage

Point the edge at a recorded OTB clip (see `docs/youtube-streams.md` for sources;
grab one with `yt-dlp`) or a directory of stills:

```bash
uv run board-reader-edge feed --video path/to/clip.mp4 --throttle --fps 5
uv run board-reader-edge feed --images path/to/frames/
```

> Note: with no `--centroids`, the estimator is still the M0 **stub** (fixed start
> position, ignores pixels). The M1 recognition path is opt-in via calibration — see
> Option E. The default M1 embedder is a lightweight stand-in that works on the
> *synthetic* renderer; the real **DINOv2** backbone (for real angled photos) plugs
> into the same `Embedder` seam next.

### Option D — See board detection + rectification (M1, WP-A)

The edge now detects the board and un-warps the angled view to a canonical top-down
crop before forwarding. Two ways to see it:

**One-shot (most observable).** Make an angled board image and rectify it to a file
you can open:

```bash
# write an angled synthetic board to /tmp/angled.png
uv run python -c "import cv2; from board_reader_core.sim import render_board, simulate_angled_view; cv2.imwrite('/tmp/angled.png', simulate_angled_view(render_board())[0])"

# detect + un-warp -> /tmp/rect.png (open it: a clean top-down board)
uv run board-reader-edge rectify /tmp/angled.png -o /tmp/rect.png
```

`rectify` prints the detected corners (TL,TR,BR,BL = a8,h8,h1,a1). On a cluttered
real photo where auto-detection fails, supply the operator corner-tap directly:

```bash
uv run board-reader-edge rectify photo.jpg -o crop.png \
  --corners x0,y0,x1,y1,x2,y2,x3,y3      # a8,h8,h1,a1
```

**In the live pipe.** Feed a *simulated angled* synthetic board and have the edge
rectify each frame before forwarding:

```bash
uv run board-reader-edge feed --angled --rectify --throttle --fps 2
```

With `--rectify`, the edge forwards canonical crops, and frames it can't localise (a
bump, or no board) are **suppressed** rather than streamed as confidently-wrong
geometry (DESIGN §4.4.1).

### Option E — Live recognition (M1, centroid few-shot)

Now the viewer shows what's *actually on the board*, not a fixed start position. The
recognizer is calibrated few-shot from one labeled position (DESIGN §4.6):

```bash
# 1. Calibrate centroids from one labeled (rectified) board — the start position has
#    all 13 classes, so one frame calibrates the whole vocabulary.
uv run python -c "import cv2; from board_reader_core.sim import render_board; cv2.imwrite('/tmp/start.png', render_board())"
uv run board-reader-inference calibrate --image /tmp/start.png -o /tmp/centroids.npz
#    -> "calibrated 13/13 class centroids"

# 2. Serve the gateway with the M1 estimator (instead of the stub):
uv run board-reader-inference serve --centroids /tmp/centroids.npz

# 3. Viewer (terminal 2) + feed any position (terminal 3):
uv run board-reader-viewer --port 8080            # open http://localhost:8080
uv run board-reader-edge feed --synthetic 0 --throttle \
  --placement "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
```

The board card now renders the **fed mid-game position** with per-square confidences —
recognition is reading the pixels. Change `--placement` and the viewer follows.

> The default `lite` embedder is tuned for the synthetic renderer. For real angled
> footage, first `board-reader-edge rectify` the frame, then calibrate + serve with the
> **DINOv2** backbone (`--backbone dinov2`). That needs the converted model at
> `models/dinov2_small.onnx` — build it once from official Meta weights:
>
> ```bash
> cd ../dinov2-export && uv run python ../board-reader/tools/export_dinov2_onnx.py \
>   --out ../board-reader/models/dinov2_small.onnx     # see tools/README.md
> uv run board-reader-inference calibrate --image rect.png --backbone dinov2 -o c.npz
> ```
>
> The centroids file records its backbone, so `serve --centroids c.npz` auto-selects the
> matching embedder.

### Option F — Phone snapshot → Lichess

Photograph a real board with your phone, get the position + a Lichess link. The
snapshot page is an application-layer consumer (DESIGN WP-H3): the CV side still
only produces placement + confidences; side-to-move, castling guess, and the
Lichess URL are assembled in the page.

```bash
# 1. Serve inference with real recognition (a DINOv2 calibration file for your set —
#    centroids or the kNN exemplar bank the CLI/session calibration now builds):
uv run board-reader-inference serve --host 0.0.0.0 --port 8000 \
  --centroids models/centroids_dinov2.npz

# 2. Serve the viewer on the LAN:
uv run board-reader-viewer --host 0.0.0.0 --port 8080

# 3. On the phone (same Wi-Fi), open  http://<this-machine's-LAN-IP>:8080/snapshot
```

Flow: take/pick a photo → the page **auto-detects the board**, places four
labeled corner handles and reads the position, with no further taps. The photo
stays on screen next to the board it produced, at the same size; check that
**a8** sits on the board's a8 corner (**Rotate labels** if not — geometry can't
know the orientation), drag any handle to fine-tune. The estimated board renders
with low-confidence squares outlined red, plus **Open in Lichess analysis /
editor** links (side-to-move toggle on the page). If auto-detection fails, stage
2 says so and the page falls back to tapping the four corners yourself in the
prompted order **a8, h8, h1, a1** (press-drag-release; a loupe magnifies under
your finger), then *Read position*. Reads send the **photo + corners** — the
server crops each square with headroom from the original view, so pieces are
seen whole rather than smeared by the top-down warp.

**The page is the diagnostic surface for the reader** (viewer aspect). Under the
photo and the board, the seven stages of the read are listed in the order they
run — *capture · board detection · rectify & geometry · square sampling ·
classification · orientation · position output*. Each row always shows its state
and headline numbers; expanding it shows the full evidence, and nothing is
behind an opt-in toggle (`?debug=1` is always requested):

1. **Capture** — posted size, JPEG size, mean luma against the server's
   `LOW_LIGHT_MEAN` (55).
2. **Board detection** — auto vs hand-placed, `/detect` latency, the four corner
   coordinates, whether server auto-rotation is still in play.
3. **Rectify & geometry** — the rectified preview plus projected aspect against
   `LOW_ANGLE_HW` (0.62), edge lengths, square pitch.
4. **Square sampling** — the sampling boxes drawn on the photo and the 64
   model-input crops as the classifier received them.
5. **Classification** — mean/weakest per-square confidence, an 8×8 confidence
   map (click a cell to inspect that square), the least-confident squares, and
   any `autocalibrated` / `bank_refreshed` / `warnings` the server volunteered.
6. **Orientation** — whether `autorotate=1` was sent and what rotation came back.
7. **Position output** — the FEN, side-to-move, Lichess links, the corrections
   made by hand, a material census (two kings, ≤8 pawns), and the raw event JSON.

**Correcting a square:** click it once to inspect — the switcher beside the
board shows the crop the model saw, what it read, the full 13-class
distribution, and a White/Black grid to assign from. Click the same square again
to toggle through its top-3 readings (the fast fix). On a desktop the arrow keys
move the selection and FEN letters assign (`K` white king, `k` black king, `.`
empty, `esc` closes). Inspecting never changes a square; only assigning does.

**Recent positions** at the bottom logs every read of the session. Selecting one
restores the whole pipeline view — that photo, its corners, its crops, its
distributions and its own corrections — so two reads can be compared without
re-shooting either. It lives in memory only and is gone on reload.

Notes:
- The page auto-targets inference on the same host it was loaded from, so no
  extra config is needed on the LAN. Override with the `BR_INFERENCE_HTTP` env
  var on the viewer if inference runs elsewhere.
- `POST /estimate` on the inference node accepts a **rectified** top-down crop
  (JPEG body) and returns the position event JSON — handy for scripting:
  `curl -s --data-binary @crop.jpg -H 'Content-Type: image/jpeg' localhost:8000/estimate`
- Without `--centroids` you get the M0 stub (fixed start position) — fine for
  testing the plumbing, useless for real photos.
- The page's **Calibrate (start position)** button hits `POST /calibrate`, which
  now builds a kNN exemplar bank in the photo's own lighting (the
  leave-one-photo-out-validated replacement for centroid averaging, DESIGN
  §4.6) and hot-swaps it in. Calibrating again — same set, different lighting,
  or a later position — **adds** that photo's exemplars to the running bank
  rather than throwing the old ones away, so repeated calibration only helps.
- Set `BR_SAVE_UPLOADS=data/uploads` on `serve` to keep every posted photo (+
  its corners/placement metadata) — real-world shots become labeled eval data.

### Option G — Zero-setup: occupancy + start-position auto-calibration

No calibration file for the set at hand? Serve with an **occupancy bank**
instead (built from ChessReD by `tools/build_exemplar_bank_chessred.py`;
CC BY-NC-SA — local use only, never committed):

```bash
BR_OCCUPANCY_BANK=models/exemplars_chessred.npz BR_SAVE_UPLOADS=data/uploads \
BR_SESSION_BANK=models/exemplars_session.npz \
  uv run board-reader-inference serve --host 0.0.0.0 --port 8000
```

`BR_SESSION_BANK` opts into persisting/restoring calibration across restarts;
without it every calibration lives in memory only (nothing is ever written —
deliberate, so tests and ad-hoc runs can never clobber a real bank).

Manual test, on the phone (`/snapshot` as in Option F):

1. Photograph any **mid-game** position of a set the server has never seen →
   the response is occupancy-only (which squares are occupied, no piece types)
   plus the hint to show the starting position. Nothing is published to the hub.
2. Photograph the **starting position** → the server detects its occupancy
   pattern (any rotation; White's side by piece brightness), silently
   calibrates piece types from that very photo, and answers with a full
   position event (response carries an `autocalibrated` block). Verify the
   rendered board is the start position.
3. Photograph the mid-game position again → full recognition, no calibration
   button ever pressed.
4. Restart the server with the same env → recognition still works: the session
   bank (`models/exemplars_session.npz`) is restored on startup whenever
   auto-calibration is enabled. (Delete/move that file to rehearse the
   cold-start flow again.)

To also demo the true cold start, temporarily move `models/exemplars_session.npz`
away before step 1.

Two further behaviors to verify on any `/estimate`:
- **Low-light warning**: a very dark photo (mean brightness < 55/255) makes the
  response carry `"warnings": ["low_light"]` and the page show a dimness hint.
- **Calibrated self-refresh**: photographing the *starting position* in lighting
  the bank doesn't cover yet (read confidence below 0.8) makes the response
  carry `"bank_refreshed": {...}` — the bank grew from that photo; subsequent
  reads in that lighting improve. Mid-game photos never trigger it.

### Option H — Trained zero-shot model + calibration booster (the default serve)

The production configuration serves the multi-style trained classifier with the
per-set bank as an ensemble booster (DESIGN §4.6):

```bash
BR_MODEL=models/run_e/model.onnx BR_SAVE_UPLOADS=data/uploads \
BR_SESSION_BANK=models/exemplars_session.npz \
  uv run board-reader-inference serve --host 0.0.0.0 --port 8000
```

`models/run_e/model.onnx` is the deployed model — the same path
[compose.yaml](../compose.yaml) and [runbook.md](runbook.md) use; take it from
there rather than from memory. The superseded
`models/piece_classifier_multistyle_v1.onnx` is still on disk and still loads,
but it has no `.onnx.meta.json` sidecar, so serving it silently applies v1's
short-crop geometry to whatever weights you point at — the decapitation
signature (kings on the back rank read as bishops). The server now warns on
stderr when a sidecar is missing; do not ignore that line.

Manual test, on the phone (`/snapshot` as in Option F):

1. With `models/exemplars_session.npz` moved away, photograph a **mid-game**
   position → a full position event from the trained model alone (zero-shot; no
   occupancy-only mode, no calibration needed).
2. Photograph the **starting position** → the response carries an
   `autocalibrated` block: the trained path bootstrapped its own booster bank
   from that photo. Subsequent reads answer through the 50/50 ensemble (better
   in bank-covered conditions, never worse elsewhere). With `BR_SESSION_BANK`
   set, the bank persists and the ensemble is restored on restart.
3. Tapping **Calibrate** at any point *adds* to the booster bank — it never
   replaces the trained model. Calibrate labels are orientation-guarded: if the
   corner labeling disagrees with the trained model's own reading of the photo
   by a rotation, the labels are rotated into agreement (response carries
   `"label_rotation"`), so a calibrate over auto-detected (arbitrarily
   oriented) corners can never poison the bank with color-swapped exemplars.
   If a bank is suspected poisoned anyway:
   `CUDA_VISIBLE_DEVICES= uv run python tools/rebuild_session_bank.py` rebuilds
   it from all retained calibrate uploads with the same verification.
4. **Orientation auto-correction**: photograph the board with the camera on
   Black's side and accept the auto-detected corners (don't touch the handles) —
   the read comes back with White at the bottom anyway and the response carries
   `"autorotated": 180` (the UI shows a banner). Hand-placing or relabeling
   corners disables this — the operator's orientation is authoritative.
5. **Sidecar labeling** (`BR_SAVE_UPLOADS`): each retained `*-estimate.json`
   now records the predicted `placement` (and `autorotated` when applied), so
   every field shot is ready-made eval data.
6. **Flat-angle warning**: photograph the board with the phone nearly at board
   level (projected quad aspect < 0.62) — the response carries
   `"warnings": ["very_low_angle"]` and the page tells you to raise the phone.
   Recognition is not trustworthy at such angles (yet).
7. **Calibrate rejection**: tap Calibrate on a photo that does NOT show the
   claimed position (e.g. a mid-game board with the default start placement) —
   the server answers 422 with the reason (fewer than 60% of claimed piece
   squares agree with its own reading) and the bank is untouched.

### Option I — Confirm ground truth, then score the reader

The two halves of measuring whether the snapshot app actually works: collect
labels through normal use, then run the deployed stack against them.

**Collecting a label** (needs `BR_SAVE_UPLOADS` — Option H's serve line has it):

1. Photograph a position and read it as usual.
2. In **7. Position output**, fix any wrong squares (tap a square, pick the
   right piece), then press **Confirm as ground truth**. The button also appears
   when nothing is wrong — confirming a flawless read is exactly as much a label
   as fixing four squares, and the perfect-board rate is made of both.
3. A `data/uploads/<stem>.label.json` appears next to the photo, holding the
   position, the corners, and what the server predicted at the time. Editing a
   square after saving re-arms the button — a label describes one exact position.
4. The label is stored in the frame the **corners** define: if the read was
   `autorotated`, the turn is undone first, so a later model whose orientation
   guess differs is still scored against the same truth.

> Labels are scoring data only. Nothing feeds them back into the calibration
> bank — that is the contamination they exist to measure.

**Scoring:**

```bash
uv run board-reader-inference eval                    # data/eval + data/uploads
BR_MODEL=models/run_e/model.onnx uv run board-reader-inference eval   # one member
uv run board-reader-inference eval --json report.json --autorotate
```

The harness builds the estimator from the environment exactly as `serve` does,
prints that configuration above the report, and leads with **perfect boards**,
not per-square accuracy — 94.5% per square is ~3.5 wrong squares on a board, and
a position with 3 wrong squares is not a position. It then breaks errors into
hallucination / missed / colour / type, and prints a reliability diagram: the
gap between claimed confidence and observed accuracy is what says whether a 0.97
means anything.

If a calibration bank is active the report warns that boards the bank was built
from are not held out. Take a number carrying that warning as an upper bound.

---

## Quick health checks

```bash
curl http://localhost:8000/healthz            # {"status":"ok",...}
curl http://localhost:8000/boards/A1          # latest position event JSON for A1
```

## Troubleshooting

- **Viewer says “disconnected — retrying”**: `inference` isn't reachable at the URL
  baked into the page. In Docker it's `ws://localhost:8000/viewer`; override with the
  `BR_INFERENCE_WS` env var on the `viewer` service.
- **No board appears**: check the edge is sending — `docker compose logs edge` — and
  that `curl localhost:8000/boards/A1` returns an event (not `404`).
- **`docker compose` build is slow the first time**: it downloads the Python base
  image and OpenCV wheels; subsequent builds are cached.
