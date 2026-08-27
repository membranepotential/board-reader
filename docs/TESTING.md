# Testing board-reader

Two layers, both required by the project's "user-testable from the start" principle
(see [../CLAUDE.md](../CLAUDE.md)):

1. **Automated tests** — fast, run on every change.
2. **The live demo** — a human runs the whole pipe and *sees* it work.

---

## 0. Read this first (the 60-second version)

The suite is **fast — ~12 s for all 783 tests**. If testing is costing you more than
that, you are doing one of the things in this section. Read it before you run
anything; do not read the remaining 600 lines unless you are adding a test.

**The only two commands you normally need:**

```bash
make test                                  # everything, ~12s. The gate before a commit.
uv run pytest packages/inference/tests/test_server.py -q    # one file, while iterating
```

| Want | Run | Cost (measured, idle machine) |
|---|---|---|
| The gate (before committing) | `make test` | **12 s** — all 783 |
| One file / one test | `uv run pytest <path>[::<test>] -q` | 2–7 s |
| One package | `uv run pytest packages/<pkg> -q` | core 5 s · edge 2 s · viewer 6 s · **inference 20 s** |
| Recognition changes only | `make test-gpu` | 2 s, **needs an idle GPU** |

Note the trap in row 3: `packages/inference` is 371 of the 783 tests and runs
*serially*, so scoping to it (20 s) is **slower than running everything** with
`make test` (12 s). Narrow to a file, or just run the gate.

**Rules that save the most time:**

- **Don't run the suite when nothing it covers changed.** A docs, comment,
  `Makefile` or `STATUS.md` edit is not a reason to spend a run (CLAUDE.md §2).
- **Don't add `-n auto` to a focused run.** Worker startup costs more than the tests
  you selected; `make test` already parallelises the full suite. Plain `uv run
  pytest <path>` is the fast path for one file.
- **Don't run `uv run pytest` bare to check your work** — that is the whole suite
  *serially*, ~23 s, and it re-enables the GPU tests. `make test` is strictly better.
- **Never run two suites at once**, or a suite alongside a training run. The tests
  are CPU-saturating by design (`-n auto`); two at once is how the flakes below get
  triggered, and it makes both runs slower than running them in sequence.
- **`make test` is green or the code is broken.** It is not "usually green". If it
  is red, that is a real finding — investigate it, don't re-run and hope.

**If something goes wrong:**

| Symptom | What it means | Do this |
|---|---|---|
| Suite hangs with no output | A test is wedged | Wait 120 s: `faulthandler_timeout` dumps every thread's stack and names the test. Then read the stack — do not just re-run. |
| Port already in use / e2e weirdness | A previous session left `make dev` running | `make stop` — it now names the pids it killed, escalates to SIGKILL, and exits non-zero if anything survived |
| Node aborts mid-stream (status 134), `BFCArena`/`CL_MEM_OBJECT_ALLOCATION_FAILURE` | The GPU is full — often a leaked server still holding a CUDA context | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, then `make stop`. A released port does **not** mean the process is gone |
| GPU test fails (`BFCArena`, OOM) | The GPU is busy, not a code bug | Re-run `make test-gpu` when idle. `make test` excludes these for exactly this reason. |
| An external monitor blanks while the node runs | The card drives your display too, and X lost the VRAM for its scanout surface | `make dev-cpu` instead. Recover the screen with `xrandr --output <out> --mode 1920x1080` and back — see [the runbook](runbook.md#when-the-card-is-also-driving-your-screens) |
| A `/boards/{id}` 404 or a WS read that never returns | You asserted on an estimate before it landed | See **the ingest rule** below |

**The ingest rule (the one gotcha worth knowing).** `WS /ingest` acks a frame when
it is **received**, not when it is **estimated** — the estimate happens later on a
per-board worker task. So a test that ingests a frame and then asserts on a
*published* position (`GET /boards/{id}`, a `/viewer` event, `frames_estimated`)
must wait for that worker. Use the fixtures in
[`packages/inference/tests/conftest.py`](../packages/inference/tests/conftest.py):

```python
def test_something(gateway, drain_ingest):
    client = gateway(create_app(...))  # NOT a bare TestClient(...)
    with client.websocket_connect("/ingest") as ws:
        ...  # send frames, read acks
    drain_ingest(client)  # deterministic; no sleep, no polling
    assert client.get("/boards/A1").status_code == 200
```

A bare `TestClient` gets a fresh event loop per call, so the worker dies with the
websocket and the estimate is **lost** — about 1 % of the time under load. That
single bug produced both a flaky 404 *and* a suite that hung for a quarter of an
hour (a `/viewer` read waiting forever for an event nobody would publish). Don't
reintroduce it by reaching for `TestClient` + `time.sleep`. If the test only checks
what the handler wrote *before* acking — `/boards/{id}/frame`, the ingress
recording — a plain `TestClient` is correct and cheaper. `conftest.py` has the full
reasoning.

---

## 1. Automated tests

```bash
uv sync                # once, to set up the workspace venv
make test              # the suite across all cores, minus GPU tests (green), ~12s
make test-gpu          # the 2 GPU tests alone — an idle GPU, or they flake
make test-all          # everything, GPU included
uv run pytest          # the whole suite, serially (~23s)
uv run pytest -q       # quiet
uv run pytest packages/inference          # just one package
uv run pytest tests/test_end_to_end.py    # the real-network walking-skeleton test
```

> [!note] Why `make test` is not the whole suite
> Two tests build a real ONNX `InferenceSession` and take the CUDA execution
> provider. On an idle machine they pass; on a busy one — a training run, or a
> second agent session running its own suite — session creation dies with
> `BFCArena Failed to allocate memory`, and the pressure has been observed
> taking unrelated OpenCV allocations down with it. They are marked `gpu` and
> deselected by default, so a red suite means the *code* is red. Run them
> deliberately with `make test-gpu` before trusting a recognition change.

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
| **Recording format** | `packages/core/tests/test_recording.py` | **byte-exact round trip, corners survive, truncated session still replays, size cap** |
| **Ingress recording** | `packages/inference/tests/test_recording_ingest.py` | **every `/ingest` frame lands on disk (incl. pre-filter drops); off unless asked; a write failure never breaks the stream** |
| **Replay** | `packages/edge/tests/test_recording_replay.py` | **replays the recorded bytes without re-encoding, paces by capture time, refuses to reprocess** |
| **Position stream consumer** | `packages/inference/tests/test_positions.py` | **transitions of a real move, first-sighting ≠ unchanged, per-board diffing, NDJSON capture rebuilds the identical event, `--changes-only` drops keepalives** |
| Hub | `packages/inference/tests/test_hub.py` | retained-latest, fan-out, overflow |
| Gateway | `packages/inference/tests/test_server.py` | ingest → viewer over WS (in-process) |
| Viewer | `packages/viewer/tests/test_viewer.py` | UI served, WS URL injected |
| **Layer split** | `tests/test_layer_separation.py` | edge/inference never import each other |
| **End-to-end** | `tests/test_end_to_end.py` | real uvicorn + edge client + viewer WS |

### Frontend (`packages/viewer/ui`)

A second, independent test layer for the SvelteKit UI — a frontend-only change is
not a reason to run `make test`, and a Python-only change is not a reason to run
these:

```bash
make test-front                    # both layers, from the repo root — Vitest then Playwright

cd packages/viewer/ui              # or, one at a time, from the package
pnpm install --frozen-lockfile     # once; mirrors `uv sync --frozen`'s discipline
pnpm test                          # Vitest — src/lib/**, node env, ~1s, 333 cases today
pnpm check                         # svelte-kit sync + svelte-check, strict TS
pnpm test:e2e                      # Playwright — 16 specs in a real Chromium, ~15s
```

`make test-ui` and `make test-e2e` are the two halves of `make test-front`; each runs
`pnpm install --frozen-lockfile` first, and `test-e2e` also runs `playwright install
chromium` (a one-off on a fresh checkout — Playwright pins its own browser build).

`pnpm test` is the tested half of what the three hand-written pages used to
triplicate: FEN⇄grid conversion (`lib/board/fen.ts`), the wire protocol parser
(`lib/protocol.ts`), the board-plane tracker (`lib/geometry/tracker.ts`, pinned by
the same checks that used to live in the deleted `tracker_checks.mjs`), the motion
meter and the `/ws/ingest` framing (`lib/edge/`) — all plain TypeScript, run in
`node`, no DOM.

`pnpm test:e2e` drives the three pages in a real Chromium (`tests/*.spec.ts`), against
the **built** app — `vite build` then `vite preview`, because the bundle is what the
Python viewer serves. There is no inference node behind it: every `/api/…` call and the
`/ws/viewer` socket is stubbed per test (`tests/helpers/node.ts`), so these need neither
a GPU nor a model. They cover what neither other layer can reach — a chessground board
that laid out with no pieces on it, a rail button 31 px tall on a phone, an
`@media (orientation: portrait)` rule that never applies, a preview that fetches while
collapsed. Each of those has shipped here at least once.

> [!note] Chromium only, and touch is emulated
> Playwright drives Chromium with mouse events synthesised for taps. iOS Safari — the
> actual target device for `/camera` and `/snapshot` — is still only testable by hand;
> see §2's phone walkthrough.

`pnpm dev` runs the Vite dev server with `/api` and `/ws` proxied to a *running*
Python viewer (`BR_VIEWER_ORIGIN`, default `http://127.0.0.1:8080` — see
`vite.config.ts`), so the dev loop exercises the same relay production uses rather
than mocking it:

```bash
uv run board-reader-inference serve --port 8000 &   # terminal 1: gateway
uv run board-reader-viewer --port 8080 &             # terminal 1: viewer, relays to it
cd packages/viewer/ui && pnpm dev                    # terminal 2: frontend -> http://localhost:5173
```

`make ui` (`pnpm build` under the hood) is the production build instead of the dev
server: it prerenders all three routes straight into
`packages/viewer/src/board_reader_viewer/web/` — see the note at the top of §2.

---

## 2. The live demo (what a user runs)

> [!note] Build the UI once
> `board-reader-viewer` serves a prerendered SvelteKit build and refuses to start
> without one (`packages/viewer/src/board_reader_viewer/web/` is generated and
> gitignored, so a fresh checkout has none). Docker's own `ui-build` stage handles
> Option A; every local-process option below needs `make ui` run once first (and
> again after any change under `packages/viewer/ui/`).

### Option A — Docker (canonical, one command): the M2 live game replay

```bash
docker compose up --build
```

Then open **http://localhost:8080**. The edge replays a **synthetic recorded
game** (looped forever) and forwards it adaptively (DESIGN D4), so you should see:

- the connection chip turn **“live”** (green), a board card for **A1** with the
  position drawn by **chessground** (real pieces, coordinates) — the same board
  the snapshot page uses,
- the position **evolving move by move** — each changed square ringed **green**,
  fading pale and gone within a second,
- the rhythm of an adaptive stream: **~1 ev/s at rest** (keepalive), jumping to
  **~10–12 ev/s** with a pulsing **“moving”** chip whenever the synthetic hand
  reaches over the board — the **stability** bar dips at the same moments,
- squares under the hand ringed **yellow** (low per-square confidence — honest
  occlusion, not a guess); the legend under the board names both rings,
- the **preview** tab on the board's left sliding open onto the **latest frame the
  node ingested**, refreshing as events arrive — it reads “no frame yet” until one
  has been forwarded, and closing it stops fetching,
- the **age** readout ticking and the card fading to **stale** if you stop the
  edge container (`docker compose stop edge`) — and recovering when it returns.

The **calibrate** button under each board rebuilds the node's piece bank from the
live frame, for the set and lighting in front of the camera. Set the board to the
**starting position**, let it settle, then press it: the card reports how many of
the 13 classes were calibrated and how big the bank now is. It is disabled (hover
it for the reason) while the card is stale, mid-motion, or not reading `ok`, and a
frame that does not actually show the starting position is **refused** by the node
with its reason shown on the card — calibrating from the wrong frame would poison
the bank.

The **rotate** button next to it turns the card's board picture in 90° steps —
purely cosmetic (CSS transform on the rendered card; the placement stays
white's-view a8-first on the wire), for matching the screen to however you're
standing relative to the physical board. At 90°/270° the coordinates turn
sideways with the picture, like turning a printed diagram. Manual check: clicking
cycles 0→90→180→270→0, a second board's card is unaffected, and the choice
survives a reload (persisted per board id in localStorage).

This runs three containers: `edge` (game feed) → `inference` (estimator +
gateway) → `viewer` (UI). The browser loads the UI from `viewer` but connects for
*data* straight to `inference` on `localhost:8000`.

Stop it with `docker compose down`.

The same demo without Docker (gateway + viewer from Option B, then):

```bash
uv run board-reader-edge feed --game --adaptive --loop --throttle
```

`--game` is the synthetic recorded game; `--adaptive` enables the
keepalive/burst forwarder (`--keepalive-fps`/`--burst-fps` to tune); `--loop`
replays forever with `seq`/`capture_ts` continuing monotonically; `--throttle`
paces output to the stream's own capture time.

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
uv run board-reader-edge feed --video path/to/clip.mp4 --throttle --adaptive
uv run board-reader-edge feed --images path/to/frames/
```

For an angled clip, add rectification — `--corners x0,y0,x1,y1,x2,y2,x3,y3`
(the operator corner-tap: a8,h8,h1,a1 in pixels, declares geometry *and*
orientation) or `--rectify` for auto-detection (re-detects on a ~1 s cadence,
crops intervening frames against the cached homography).

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
- The page reaches inference through the viewer's `/api` relay, on whatever origin
  it was loaded from, so no extra config is needed on a LAN, over https, or behind
  a tunnel. `BR_INFERENCE_HTTP` on the viewer is what points that relay at the node
  when inference runs elsewhere.
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

**Two models instead of one (the cascade).** `BR_MODEL_B` names a second crop
classifier trained on a *different* crop geometry and turns the pair into a
confidence-gated cascade: every square is read by `BR_MODEL`, and only the
squares it was unsure about (max probability below `BR_CASCADE_TAU`, default
`0.95`) are re-read by `BR_MODEL_B` — cropped with **its own** sidecar geometry —
and the two distributions averaged 50/50.

```bash
BR_MODEL=models/run_e/model.onnx \
BR_MODEL_B=scratch/runs/run_f/model.onnx \
  uv run board-reader-inference serve --host 0.0.0.0 --port 8000
```

Measured 2026-08-11 on 60 boards: `user_photos` 0.938 → 0.956,
`chesscog_transfer` 0.819 → 0.861, OVERALL 0.696 → 0.718, at ~1.3× the single
model's cost because the second model sees under 30% of squares. Two things to
check by hand:

- `GET /healthz` grows a `cascade` block naming both models and counting
  `squares` / `deferred` / `deferred_fraction`. After a handful of real photos
  the fraction should sit near 0.3; if it is near 1.0 the first model is
  unconfident on everything (wrong geometry? wrong model?), and if it is 0.0 the
  second model is never being consulted.
- A `BR_MODEL_B` that cannot load, or that has no `.onnx.meta.json` sidecar, is a
  **hard startup failure**, not a warning — unlike `BR_MODEL`, whose fresh-clone
  fallback exists so `docker compose up` works without `models/`. Asking for the
  cascade and silently getting one model would only ever show up much later as
  "the ensemble didn't help".

`run_f` is deployed at `models/run_f/` and `compose.yaml` sets both members, so
the Docker demo runs the cascade whenever `models/` is populated. A fresh clone
(neither model file present) degrades to the stub exactly as `BR_MODEL` alone
always has; only a *partial* pair is a hard startup error.

**A model tuned for one board (the calibration average).** `BR_MODEL_TUNED` names
a fine-tune of `BR_MODEL` on the operator's own board, produced offline by
`tools/train/calibrate_finetune.py`. It is a different trade from the cascade:
the cascade buys accuracy on *any* board, this buys more accuracy on *one*.

Producing it needs only calibrate frames already sitting in `data/uploads` — a
calibrate sidecar carries the corners and means "this board is in the start
position", which is where the labels come from for free:

```bash
tools/train/calibrate_finetune.sh \
  --frames ../../data/uploads/20260717-232704-0011-calibrate.jpg \
           ../../data/uploads/20260718-155204-0010-calibrate.jpg \
           ../../data/uploads/20260806-232827-0008-calibrate.jpg \
  --out ../../scratch/tuned/user_board          # add --dry-run to just see the sessions
```

The tool **refuses to run on frames from fewer than three capture sessions** (it
clusters them by sidecar `ts`; a gap over an hour starts a new session). That is
not caution, it is the measurement: 3 frames from 1 sitting scored 0.932 on
`user_photos`, *below* the un-tuned 0.938, while 3 frames from 3 sittings scored
0.968. Condition diversity is the ingredient, not frame count. `--force`
overrides and says loudly that the result is outside every measured
configuration. Expect ~60 s on the GPU, and note that the onnxscript traceback
printed during export is expected noise — the success signal is the `round-trip
OK` line under it.

```bash
BR_MODEL=models/run_e/model.onnx \
BR_MODEL_TUNED=scratch/tuned/user_board/tuned.onnx \
  uv run board-reader-inference serve --host 0.0.0.0 --port 8000
```

Measured 2026-08-11 on the five `user_photos` boards: 0.938 → **0.969**,
hallucinations unchanged at 1, wrong-type 19 → 9, and the decapitation
confusions gone (queen recall 0.40 → 1.00). Verified through this serving path,
not just the experiment. Three things to check by hand:

- `GET /healthz` grows a `tuned` block naming the tuned model **and its base**.
  If it is missing, the average is not running.
- **Never serve the tuned model as `BR_MODEL`.** Alone it reads 0.966 but invents
  pieces on empty squares (hallucinations 1 → 4 on the same five boards); the
  base's confident "empty" is what votes them back down. The server enforces
  this: `BR_MODEL_TUNED` without `BR_MODEL` is a hard startup error, with no
  fresh-clone grace — a tuned model only exists because somebody ran the tool.
- `BR_MODEL_TUNED` **and** `BR_MODEL_B` together is also a hard startup error.
  Both pairings are measured; stacking them is not, and the error names the two
  numbers (0.969 calibrated vs 0.956 zero-shot) so the choice is informed.

The `≥3 sessions` rule is unit-tested in the workspace suite
(`tools/train/tests/test_calibrate_finetune.py`, the one torch-free file in that
directory — it is named explicitly in `testpaths`).

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

### Option J — Phone live camera (the browser edge, DESIGN §6 "virtual Pi")

Point a phone at a real board and stream it live — capture, crop and adaptive
forwarding all happen in the phone's browser; the desktop shows the position
evolving on the live viewer.

`make dev` sets all of this up — it starts inference plus **two** binds of the
viewer, serves the **same cascade `docker compose up` does** (it prints the
composition it resolved, and falls back to `BR_MODEL` alone with a warning if
`models/run_f/` is absent), and prints the URL for each page:

```bash
make dev
#   camera     https://<lan-ip>:8443/camera   (accept the warning once)
#   snapshot   http://<lan-ip>:8080/snapshot
#   live       http://<lan-ip>:8080
```

The camera page gets its own https origin because it is the only page that *needs*
one: `getUserMedia` is refused outside a secure context. The https bind serves all
three pages, though — every endpoint any of them touches is relayed by the viewer
itself, so there is no mixed content to block. `/` and `/snapshot` are offered over
http purely to spare the desktop a self-signed-certificate warning.

To reach any of them from off the LAN, forward the http bind through a tunnel that
terminates TLS for you — see `.claude/skills/tunnel`.

By hand, if you want different ports or only the camera:

```bash
# 1. Inference with real recognition (same env as Option H):
BR_MODEL=models/run_e/model.onnx uv run board-reader-inference serve --host 0.0.0.0

# 2. Viewer with TLS — getUserMedia requires a secure context on a phone:
BR_INFERENCE_WS=ws://localhost:8000/viewer BR_INFERENCE_HTTP=http://localhost:8000 \
uv run board-reader-viewer --host 0.0.0.0 --port 8443 --ssl
# it prints the https://<lan-ip>:8443/camera URLs to type on the phone

# 3. On the phone: open the printed URL, accept the self-signed certificate once.
#    (Android over USB instead: `adb reverse tcp:8443 tcp:8443`, then plain
#    http://localhost:8443/camera — localhost is a secure context, no cert.)
```

Flow: open the page **in landscape** — not just by design preference: a portrait
frame is **hard-blocked** (an overlay says "turn the phone sideways"; detect,
calibrate and start are disabled until the phone turns). The node's per-square
crops extend toward the top of the image, so a board lying sideways in the frame
reads garbage that looks like a position problem — it cost the 2026-08-27 field
day, hence a block rather than a warning.
Allow the camera and the preview fills the screen. **The board is found on its own**:
once the picture holds still, after a short settle delay (~2.5 s, so autoexposure and
autofocus have caught up) the page detects the board without being asked — no framing
step in the common case. Six controls sit on a rail (a right-hand column in landscape,
a 3×2 dock at the bottom in portrait): `more` · `position` · `detect board` ·
`rotate labels` · `calibrate` · `start`/`stop`.

**rotate labels** is still how the operator *overrides* which corner is a8 —
geometry cannot know that (decision #15) — but a calibration now sets it without
help: `/calibrate` resolves the true labelling from the pieces and the page folds
the reply's `label_rotation` into its own corners (the banner says
`labels aligned from the pieces (turned 90°)`), so nobody presses rotate after a
calibrate any more. And when `/calibrate` refuses because the *photo* is rotated
(the portrait failure reaching the node some other way), the banner says that one
actionable sentence instead of the node's full paragraph. **detect
board** re-runs detection on demand, for the cases automatic detection does not reach
(a bad angle, nothing recognisable as a board); the four corner handles are always
there to drag by hand too. **calibrate** grabs one full-resolution frame and posts it
through the guarded `/calibrate` (board must show the starting position, pieces at
rest) — and the FIRST one fires by itself: once streaming is on and the live
reading looks like a start position sitting still (`shouldAutoCalibrate`,
occupancy-banded in any rotation, stability ≥ 0.9), the page runs the same
full-res sharpest-frame burst unprompted, so the whole session flow is aim →
start → play. The node's agreement gate stays the authority — a false trigger is
refused quietly and retried after a 30 s cooldown. Because `/calibrate` is
additive, tapping the pill from a second or third angle still grows the same bank
rather than replacing it, and the button's own label carries the running count
(`calibrate · 3`). **position** shows or hides a floating live-position
board that can be dragged by its header and resized from a corner grip. **more** opens
a slide-up sheet holding the board id, the lens picker, tracking on/off, reset crop,
the stats strip (frames/rate/stability), a **what the reader sees** section carrying the
placement string and the low-confidence legend, and the framing explanation.
**start**/**stop** toggles streaming.

What streams is not the raw frame, and the page shows you exactly what does: the dashed
box drawn around the quad is the crop being sent — the quad's bounding box with a margin,
and considerably more headroom above the top edge, which is where tall pieces stand up out
of the board. It goes downscaled, with the quad riding on every frame header as `corners`
— the gateway then cuts full-piece boxes from the original view, exactly as a snapshot
read would. The
rate is adaptive (~1 fps at rest, bursts through motion). The desktop live viewer
(`/`) shows the same board id, where the preview pane displays the very frames the
phone is sending and its calibrate button now honours the streamed corners
(`X-Corners`).

What to check:
- Stability ~1.0 and ~1 fps with the board at rest; a burst and a dipping stability
  bar the moment a hand enters; the moved piece appearing on both boards within a
  second or two of the hand leaving.
- Point the page at a real board for the first time: it detects the board on its own
  within a few seconds of holding still, no tap needed. Point it at something with no
  board in it instead (a wall, an empty table) — it retries with a growing gap between
  attempts rather than on a flat timer, gives up after four tries, and says so as a
  tappable line in the hint bar (`No board found — drag the handles onto it, or
  look again`). Measured in a browser: four attempts, gaps of 8.7 s, 18.5 s and 32.7 s,
  then silence. That matters more than it looks — a miss costs `/detect` 2-4 s of
  OpenCV, so a flat 5 s retry would have kept the inference node busy roughly half the
  time it was pointed at a wall.
- After a re-detection that follows losing the board (see Tracking below), the labels
  may have landed on a different physical corner: the warning banner does not
  auto-dismiss, and **rotate labels** is the fix — nothing else clears it.
- Rotate the phone mid-session (landscape to portrait and back) — the portrait
  overlay appears at once, sending pauses (`stream paused` in the overlay), and on
  turning back both clear by themselves: the quad is still on the board and frames
  flow again, nothing pressed.
- Shake the phone gently in its holder while streaming: readings should hold up
  better than the shake suggests — every frame now carries an edge-measured
  `sharpness` (tenengrad on the tracker's frame), and when the estimator is
  behind, the node's inbox keeps the *sharpest* frame of the last second rather
  than blindly the newest (`ingest.prefer_frame`). Verifiable after the fact in a
  recording: `frames.ndjson` headers carry `sharpness`, `events.ndjson` carries
  the chosen `frame_seq`.
- Drag **position**'s header to move it and its corner grip to resize it; it survives
  a rotation without ending up off-screen.
- After **calibrate**, a banner reports the class count and the bank size, and the
  button's own count goes up.
- Zero-tap: on a fresh bank, set up the start position, press only **start** —
  within a few seconds of the board sitting still the banner reads `Calibrated
  automatically from the start position, …` with no calibrate tap. Set up a
  mid-game position instead and it never fires.

**Tracking (DESIGN §4.4.1(a)).** Once the corners are set they follow the board: pick
the phone up, move it, put it down somewhere else and the quad — with its `a8`
labelling intact — is still on the board. The status chip reads `tracking — locked`.
Things to try:

- Cover the lens with a hand. The chip turns red to `tracking — lost, quad frozen`
  and the quad goes dashed red and **stops moving** rather than wandering. Take the
  hand away quickly and it re-locks by itself, no re-detection involved — a hand
  sweeping over the board during a real game must not, and does not, trigger one.
  Leave it lost for longer instead (several seconds, sustained) and the page
  escalates: it re-runs detection on its own, deliberately keeps the labels as they
  were, and raises the non-auto-dismissing warning described above.
- Move fast enough to smear the picture — same short-loss recovery, and it re-locks
  when you stop.
- Drag a handle at any point: that re-anchors tracking on what you just said, so a
  quad that has drifted is one touch from right. **tracking off** (in the **more**
  sheet) disables it outright.

**Lens (iPhone).** With more than one camera a **lens** picker is offered in the
**more** sheet. Pick *Back Ultra Wide Camera* when the board will not fit the main
lens from where the phone can sit; the choice is remembered by label. Expect visible
barrel distortion near the frame edges — the four-corner quad is a projective model
and fits the board progressively less well out there, so keep the board near the
middle. Re-check the corners after switching: the frame size changes with the lens.

**Camera stall recovery.** Phone cameras die quietly (backgrounding, a call,
another app taking the lens); the page must come back on its own. Things to try
mid-stream: switch to another app for ten seconds and return — the preview
resumes (worst case after the `The camera stalled — restarting it…` banner);
lock the phone and unlock it — same; open the system camera app and come back —
the track is dead outright, and the page reopens the lens. The watchdog reads
`video.currentTime` every tick, so it needs no cooperation from the browser: any
stall of a few seconds escalates play → reopen with backoff. Every attempt is in
`client-logs/` (see Option K's post-mortem kit).

**Impossible boards.** The live panel draws the node's prior-repaired placement:
force a two-kings read (swap the queen for a spare king mid-stream, or just watch
a marginal angle) and the panel shows one king, the repaired square ringed red,
`repaired N` in the status line. The raw argmax stays on the wire untouched.

### Option K — Reproduce a run from its recording (WP-H2)

Every frame that reaches `WS /ingest` is written to disk, so **any** of the runs
above can be replayed afterwards — including Option J, since the phone's frames
cross the same socket through the viewer's proxy. `serve` records to
`recordings/` by default; `docker compose` records to the bind-mounted
`./recordings`.

```bash
# 1. Do a run — anything that streams. Then check what landed:
curl -s http://localhost:8000/healthz | jq .recording
#    {"path": "recordings/20260807-143012", "frames": 412, "active": true, ...}

ls recordings/20260807-143012        # manifest.json  frames.ndjson  jpeg/
head -1 recordings/*/frames.ndjson | jq   # the wire header + where its bytes went

# 2. Replay it into a (restarted) gateway, at the speed it happened:
uv run board-reader-edge feed --recording recordings/ --throttle
#    -> "replayed 412 frames from recordings/20260807-143012 to ws://..."
```

`--recording` with no session id takes the newest session in the directory —
"the run that just happened". The replay re-sends the **stored bytes**, not a
re-encode, so the estimator sees the same pixels it saw the first time.

What to check: the viewer replays the original run move for move; the replaying
server's own `recordings/` now holds a second session with the same frame count.
That second recording is the reproduction claim — a run whose input can be
re-fed is a run you can bisect a recognition regression against.

> [!note] Why `--throttle` matters here
> The gateway's latest-wins pre-filter drops frames as a function of *arrival
> rate*. A flat-out replay arrives far faster than the camera did, so it drops
> more and reproduces an easier run than the recorded one. Recording happens
> **before** the pre-filter, so the recording itself is complete either way.
>
> `--adaptive`, `--rectify`, `--corners`, `--forward-original` and `--loop` are
> refused with `--recording`: a recording already holds the edge's *output*, and
> re-running those stages over it would put something else on the wire.

Turn it off with `serve --no-record` (runs are then not reproducible); cap it
with `BR_RECORD_MAX_MB` on a small disk — the session truncates rather than
rotating, so a capped recording keeps its beginning.

**What a trial leaves behind (the post-mortem kit).** Beside `frames.ndjson`
(the inputs) a session also holds `events.ndjson` — every *published* position,
one compact line each (`square_dist` stripped; `frame_seq`/`capture_ts`/
`published_ts` join it to the exact frame it came from, and `prior` records what
the hard-constraint decoder repaired):

```bash
tail -f recordings/*/events.ndjson | jq -c '{seq, placement, prior: .prior.flips}'
```

Each line also carries `evidence`: the **top-2 digest**, 64 squares in board
order, each `[[class, p], [runner-up, p]]`. It is what survives dropping
`square_dist`, and it is the difference between "d8 read `k`" and "d8 read `k` at
0.51 over `q` at 0.49" — a near-miss the crop nearly answered, or a conviction the
evidence never contained. Read one square straight off the file:

```bash
# d8 is index 3 (rank 8 -> rank 1, files a -> h), i.e. square_name_at(3)
jq -c '[.frame_seq, .evidence[3]]' recordings/<session>/events.ndjson | tail -5
```

> [!note] Sessions recorded before 2026-08-27 carry no digest
> They score exactly as they always did; the report says the evidence is missing
> rather than implying it was empty.

Three more places to look after a phone trial:

- the serve terminal — one INFO line per *changed* read
  (`read <board> seq=… conf=… latency=… <placement> [d8 k->q]`), WARNINGs for
  refused frames; `BR_LOG_LEVEL=DEBUG` logs every read.
- `client-logs/*.ndjson` (written by the **viewer**) — the phone's own story:
  socket status changes, camera track mute/ended, stall-watchdog play/reopen
  attempts, calibrate/detect outcomes, `pagehide`. This is the file that says
  *why the node heard silence* — a page that never connected, a camera that
  froze, or a stream whose every frame was refused all look identical from the
  node's side. While streaming it also beats every 10 s (`kind: "heartbeat"`,
  with `sentPerSec`/`droppedPerSec`/`unacked`/`behind`), so a quiet stretch says
  which kind of quiet it was — `jq -c 'select(.kind=="heartbeat") | .detail'
  client-logs/*.ndjson`.
- `curl -s localhost:8000/healthz | jq .ingest` — `received/estimated/dropped/
  failed/skipped/refused` counters; all zeros while the phone claims to stream
  means the frames never arrived.

### Option L — Watch or capture the position stream

Every option above ends at a screen. This one ends at a file. `positions` is the
headless consumer of `WS /viewer`: it tails the stream in readable form, and with
`--out` writes each event as one line of the exact §5 JSON the socket carried, so
a captured run *is* the stream and two runs can be diffed.

Start any streaming run (Option A, B, J or K), then, in another terminal:

```bash
uv run board-reader-inference positions                        # tail every board
uv run board-reader-inference positions --changes-only         # only when a board moves
uv run board-reader-inference positions --out run.ndjson --quiet
uv run board-reader-inference positions --board A1 --limit 20  # scriptable: stop after 20
```

The tail is one line per event — identity, the derived numbers, and what changed:

```
A1 #41 2026-08-07T14:31:02.118Z  conf 0.94  stab 0.98  2Δ  e4 ·→P, e2 P→·
A1 #42 2026-08-07T14:31:03.121Z  conf 0.94  stab 0.99  (unchanged)
```

What to check:

- against the Docker demo (Option A), the tail's transitions match the squares the
  viewer rings green, at the same moments;
- `--changes-only` collapses the ~1 ev/s keepalive to nothing while the board is at
  rest, and prints a burst of lines through a move;
- `jq -r .placement run.ndjson | uniq` reads the game's positions back out in order, and
  `wc -l run.ndjson` matches the "wrote N position events" line on stderr. Under
  `--changes-only` that line also names what it held back ("… (7 unchanged, not written)"),
  so the two numbers still account for the whole stream;
- replaying a recording (Option K) into a fresh gateway and capturing again gives a
  second file to diff against the first — that diff is the reproduction claim.

Exit codes are meant for scripts: `0` when events arrived, `1` when the stream produced
nothing (an empty stream is not a working one), `2` when there is no gateway at `--url`.

> [!note] It smooths nothing, and that is the point
> A square flickering between two classes on alternate frames produces a transition on
> *every* frame — the tail diffs consecutive placements and has no memory beyond the last
> one. That is the honest output of a stateless per-frame estimator (DESIGN D1), not a bug
> to fix here: temporal smoothing belongs to a consumer layer that does not exist yet. See
> [[architecture#Temporal consistency — the layer nobody has built]].

---

### Option M — Score a trial recording (and export crops from it)

Option K replays a recording. This one *grades* it. A trial — a person at a real
board with a phone on the /camera page — records every frame and every published
event, which makes it eval data as soon as somebody writes down what was actually
on the board. That is `truth.json`, and it goes **inside the recording it
describes**:

```bash
# 1. Find the session (they sort chronologically) and see what it needs:
uv run board-reader-inference eval-session recordings/20260813-193308
#    -> exits 2, prints a skeleton with the board id and first seq filled in
```

Redirect that skeleton into place and fill in the boards, one segment per
position, `from_seq` being the frame seq the position started holding from
(`jq -r '[.frame_seq, .prior.placement] | @tsv' recordings/<s>/events.ndjson`
is a good way to find the boundaries — read the *board*, not the page, or the
label is only as good as the estimator that produced it):

```json
{
  "format": 1,
  "board_id": "phone",
  "segments": [
    {"from_seq": 0,   "placement": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"},
    {"from_seq": 178, "placement": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"}
  ],
  "guard_seqs": 12
}
```

```bash
# 2. Score it (no model, no GPU — it reads events.ndjson, ~0.1 s):
uv run board-reader-inference eval-session recordings/20260813-193308

# 3. Same run, plus per-square training crops labeled from the same truth:
uv run board-reader-inference eval-session recordings/20260813-193308 \
    --export-crops data/external/trial_20260813 --every 5 --json trial.json
```

Scoring **always** writes the full report as JSON — `<recording>/report.json`
unless `--json` names somewhere else, and the rendered report names the file it
wrote. That is deliberate: this project has twice quoted a headline accuracy from
a run that left no artifact, and neither number could be reproduced afterwards.

What to check:

- the three streams are scored apart — raw `placement`, the hard-constraint repair
  (`prior.placement`) and the debounce (`settled.placement`). A repair that fixes
  as many squares as it breaks shows up as an unchanged per-square number with a
  *different* perfect-board rate, which is the whole reason they are not summed;
- `settled: DERIVED` in the header means the recording predates the settled block
  and that row was replayed offline through today's `BoardStabilizer` — a
  prediction, not a measurement;
- excluded counts are on the `events:` line. `guard_seqs` drops the events within
  that many frames *after* a segment boundary (the hand is over the board); if it
  is swallowing most of the session, the boundaries are wrong, not the guard;
- **`raw · evidence on the N wrong squares`** answers "is this fixable?" without
  putting the model back in the loop: how often truth was the runner-up, the
  median `p(read) - p(truth)` margin, and the same split per confusion pair.
  Small margins with truth ranked second mean the crop carries the answer;
  truth outside the top 2 means the evidence is not there at all;
- **`timing`** is the rate story a placement score cannot show — frames in (fps),
  reads out (reads/s), frames per read, and the capture→publish latency a live
  display actually feels. It reads the frame *index* only, never the JPEGs;
- the export writes a `labels_normalized.json` crop-level style plus an
  `export.json` naming the geometry that cut it. `uv run --directory tools/train
  python styles.py` should then list it, and `python cropcache.py <style>` shard it.

Exit codes: `0` scored, `1` nothing left to score, `2` no `truth.json` (skeleton
printed) or a `truth.json` that cannot be trusted.

### Option N — Replay a take through today's stack (offline benchmarking)

Option M grades the events a run *published*. This one produces fresh ones: it
pushes the recorded frames back through the estimator stack the environment
builds today — the same routing decision, the same prior repair, the same
debounce, the same event writer the live gateway uses — and writes them as a new
recording-shaped directory. **A recorded take replaces a live one for
iteration:** change the code, replay, score, diff. No phone, no board, no server.

```bash
# Whatever `serve` runs is what replay runs — same env vars, same builder:
BR_MODEL=models/run_e/model.onnx BR_MODEL_B=models/run_f/model.onnx \
BR_SESSION_BANK=models/exemplars_session.npz BR_GPU_MEM_LIMIT_MB=1024 \
uv run board-reader-inference replay recordings/20260813-193308 --score
```

It lands beside the take it came from, so a benchmark can always be traced to its
input — and the source recording itself is never written to:

```
recordings/20260813-193308/replays/<name>/manifest.json  # source, git, BR_MODEL*, digest
                                        /events.ndjson   # the fresh reads
                                        /frames.ndjson   # the replayed lines (thin)
                                        /jpeg -> ../../jpeg          (symlink)
                                        /truth.json -> ../../truth.json
```

`<name>` defaults to `<git describe>-<digest of the BR_MODEL* env>`: the same code
against the same models replays into the same directory (a re-run supersedes its
own answer), a different model lands beside it. `--name` overrides. `--score` runs
Option M's scorer on the result straight away; without it the command prints the
`eval-session` line to run.

What to check:

- **the replay is deterministic** — same recording, same code, same env gives a
  byte-identical `events.ndjson` (modulo the model's own float nondeterminism), so
  two replays can be diffed as a claim about the *code*. Everything time-based
  runs on each frame's recorded `capture_ts`, never the wall clock: a 2.5 s settle
  window is 2.5 seconds *of the recording* whether the replay takes ten minutes or
  ten milliseconds;
- **more events than the run published.** The live gateway's latest-wins inbox
  drops frames when the estimator falls behind (08-13: 85 events from 748 frames);
  replay reads every frame, so the report's `timing` block and the settled row are
  denser than the live ones. Compare the raw/prior rows with the recorded report,
  not the rates;
- **`published_ts` is absent** and the report says `latency not stamped` — nothing
  was published, and a fabricated capture→publish number is worse than none;
- **multi-session recordings**: a reconnecting edge restarts `seq` at 0, and truth
  is a step function over `seq`, so only the **longest** session is replayed and a
  warning names the others. `--all-sessions` replays them all into one file;
- `--limit N` stops after N estimates — a smoke run when the GPU is busy;
- `--board <id>` picks one board out of a recording that holds several.

> [!note] Which frames get estimated
> The loop asks a *frame policy* per frame, and the only one today is
> `EveryFrame` — offline there is no reason to leave a recorded frame unread. It
> is the seam the move-triggered estimation milestone plugs into (estimate at
> burst end, slow keepalive, skip mid-burst): the policy sees the recorded header
> (`seq`, `capture_ts`, `stability`, `sharpness`, `burst_id`) and not the pixels,
> so a declined frame costs not even a JPEG read.

---

## Quick health checks

```bash
curl http://localhost:8000/healthz            # {"status":"ok",...} + .recording
curl http://localhost:8000/boards/A1          # latest position event JSON for A1
```

## Troubleshooting

- **Viewer says “disconnected — retrying”**: the page reached the viewer (it loaded),
  so the broken hop is viewer→node. Check `BR_INFERENCE_WS` on the `viewer` service —
  in Docker it is `ws://inference:8000/viewer`. `curl <viewer>/api/healthz` isolates
  it: 200 means the relay is fine and the node has simply sent no events yet, 502
  names the address the viewer failed to reach.
- **No board appears**: check the edge is sending — `docker compose logs edge` — and
  that `curl localhost:8000/boards/A1` returns an event (not `404`).
- **`docker compose` build is slow the first time**: it downloads the Python base
  image and OpenCV wheels; subsequent builds are cached.
