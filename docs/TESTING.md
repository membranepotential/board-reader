# Testing board-reader

Two layers, both required by the project's "user-testable from the start" principle
(see [../CLAUDE.md](../CLAUDE.md)):

1. **Automated tests** — fast, run on every change.
2. **The live demo** — a human runs the whole pipe and *sees* it work.

---

## 0. Read this first (the 60-second version)

The suite is **fast — ~16 s for all ~860 tests**. If testing is costing you more
than that, you are doing one of the things in this section. Read it before you run
anything; do not read the remaining 600 lines unless you are adding a test.

**The only two commands you normally need:**

```bash
make test                                  # everything, ~16s. The gate before a commit.
uv run pytest packages/inference/tests/test_server.py -q    # one file, while iterating
```

| Want | Run | Cost (measured 2026-11-24, one suite at a time) |
|---|---|---|
| The gate (before committing) | `make test` | **~16 s** — all ~860 |
| One file / one test | `uv run pytest <path>[::<test>] -q` | 1–8 s |
| One package | `uv run pytest packages/<pkg> -q` | core 1 s (128) · edge 1 s (85) · inference 11 s (495). The viewer has no Python tests: it is Vitest and Playwright (`make test-front`) |
| Recognition changes only | `make test-gpu` | a few s, **needs an idle GPU** |

The rest of the ~860 are `tests/` (54) and `tools/video/tests` (98). A package runs
*serially*, so scoping to `packages/inference` costs about as much as `make test`.
Narrow to a file, or just run the gate.

**Rules that save the most time:**

- **Don't run the suite when nothing it covers changed.** A docs, comment,
  `Makefile` or `STATUS.md` edit is not a reason to spend a run (CLAUDE.md §2).
- **Don't add `-n auto` to a focused run.** Worker startup costs more than the tests
  you selected; `make test` already parallelises the full suite. Plain `uv run
  pytest <path>` is the fast path for one file.
- **Don't run `uv run pytest` bare to check your work** — that is the whole suite
  *serially*, ~23 s, and it re-enables the GPU test. `make test` is strictly better.
- **Never run two suites at once**, or a suite alongside a replay or `make gate`. The tests
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
make test              # the suite across all cores, minus the GPU test (green), ~16s
make test-gpu          # the GPU test alone — an idle GPU, or it flakes
make test-all          # everything, GPU included
uv run pytest          # the whole suite, serially (~23s)
uv run pytest -q       # quiet
uv run pytest packages/inference          # just one package
uv run pytest tests/test_end_to_end.py    # the real-network walking-skeleton test
```

> [!note] Why `make test` is not the whole suite
> One test (the real-V6 still in `test_chessq.py`) builds a real ONNX
> `InferenceSession` and takes the CUDA execution provider. On an idle machine it
> passes; on a busy one — a replay, or a second agent session running its own
> suite — session creation dies with `BFCArena Failed to allocate memory`, and the
> pressure has been observed taking unrelated OpenCV allocations down with it. It is
> marked `gpu` and deselected by default, so a red suite means the *code* is red.
> Run it deliberately with `make test-gpu` before trusting a recognition change.

What's covered today — the main files (`git ls-files 'packages/*/tests/*.py'
'tests/*.py'` lists them all):

| Area | File | Proves |
|---|---|---|
| L3 protocol | `packages/core/tests/test_protocol.py` | placement⇄distribution, validation, JSON roundtrip |
| Wire format | `packages/core/tests/test_wire.py` | JPEG + header frame roundtrip; legacy `corners` still parse |
| Sim renderer + warp | `packages/core/tests/test_sim.py` | board images render; perspective warp + ground-truth quad |
| Frame sources | `packages/edge/tests/test_source.py` | video/dir/synthetic/repeat sources |
| Edge adapter | `packages/edge/tests/test_adapter.py` | frames → wire envelopes, throttle, **static `--roi` crop, motion measured inside it, no `corners` sent** |
| **Recognizer seam** | `packages/inference/tests/recognition/test_recognizer.py` | **the stub, the event a distribution becomes, the one frame read, the environment builder and its loud fallback** |
| **ChessQ** | `packages/inference/tests/recognition/test_chessq.py` | **preprocessing and output mapping on a fake model; one `gpu` test reads a held-out still with the real V6** |
| **Repair + settle** | `packages/inference/tests/stream/test_prior.py`, `test_stabilize.py` | **an impossible board is repaired at its weakest square and a possible one is left byte-for-byte; the per-square debounce behind `settled`** |
| **Recording format** | `packages/core/tests/test_recording.py` | **byte-exact round trip, corners survive, truncated session still replays, size cap** |
| **Ingress recording** | `packages/inference/tests/test_recording_ingest.py` | **every `/ingest` frame lands on disk (incl. pre-filter drops); off unless asked; a write failure never breaks the stream** |
| **Replay (edge)** | `packages/edge/tests/test_recording_replay.py` | **replays the recorded bytes without re-encoding, paces by capture time, refuses to reprocess** |
| **Replay (inference)** | `packages/inference/tests/measurement/test_replay.py` | **driven by the recorded clock, byte-identical events on a re-run, the frame policy is asked** |
| **Labels + eval** | `test_labels.py`, `test_server_labels.py`, `test_evaluate.py`, `test_trial.py`, `test_calibration.py` | **chess-frame labels, `eval`, `eval-session`, temperature scaling** |
| **Position stream consumer** | `packages/inference/tests/test_positions.py` | **transitions of a real move, first-sighting ≠ unchanged, per-board diffing, NDJSON capture rebuilds the identical event, `--changes-only` drops keepalives** |
| Hub | `packages/inference/tests/stream/test_hub.py` | retained-latest, fan-out, overflow |
| Gateway | `packages/inference/tests/test_server.py` | ingest → viewer over WS (in-process) |
| **Read scheduling** | `packages/inference/tests/stream/test_schedule.py` | **burst detection matches the forwarder's hysteresis; the keepalive floor survives a dead, stuck, absent or NaN meter and a backwards clock; no config spelling switches it off; a new move cancels the last one's owed reads; spec strings round-trip** |
| **Scheduling, live** | `packages/inference/tests/stream/test_ingest_schedule.py` | **the policy is observed on every arriving frame and asked at the drain, so a burst-end read gets `prefer_frame`'s sharper contender; per-board isolation; a frozen `capture_ts` falls back to the node's own keepalive; `BR_SCHEDULE` reaches the socket and `every-frame` restores the old gateway** |
| **Camera control** | `packages/core/tests/test_control.py`, `packages/inference/tests/control/test_camera.py` | **the config, report and socket messages refuse malformed input; the registry pushes a config to every phone and every control page, previews are wanted exactly while a page watches, a reconnecting phone hears its config again; with `BR_CAMERA_CONFIG` set the configs survive a restart (written atomically), a corrupt file is ignored with a warning, unset writes nothing, and `serve` sets the default; the board direction is the median of the latest reads, names the nearest side, flags a moved camera once the angle settled and resets on a new crop, lens or resolution; over the real sockets a config reaches the phone, a preview and the last streamed frame reach the page, a health tick arrives with the direction, and a bad message is refused without closing the socket** |
| Viewer | Vitest, not this suite: `packages/viewer/ui/server/*.test.ts`, `src/routes/api/**/server.test.ts` (see *Frontend* below) | the `/api` relay forwards verbatim and names an unreachable node; the `/ws` relay; the cached TLS certificate; the banner; the client log |
| **Layer split** | `tests/test_layer_separation.py` | edge/inference never import each other |
| **Deploy parity** | `tests/test_deploy_parity.py` | compose and `make dev` serve the same recognizer |
| **End-to-end** | `tests/test_end_to_end.py`, `test_m2_stream.py` | real uvicorn + edge client + viewer WS; the adaptive stream |
| **Position stream, e2e** | `tests/test_position_stream_e2e.py` | a fed position comes out of the pipe, read off the pixels by `tests/_sim_recognizer.py` — a test-side template matcher that reads `core.sim.render_board` renders exactly (ChessQ reads 0/14 of them) |

### Frontend (`packages/viewer/ui`)

A second, independent test layer for the SvelteKit UI — a frontend-only change is
not a reason to run `make test`, and a Python-only change is not a reason to run
these:

```bash
make test-front                    # both layers, from the repo root — Vitest then Playwright

cd packages/viewer/ui              # or, one at a time, from the package
pnpm install --frozen-lockfile     # once; mirrors `uv sync --frozen`'s discipline
pnpm test                          # Vitest — src/lib/**, node env, ~1s, 262 cases today
pnpm check                         # svelte-kit sync + svelte-check, strict TS
pnpm test:e2e                      # Playwright — 21 tests in a real Chromium, ~15s
```

`make test-ui` and `make test-e2e` are the two halves of `make test-front`; each runs
`pnpm install --frozen-lockfile` first, and `test-e2e` also runs `playwright install
chromium` (a one-off on a fresh checkout — Playwright pins its own browser build).

`pnpm test` is the tested half of what the three hand-written pages used to
triplicate: FEN⇄grid conversion (`lib/board/fen.ts`), the wire protocol parser
(`lib/protocol.ts`, including the camera config), the motion meter, focus score and
`/ws/ingest` framing (`lib/edge/`), the phone's config-to-camera decisions
(`components/camera/state.ts`), the control page's crop editing and framing warnings
(`lib/control/`), and the camera control channel: its messages
(`lib/control/messages.ts`), the reconnecting socket both clients share
(`lib/socket.ts`, with the fake in `lib/fake-socket.ts`), the phone's link (a report
only when it changed, previews only while wanted, the config kept across a lost link)
and the control page's link — all plain TypeScript, run in `node`, no DOM.

`camera.spec.ts` plays the node's half of the control channel: on the phone's
`/ws/boards/{id}/camera` socket it pushes a config of its choosing and switches
previews on and off, and it decodes the JPEGs that reach `/ws/ingest`, so "the crop
the config asked for is the crop sent" is measured in pixels.
It also sends a drawn hand across the board and checks that a burst reaches the wire,
and that changing the crop alone does not open one.
`control.spec.ts` drives `/control` against a simulated node on its
`/ws/boards/{id}/control` socket, including the side White sits on and the
camera-moved warning, both from the health tick.

`pnpm test:e2e` drives the pages in a real Chromium (`tests/*.spec.ts`), against
the **built** app — `vite build`, then the real viewer server
(`node server/main.ts`), so what is driven is what ships. There is no inference node behind it: every `/api/…` call and the
`/ws/viewer` socket is stubbed per test (`tests/helpers/node.ts`), so these need neither
a GPU nor a model. They cover what neither other layer can reach — a chessground board
that laid out with no pieces on it, a rail button 31 px tall on a phone, an
`@media (orientation: portrait)` rule that never applies, a preview that fetches while
collapsed. Each of those has shipped here at least once.

> [!note] Chromium only, and touch is emulated
> Playwright drives Chromium with mouse events synthesised for taps. iOS Safari — the
> actual target device for `/camera` and `/snapshot` — is still only testable by hand;
> see §2's phone walkthrough.

`pnpm dev` runs the Vite dev server (hot reload) against a *running* inference node:
the `/api` routes relay to `BR_INFERENCE_HTTP` and Vite's proxy stands in for the
`/ws` relay, reaching `BR_INFERENCE_WS`'s origin (default `ws://localhost:8000` — see
`vite.config.ts`):

```bash
uv run board-reader-inference serve --port 8000 &   # terminal 1: gateway
cd packages/viewer/ui && pnpm dev                    # terminal 2: frontend -> http://localhost:5173
```

`make ui` (`pnpm build` under the hood) is the production build instead of the dev
server: it builds the app into `packages/viewer/ui/build/`, which the viewer server
(`server/main.ts`) serves — see the note at the top of §2.

---

## 2. The live demo (what a user runs)

> [!note] Build the UI once
> The viewer server (`node packages/viewer/ui/server/main.ts`) serves a SvelteKit build
> and refuses to start without one (`packages/viewer/ui/build/` is generated and
> gitignored, so a fresh checkout has none). Docker's own `ui-build` stage handles
> Option A; every local-process option below needs `make ui` run once first (and
> again after any change under `packages/viewer/ui/`).

### Option A — Docker (canonical, one command): the M2 live game replay

```bash
docker compose up --build
```

Then open **http://localhost:8080**. The edge replays a **synthetic recorded
game** (looped forever) and forwards it adaptively (DESIGN D4). ChessQ does not read
these synthetic renders (0/14 perfect boards, [the runbook](runbook.md#running-it)),
so this demo proves the pipe, not recognition: positions change, but they are not the
game's positions. Without the weights the stub shows the start position throughout.
You should see:

- the connection chip turn **“live”** (green), a board card for **A1** with the
  position drawn by **chessground** (real pieces, coordinates) — the same board
  the snapshot page uses,
- the position **changing as the game moves** — each changed square ringed
  **green**, fading pale and gone within a second,
- the rhythm of a **scheduled** stream (DESIGN §4.3): about one event every 2 s at
  rest (the node's keepalive read), none while the synthetic hand is over the board,
  then three events ~0.3 s apart once it leaves (the burst-end reads). The edge still
  forwards ~12 fps through the hand; `curl -s localhost:8000/healthz | jq .ingest`
  shows `unscheduled` climbing while `estimated` crawls. `BR_SCHEDULE=every-frame`
  on the `inference` service restores one event per frame,
- squares the settled board has not caught up with yet ringed **dashed amber**; the
  legend under the board calls them “not settled yet”. The card shows no confidence
  bar and no stability bar: ChessQ reports about 1.0 on its wrong squares too,
- the **motion chip** on the card, once the node sends motion messages on `/viewer`:
  **“moving”** while a hand is over the board, **“still”** at rest, and **“motion ?”**
  when the last motion message is more than 3 s old (stop the edge to see it). A node
  that sends no motion messages shows no chip at all,
- the **preview** tab on the board's left sliding open onto the **latest frame the
  node ingested**, refreshing as events arrive — it reads “no frame yet” until one
  has been forwarded, and closing it stops fetching,
- the **age** readout ticking and the card fading to **stale** if you stop the
  edge container (`docker compose stop edge`) — and recovering when it returns.

The **rotate** button under each board turns the card's board picture in 90° steps —
purely cosmetic (CSS transform on the rendered card; the placement stays
white's-view a8-first on the wire), for matching the screen to however you're
standing relative to the physical board. At 90°/270° the coordinates turn
sideways with the picture, like turning a printed diagram. Manual check: clicking
cycles 0→90→180→270→0, a second board's card is unaffected, and the choice
survives a reload (persisted per board id in localStorage).

This runs three containers: `edge` (game feed) → `inference` (recognizer +
gateway) → `viewer` (UI). The browser talks only to `viewer`, which relays the node's
sockets and HTTP API; the published port 8000 is for debugging.

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

Three terminals:

```bash
uv run board-reader-inference serve --port 8000          # 1: gateway
node packages/viewer/ui/server/main.ts --port 8080       # 2: UI  -> open http://localhost:8080
uv run board-reader-edge feed                            # 3: synthetic feed (30 frames)
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

To send only the board's part of a wide shot, add `--roi x,y,w,h` (Option D).

> Note: the node reads the whole frame it receives with ChessQ. Real footage needs
> the weights (`make models`); without them the node serves the stub.

### Option D — Send only the board's rectangle (`--roi`)

The edge never looks for the board (DESIGN D11). It sends the whole
frame, or the static rectangle `--roi x,y,w,h` selects, in fractions of the frame
(top-left corner, width, height). Motion for `--adaptive` is measured inside the
rectangle too.

1. Start the node: `uv run board-reader-inference serve`.
2. Feed an angled synthetic board, cropped to its middle:
   `uv run board-reader-edge feed --angled --roi 0.2,0.2,0.6,0.6 --throttle --fps 2`.
3. Save the last frame the node received:
   `curl -s localhost:8000/boards/A1/frame -o /tmp/roi.jpg`, and open it. Expected:
   only the middle of the angled board, 60 % of the full view in each direction.
4. Pass a rectangle that leaves the frame, e.g. `--roi 0.6,0,0.5,1`. Expected: the
   CLI exits with a one-line `--roi` error and dials nothing.

> [!note] Options E, G and H are gone
> They demoed the earlier recognition stack — centroid few-shot calibration (E), the
> occupancy bank with start-position auto-calibration (G), and the trained crop
> classifier with its bank, cascade and calibration average (H). That stack was
> removed when ChessQ became the only recognizer; the last commit that has it, with
> these walkthroughs, is `457cf96`. The letters are kept so links to the other
> options stay valid.

### Option F — Phone snapshot → Lichess

Photograph a real board with your phone, get the position + a Lichess link. The
snapshot page is an application-layer consumer (DESIGN WP-H3, §2.3): the CV side still
only produces placement + confidences; side-to-move, castling guess, and the
Lichess URL are assembled in the page.

```bash
# 1. Serve inference with real recognition (ChessQ; `make models` first):
BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx \
BR_SAVE_UPLOADS=data/uploads \
  uv run board-reader-inference serve --host 0.0.0.0 --port 8000

# 2. Serve the viewer on the LAN:
node packages/viewer/ui/server/main.ts --host 0.0.0.0 --port 8080

# 3. On the phone (same Wi-Fi), open  http://<this-machine's-LAN-IP>:8080/snapshot
```

`make dev` does steps 1 and 2 in one command and prints the URL.

Flow: take or pick a photo → the page reads it at once, with no further taps.
There are no corners to set: ChessQ finds the board and its orientation in the
whole photo (DESIGN D10). The photo stays on screen next to the board it produced,
at the same size. The board renders with low-confidence squares outlined red, plus
**Open in Lichess analysis / editor** links (side-to-move toggle on the page).
**Read position** reads the same photo again.

**The page is the diagnostic surface for the reader** (viewer aspect). Under the
photo and the board, the four stages of the read are listed in the order they run —
*capture · read · impossible-board repair · position output*. Each row always shows
its state and headline numbers; expanding it shows the full evidence:

1. **Capture** — posted size (downscaled in the browser to a 1600 px long side;
   that image is exactly what is POSTed), JPEG size, mean luma against the server's
   `LOW_LIGHT_MEAN` (55).
2. **Read** — mean/weakest per-square confidence, an 8×8 confidence map (click a
   cell to inspect that square), the least-confident squares, and any `warnings`
   the server volunteered. The read is always in the chess frame (rank 8 on top).
3. **Impossible-board repair** — every square the hard-constraint decoder
   relabelled (two kings, a pawn on the back rank, too many pieces), cheapest first.
   The board above shows the repaired position; the raw distribution is untouched.
4. **Position output** — the FEN, side-to-move, Lichess links, copy FEN / copy raw
   event, the corrections made by hand, and a material census (two kings, ≤8 pawns).

**Correcting a square:** click it once to inspect — the inspector beside the board
shows what the model read and the full 13-class distribution, and a White/Black grid
to assign from. Click the same square again to toggle through its top-3 readings
(the fast fix). On a desktop the arrow keys move the selection and FEN letters
assign (`K` white king, `k` black king, `.` empty, `esc` closes). Inspecting never
changes a square; only assigning does.

**Saving a label:** **Save labelling** under the board keeps the board as shown as
the true position for this photo — see Option I.

**History** at the bottom lists every read of the session merged with every photo
the node still retains, newest first. Selecting a session read restores the whole
pipeline view — that photo, its distribution and its own corrections — so two reads
can be compared without re-shooting either. Selecting a retained photo loads it and
reads it again.

Notes:
- The page reaches inference through the viewer's `/api` relay, on whatever origin
  it was loaded from, so no extra config is needed on a LAN, over https, or behind
  a tunnel. `BR_INFERENCE_HTTP` on the viewer is what points that relay at the node
  when inference runs elsewhere.
- `POST /estimate` on the inference node accepts a whole photo (JPEG body) and
  returns the position event JSON — handy for scripting:
  `curl -s --data-binary @photo.jpg -H 'Content-Type: image/jpeg' localhost:8000/estimate`
- **Low-light warning**: a very dark photo (mean brightness < 55/255) makes the
  response carry `"warnings": ["low_light"]` and the page show a dimness hint.
- Without the ChessQ weights you get the stub (fixed start position) — fine for
  testing the plumbing, useless for real photos.
- Set `BR_SAVE_UPLOADS=data/uploads` on `serve` to keep every posted photo (+ its
  sidecar: predicted placement, read frame, prior flips) — real-world shots become
  eval data.

### Option I — Confirm ground truth, then score the reader

The two halves of measuring whether the snapshot app actually works: collect
labels through normal use, then run the deployed stack against them.

**Collecting a label** (needs `BR_SAVE_UPLOADS` — Option F's serve line and
`make dev` set it):

1. Photograph a position and read it as usual.
2. Fix any wrong squares (tap a square, pick the right piece), then press **Save
   labelling** under the board. The button is there when nothing is wrong too —
   saving a flawless read is exactly as much a label as fixing four squares, and
   the perfect-board rate is made of both.
3. A `data/uploads/<stem>.label.json` appears next to the photo, holding the
   position in the chess frame (`"frame": "chess"`), the squares the decoder
   repaired (`prior_flips`), and what the server predicted at the time. Editing a
   square after saving turns the button into **Update labelling** — a label
   describes one exact position.

> Labels are scoring data only. Nothing reads them back into the reader — a reader
> fitted on its own eval boards is the contamination they exist to measure.

**Scoring:**

```bash
uv run board-reader-inference eval                    # data/eval + data/uploads
uv run board-reader-inference eval --json report.json
uv run board-reader-inference eval --no-prior         # the raw read, no repair
```

The harness builds the recognizer from the environment exactly as `serve` does,
prints that configuration above the report, and leads with **perfect boards**,
not per-square accuracy — 94.5% per square is ~3.5 wrong squares on a board, and
a position with 3 wrong squares is not a position. It then breaks errors into
hallucination / missed / colour / type, and prints a reliability diagram: the
gap between claimed confidence and observed accuracy is what says whether a 0.97
means anything. Labels still in the corner frame are skipped with a warning; see
Option P.

### Option J — Phone live camera, configured from the laptop (DESIGN D11)

Point a phone at a real board and stream it live. The phone is a **dumb camera**: it
only films and streams, and it has nothing to press. You set everything — the crop
rectangle, the lens, the resolution, start and stop — on the laptop's `/control` page.
The node pushes each change to the phone the moment you make it, over the phone's
control socket, and the phone applies it; capture, crop, motion and adaptive
forwarding all still happen in the phone's browser. Nothing polls: the server logs
show one line per socket when it opens, not one per message.

`make dev` sets all of this up — it starts inference plus the viewer on **two**
ports (http :8080 and https :8443, one `node packages/viewer/ui/server/main.ts --host
0.0.0.0 --tls-port 8443` process), serves the **same recognizer `docker compose up` does** (it prints what it
serves, or a `!!` line when the ChessQ weights are missing and the stub serves), and
prints the URL for each page:

```bash
make dev
#   camera     https://<lan-ip>:8443/camera   (accept the warning once)
#   snapshot   http://<lan-ip>:8080/snapshot
#   live       http://<lan-ip>:8080
# the control page is http://localhost:8080/control (the viewer's banner names it too)
```

The camera page gets its own https origin because it is the only page that *needs*
one: `getUserMedia` is refused outside a secure context. The https bind serves every
page, though — every endpoint any of them touches is relayed by the viewer itself, so
there is no mixed content to block. `/`, `/snapshot` and `/control` are offered over
http purely to spare the desktop a self-signed-certificate warning.

To reach any of them from off the LAN, forward the http bind through a tunnel that
terminates TLS for you — see `.claude/skills/tunnel`.

By hand, if you want different ports:

```bash
# 1. Inference with real recognition (same env as Option F):
BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx \
  uv run board-reader-inference serve --host 0.0.0.0

# 2. Viewer with a TLS port — getUserMedia requires a secure context on a phone:
BR_INFERENCE_WS=ws://localhost:8000/viewer BR_INFERENCE_HTTP=http://localhost:8000 \
node packages/viewer/ui/server/main.ts --host 0.0.0.0 --port 8080 --tls-port 8443
# it prints the https://<lan-ip>:8443/camera URLs to type on the phone,
# and the /control URL to open on this machine

# 3. On the phone: open the printed URL, accept the self-signed certificate once.
#    (Android over USB instead: `adb reverse tcp:8080 tcp:8080`, then plain
#    http://localhost:8080/camera — localhost is a secure context, no cert.)
```

**The session, step by step:**

1. Run `make dev` on the laptop. Expected: it prints the camera, snapshot and live URLs.
2. On the phone, open `https://<lan-ip>:8443/camera` and accept the certificate warning
   once. Expected: the camera permission prompt.
3. Allow the camera. Expected: the picture fills the screen, and the status line at the
   bottom reads `board phone · node connected · not streaming · 1920×1080 · <lens>`.
4. Mount the phone over the board, in **landscape**. If it is upright, the status line
   adds `portrait: turn the phone sideways` (it still streams; upright framing costs
   about a quarter of perfect boards, [[operating-envelope#Framing]]).
5. On the laptop, open `http://localhost:8080/control` (or the **camera control** link on
   the live page). Expected: the chip reads `phone connected`, and within ~2 s the
   phone's whole view appears as a preview, with the phone's lenses in the **lens** list
   and its size under **frame delivered** in the Status panel. The phone sends previews
   only while a `/control` page is open (or not hidden); close the tab and they stop.
6. Drag a rectangle around the board on the preview, with a small margin, and press
   **Save crop**. Expected: at once the phone shows the same rectangle as an outline
   with the rest dimmed, and a warning on `/control` disappears or appears as the shape
   needs (whole frame sent, a rectangle taller than wide or more than twice as wide as
   tall, an upright phone).
7. Optional: choose another **lens** (an iPhone lists each rear lens, e.g. *Back Ultra
   Wide Camera*) or **resolution**. Expected: the phone's picture switches, and its status
   line names the new lens and size. The rectangle stays on the board, because it is
   stored in fractions of the frame.
8. Press **Start streaming** on `/control`. Expected: the phone's status line turns
   green, `streaming ~1.0 fps` at rest, and `/control` shows the cropped frame under
   **What the node receives**, with **frames/s received** and **last frame** filled in
   the Status panel and the **recognizer** named (`chessq`, or `stub` with the reason).
   Once the node has read the board, **board direction** in the Status panel names the
   side White sits on as the preview shows it, e.g. `White sits on the left (89°)`.
   (The stub reads no direction, so the line stays `—` without the ChessQ weights.)
9. Open the live page `http://localhost:8080` and move a piece. Expected: a burst on the
   phone (fps rises to ~12 while the hand moves) and the new position on the live page
   about two seconds after the hand leaves (three burst-end reads 0.3 s apart, then the
   0.5 s settle; in the offline sweep the mean from hand leaving to settled was 1.7 s).
10. Press **Stop streaming** on `/control`. Expected: the phone's status line reads
    `not streaming` within ~1 s, and no new frames reach the node.

What to check beyond the steps:
- Stability ~1.0 and ~1 fps with the board at rest; a burst the moment a hand enters
  the rectangle — and **no** burst for a hand moving outside it, since motion is
  measured on the crop only.
- Shake the phone gently in its holder while streaming: every frame carries an
  edge-measured `sharpness` (tenengrad on the sent crop), and when the estimator is
  behind, the node's inbox keeps the *sharpest* frame of the last second
  (`ingest.prefer_frame`). Verifiable in a recording: `frames.ndjson` headers carry
  `sharpness`, `events.ndjson` carries the chosen `frame_seq`.
- Restart the inference node mid-session (stop `make dev` with Ctrl-C, run it again).
  Expected: the phone's status line says `node unreachable`, then `node connected`, and
  the phone resumes streaming with the same crop, lens and resolution, with nothing
  pressed on `/control`. `serve` keeps each board's config in
  `data/camera-config.json` (`BR_CAMERA_CONFIG`; set it empty to keep the config in
  memory only, and then a restart comes back with streaming off). A corrupt file is
  ignored with a WARNING in the node's log, and the node starts with streaming off.
- Move the camera while streaming: turn the phone holder by a quarter turn, so the
  board lies a different way in the preview. Expected: after a few reads the
  **board direction** line names the new side and turns red, and a red
  `The camera moved…` warning appears under the preview. Redraw the crop around the
  board and press **Save crop**. Expected: the warning disappears, and the direction
  settles again on the new angle (a new lens or resolution does the same; starting or
  stopping the stream does not).
- Two boards: open `/camera?board=left` on one phone and `/control?board=left` on the
  laptop (or type the id into the **board** field). Each board keeps its own config.

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
force a two-kings read (swap the queen for a spare king mid-stream) and the panel shows one king, the repaired square ringed red,
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
> `--adaptive`, `--roi`, `--loop` and `--jpeg-quality` are
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
0.51 over `q` at 0.49" — a near-miss the read nearly answered, or a conviction the
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
  attempts, each config the phone applied (`config`, `capture`), the control link
  going up or down (`control-link`), `pagehide`. This is the file that says
  *why the node heard silence* — a page that never connected, a camera that
  froze, or a stream whose every frame was refused all look identical from the
  node's side. While streaming it also beats every 10 s (`kind: "heartbeat"`,
  with `sentPerSec`/`droppedPerSec`/`unacked`/`behind`), so a quiet stretch says
  which kind of quiet it was — `jq -c 'select(.kind=="heartbeat") | .detail'
  client-logs/*.ndjson`.
- `curl -s localhost:8000/healthz | jq .ingest` — `received/estimated/dropped/
  failed/refused/unscheduled/backstopped` counters; all zeros while the
  phone claims to stream means the frames never arrived. `unscheduled` far above
  `estimated` is the **read scheduler working** (`BR_SCHEDULE`),
  not a stall; `backstopped` above zero means that edge's `capture_ts` stopped
  advancing and the node is reading on its own clock to keep the board alive.

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
- `--changes-only` collapses the keepalive reads (one every ~2 s) to nothing while the
  board is at rest, and prints a burst of lines through a move;
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

### Option M — Score a trial recording

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

> [!important] Live event streams are SPARSE — pick `guard_seqs` for the schedule,
> not for the camera
> Since the move-trigger milestone the node does **not** publish one event per
> frame. `BR_SCHEDULE` defaults to `burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3`:
> three reads 0.3 s apart when a move finishes, then one every 2 s while the board
> rests — far fewer events than the frames that arrived
> ([the 08-27 sweep](reports/2026-08-27-move-trigger-sweep.md), [the settle tuning](reports/2026-11-24-settle-tuning.md)). `frames.ndjson` is
> unchanged — every arriving frame is still recorded, because recording happens in
> the `/ingest` handler ahead of the scheduler — so **`from_seq` still means a
> frame seq and is unaffected**, but `events.ndjson` now has holes in it.
>
> Two consequences when authoring truth:
>
> - `guard_seqs` counts *frames*, and it is there to drop the events captured
>   while the hand was still over the board. A burst-end policy already declines
>   those, so a big guard on a scheduled recording mostly throws away good
>   evidence: at ~0.5 reads/s, `guard_seqs=12` can be the entire segment. The
>   sweep's own replays used **`guard_seqs=3`** for exactly this reason; start
>   there for a scheduled take and only raise it if the excluded-count line shows
>   post-boundary events still landing mid-move.
> - a segment can legitimately contain **one** event, or none. `eval-session`
>   scores what is there; a segment with no events is not a bug, it is a move that
>   the keepalive had not yet come round to. Compare `events:` against the read
>   count in `/healthz` (`estimated` vs `unscheduled`) before concluding the node
>   was broken during the take.
>
> `BR_SCHEDULE=every-frame` restores the old per-frame stream if a take is being
> recorded specifically to be scored densely.

```bash
# 2. Score it (no model, no GPU — it reads events.ndjson, ~0.1 s):
uv run board-reader-inference eval-session recordings/20260813-193308
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
  Small margins with truth ranked second mean the read nearly had the answer;
  truth outside the top 2 means the evidence is not there at all;
- the **`frame:`** line says how many raw reads agree with the truth unturned; a
  `! WARNING ... turned` line means the truth file is not in the chess frame
  (Option P);
- **`timing`** is the rate story a placement score cannot show — frames in (fps),
  reads out (reads/s), frames per read, and the capture→publish latency a live
  display actually feels. It reads the frame *index* only, never the JPEGs.

Exit codes: `0` scored, `1` nothing left to score, `2` no `truth.json` (skeleton
printed) or a `truth.json` that cannot be trusted.

### Option N — Replay a take through today's stack (offline benchmarking)

Option M grades the events a run *published*. This one produces fresh ones: it
pushes the recorded frames back through the recognizer the environment builds
today — the same read, the same prior repair, the same debounce, the same event
writer the live gateway uses — and writes them as a new
recording-shaped directory. **A recorded take replaces a live one for
iteration:** change the code, replay, score, diff. No phone, no board, no server.

```bash
# Whatever `serve` runs is what replay runs — same env vars, same builder:
BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx \
uv run board-reader-inference replay recordings/20260813-193308 --score
```

`make gate GATE_NAME=<name>` runs this over the four gate takes and prints the table
(Option P, step 4).

**The debounce's knobs.** `--settle-seconds`, `--settle-min-reads` and
`--stale-windows` replay the same frames under another settle (`serve` reads the
same knobs from `BR_SETTLE_SECONDS`, `BR_SETTLE_MIN_READS`,
`BR_SETTLE_STALE_WINDOWS`; replay falls back to those variables too).

1. Run `uv run board-reader-inference replay recordings/20260828-111421 --policy
   burst-end --settle-seconds 1 --settle-min-reads 1 --name settle-check --score`.
   Expected: a `settle:    1 s, >= 1 reads, stale after 2 windows` line, and
   `replays/settle-check/manifest.json` carries `settle_seconds: 1.0`,
   `settle_min_reads: 1`, `stale_windows: 2`. Without `--name` on a clean tree the
   directory name ends in `-settle1s-1r-2w`.
2. Start `BR_SETTLE_SECONDS=1 make dev` and run `curl -s localhost:8000/healthz`.
   Expected: `"settle": {"settle_seconds": 1.0, ...}`; the node log prints
   `settling with {...}`. `BR_SETTLE_MIN_READS=x` makes the node refuse to start.

Automated: `test_stabilize.py` (`SettleConfig`, its env, the deadline),
`test_replay.py` (name, digest, manifest, the CLI flags), `test_ingest_stream.py`
(each board's debounce, `/healthz`).

It lands beside the take it came from, so a benchmark can always be traced to its
input — and the source recording itself is never written to:

```
recordings/20260813-193308/replays/<name>/manifest.json  # source, git, BR_CHESSQ, digest
                                        /events.ndjson   # the fresh reads
                                        /frames.ndjson   # the replayed lines (thin)
                                        /jpeg -> ../../jpeg          (symlink)
                                        /truth.json -> ../../truth.json
```

`<name>` defaults to `<git describe>-<digest of BR_CHESSQ, including the model
file's size and mtime>`: the same code against the same models replays into
the same directory (a re-run supersedes its own answer), a different model — or
the same path retrained in place — lands beside it. **The default is only offered
on a clean tree**: two sets of uncommitted changes describe identically, so they
would replay into one directory and the second run would destroy the baseline the
first was meant to be diffed against. A dirty tree is asked for `--name` (one path
component, no `/` and no `..` — the replay's symlinks hop a fixed two levels).
`--score` runs Option M's scorer on the result straight away; without it the
command prints the `eval-session` line to run.

> [!warning] Replay vs replay, never replay vs the recorded run
> The live gateway's latest-wins inbox is a **quality** filter, not just a
> staleness one: `ingest.prefer_frame` keeps the *sharper* of two contenders
> within a one-second window, because a handheld phone smears half its frames. A
> live run's recorded score is therefore measured on a blur-filtered subset of
> its own frames, while a replay reads every frame the policy asks for — blurry
> ones included. **A replay-vs-recorded raw delta is not a code result**: the two
> numbers come from different frame populations and the replayed one is
> systematically pessimistic. Treat it as directional at most. (Takes recorded
> before 2026-11-24 have a second gap: the old stack's calibration swapped the
> reader mid-take, which a replay cannot reproduce.)
>
> The comparison that means something is **two replays of the same recording**:
> same frames, same order, same freeze — so a difference between them is a
> difference in the code or the models. Diff replays; quote the recorded run only
> as the run that happened.

What to check:

- **the replay is deterministic** — same recording, same code, same env gives a
  byte-identical `events.ndjson` (modulo the model's own float nondeterminism), so
  two replays can be diffed as a claim about the *code*. Everything time-based
  runs on each frame's recorded `capture_ts`, never the wall clock: a 1 s settle
  window is 1 second *of the recording* whether the replay takes ten minutes or
  ten milliseconds;
- **more events than the run published** (08-13: 748 against the run's 85), for the
  reason in the warning above. The report's `timing` block and its settled row are
  therefore denser than the live ones — the debounce is time-windowed rather than
  count-based so that stays as comparable as it can be, which is not the same as
  being the same measurement;
- **`published_ts` is absent** and the report says `latency not stamped` — nothing
  was published, and a fabricated capture→publish number is worse than none;
- **multi-session recordings**: a reconnecting edge restarts `seq` at 0, and truth
  is a step function over `seq`, so only the **longest** session is replayed and a
  warning names the others. `--all-sessions` replays them all into one file;
- `--limit N` stops after N estimates and leaves the rest of the take **unread** —
  a smoke run when the GPU is busy. The result is a shorter take, not a sparser
  reading of a long one: the unread frames are not in the replay's `frames.ndjson`
  either, or `eval-session` would report a run that read one frame in 249;
- `--board <id>` picks one board out of a recording that holds several.

> [!note] Which frames get estimated — `--policy`
> The loop asks a *frame policy* per frame (`board_reader_inference.stream.schedule`).
> `--policy every-frame` is the default and the control; `burst-end` is the
> scheduler under test — estimate when a move finishes, skip mid-burst, and read a
> keepalive floor at rest. Spec strings are `name[:key=value,...]`
> (`--policy burst-end:keepalive_s=5,burst_end_reads=3`), and a policy's name
> round-trips, so the manifest's `policy` field is literally the argument that
> reproduces the run. A policy sees the recorded header (`seq`, `capture_ts`,
> `stability`, `sharpness`, `burst_id`) and not the pixels, so a declined frame
> costs not even a JPEG read — and "estimate this one" is a request, not a
> guarantee (the frame may have no JPEG behind it; that lands in its own counter
> and is not charged to the policy).
>
> **`--restamp-stability` is mandatory when scheduling the older takes.** Anything
> recorded before the 2026-08-27 evening meter fix carries the old absolute-scale
> meter's `stability`, which read <0.35 on every frame on the deployment phone; a
> policy scheduled against it measures the bug. The flag re-runs today's meter over
> the recording's own JPEGs in seq order and overrides the field before the policy
> sees it (deterministic; marked `stability_restamped` in the replay's
> `frames.ndjson`). It costs one JPEG decode per frame *including declined ones* —
> the meter must see every frame — and must be applied uniformly across a sweep or
> not at all. Nothing but the policy reads the field, so a restamped every-frame
> replay publishes the same boards as an unrestamped one; only the advisory
> `stability` on the event differs (pinned by
> `test_replay.py::test_restamping_changes_the_schedule_and_nothing_else`).
> Sparse policies and restamps join the default replay directory name, so a sweep's
> rows never land on top of each other.

### Option O — ChessQ Lite, the recognizer (`BR_CHESSQ`)

`BR_CHESSQ=<model.onnx>` serves Lichess's ChessQ Lite: one model reads the whole
frame and finds the board and its orientation itself (DESIGN D10). Both entry
points set it: `make dev` to the fp32 export (GPU), `make dev-cpu` and compose to
the author's int8 file (CPU). The weights are not in the repo; `make models`
fetches the int8 ONNX and the FP16 safetensors from the pinned revision their
committed descriptors name, checks the SHA-256, and exports the fp32 ONNX.

0. Run `make models`. Expected: `ok cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx
   (sha256 verified)` and the same for the `.safetensors` when the files are already
   there, `fetched ...` otherwise; then either `exported cq-vit-s-644-v6-seed1-onnx-fp32.onnx
   (... 192/192 squares agree with the package ...)` or, when it is current,
   `ok cq-vit-s-644-v6-seed1-onnx-fp32.onnx (export sha256 verified against its
   descriptor)`. Run it again: every line says `ok`, nothing is downloaded or
   exported, and `git status` shows no change to `models/chessq/*.json`.
1. Start the stack: `make dev`. Expected: `recognition ChessQ Lite V6
   (models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx)`, and the inference log
   prints `serving ChessQ from BR_CHESSQ=...`. `make dev-cpu` names the int8 file
   instead.
2. Run `curl -s localhost:8000/healthz`. Expected: a `"recognizer": {"name":
   "chessq", "model": ...}` entry naming the file.
   **Missing weights:** `make dev BR_CHESSQ=models/chessq/absent.onnx`. Expected:
   `!! ... missing — the node serves the STUB recognizer; run 'make models'`, the
   node starts anyway, its log prints `!! BR_CHESSQ=... does not name a model file`,
   and `/healthz` shows `"recognizer": {"name": "stub", ..., "fallback": "..."}`.
   The live page shows the start position whatever the camera sees.
3. Open `/camera` on the phone (Option J) and point it at a board. Expected: each read
   takes ~0.2 s (ChessQ fp32 on the GPU; int8 ~0.6–1.1 s on CPU), and a moved piece
   settles on the live page about two seconds after the hand leaves (the burst-end
   reads; with no burst, the 2 s keepalive gives 2–3 s).
4. Replay a take with ChessQ (Option N), with **only** `BR_CHESSQ` set:
   `BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx uv run
   board-reader-inference replay recordings/20260828-111421 --policy burst-end
   --name cq-v6 --score`. Expected: the manifest's `estimator` is
   `ChessQRecognizer` and its `env` lists `BR_CHESSQ`.

Automated: `packages/inference/tests/recognition/test_chessq.py` and `test_recognizer.py` (fake
model and the stub; the one real-weights test is `gpu`-marked and runs under `make
test-gpu`, skipping when the weights or the held-out still are absent),
`tests/test_fetch_models.py` and `tests/test_chessq_export.py` (`make models`:
the downloads, the export's source check and descriptor) and
`tests/test_deploy_parity.py` (both entry points serve ChessQ V6, fp32 exactly where
a GPU is declared, and nothing of the old stack). The e2e stream test
uses `tests/_sim_recognizer.py` instead, because ChessQ does not read sim renders.

### Option P — The chess frame: label, eval and gate a recognizer (DESIGN D10)

A recognizer reads the chess frame and finds the orientation itself, so labels,
eval and replays are all in the chess frame.

1. Under ChessQ (Option O), photograph a board on `/snapshot`, correct any wrong
   squares, and press **Save labelling**. Expected: the label is saved (`/labels`
   answers `"frame": "chess"`), and `data/uploads/<id>.label.json` carries
   `"frame": "chess"` and `"corners": null`.
2. Run `BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx uv run
   board-reader-inference eval data/eval data/uploads`. Expected: the header names
   `ChessQRecognizer(...)`; labels still in the corner frame are skipped with a
   warning that names `tools/migrate_labels_to_chess_frame.py`; a `frame:` line says
   how many boards were read in the truth's frame and lists any `TURNED` board.
3. Run `uv run board-reader-inference eval-session recordings/20260828-111421/replays/cq-v6`.
   Expected: a `frame:` line, `21/21 raw reads agree with the truth unturned`
   (a `! WARNING ... turned 180°` line instead means the truth file is not in the
   chess frame).
4. Run `make gate GATE_NAME=<new-name>` on an idle GPU. Expected: four
   `replaying ...` lines (about a minute in all), then the gate table
   `ka2-tuned → <new-name>` with a total row. An unchanged pipeline prints every
   cell once (no `→`): 110/119 raw, 110/119 prior, 104/119 settled perfect.
   `uv run python tools/gate_table.py tkr-shipped cq-v6` prints the adoption gate.

Automated: `test_labels.py`, `test_server_labels.py`, `test_evaluate.py` and
`test_trial.py` (frame sections), `tests/test_gate_table.py`.

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
