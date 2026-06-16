# Testing board-reader

Two layers, both required by the project's "user-testable from the start" principle
(see [../CLAUDE.md](../CLAUDE.md)):

1. **Automated tests** — fast, run on every change.
2. **The live demo** — a human runs the whole pipe and *sees* it work.

---

## 1. Automated tests

```bash
uv sync                # once, to set up the workspace venv
uv run pytest          # the whole suite (should be green)
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
