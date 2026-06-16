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

What's covered today (M0):

| Area | File | Proves |
|---|---|---|
| L3 protocol | `packages/core/tests/test_protocol.py` | placement⇄distribution, validation, JSON roundtrip |
| Wire format | `packages/core/tests/test_wire.py` | JPEG + header frame roundtrip |
| Sim renderer | `packages/core/tests/test_sim.py` | board images render |
| Frame sources | `packages/edge/tests/test_source.py` | video/dir/synthetic/repeat sources |
| Edge adapter | `packages/edge/tests/test_adapter.py` | frames → wire envelopes, throttle |
| Estimator | `packages/inference/tests/test_estimator.py` | stub emits valid events, stateless |
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

> Note (M0): the estimator is a **stub** that always reports the start position —
> it ignores pixels. Real per-frame recognition arrives in **M1** (DESIGN §7).
> Until then, feeding real video exercises the *transport + timing + viewer* path,
> not recognition.

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
