---
aliases: [runbook, ops, how to run it]
tags: [board-reader/ops]
---

# Runbook

How to run the system and what to check after touching it. For the *test* path — automated suite and demo walkthrough —
see [[TESTING]].

## Running it

```bash
uv sync                                   # the whole workspace + dev tools
make models                               # the ChessQ weights (pinned, SHA-256 checked) + fp32 export + -attn copies
make test                                 # must be green before any commit
```

`make models` does three things. First it reads the release descriptors committed
under `models/chessq/` and downloads the author's int8 ONNX and FP16 safetensors
from the pinned Hugging Face revision; a file whose SHA-256 already matches is left
alone, and a download that does not match replaces nothing. Then
`tools/chessq_export` (its own uv project with PyTorch and the pinned official
`chessqueries` package) exports the safetensors to
`cq-vit-s-644-v6-seed1-onnx-fp32.onnx`, checks that its per-square argmax equals
the package's own prediction on a few drawn boards, and writes
`cq-vit-s-644-v6-seed1-onnx-fp32.json` (source URL and SHA-256, package commit,
library versions, export SHA-256). The export is byte-reproducible under the lock,
so the committed descriptor names the exact bytes; an export that is already
current is left alone. The first run installs that environment (~1 GB) once.
Last, `tools/chessq_export/attention.py` writes the files the node serves:
`cq-vit-s-644-v6-seed1-onnx-fp32-attn.onnx` and `…-onnx-int8-r1-attn.onnx`, the same
graphs with one more output, the square attention Auto-frame locates the board with
(DESIGN D20). It checks that `logits` and `direction` stay bit-identical to the source
file and writes `<release>-attn.json` (source and copy SHA-256); a copy that is
current is left alone (`ok ... (sha256 verified against its descriptor)`).

**Which file serves.** One release, two files, chosen by hardware, each as its
`-attn` copy: on a GPU the fp32
export (~205 ms per read on the GTX 1650), on a CPU the author's int8 (~660 ms,
against ~990 ms for fp32 on the CPU). The int8 file runs *slower* on CUDA (600+ ms)
because its integer matmuls have no CUDA kernels. Both read every gate frame
identically. `make dev` and `make gate` serve fp32; `make dev-cpu` and `compose.yaml`
(which declares no GPU device) serve int8 ([[2026-11-24-chessq-gpu-formats]]). Without the weights every entry point still starts, but the node serves
the **stub recognizer** (a fixed start position): it prints `!! BR_CHESSQ=... does
not name a model file; run `make models`` at startup, and `GET /healthz` carries a
`recognizer.fallback` with the same reason.

**The canonical demo** — three containers, `edge → inference → viewer`:

```bash
docker compose up --build                 # then open http://localhost:8080
```

**Locally, three terminals:**

```bash
uv run board-reader-inference serve       # gateway  :8000
make ui && node packages/viewer/server/main.ts --port 8080   # UI :8080 -> open it
uv run board-reader-edge feed             # synthetic feed
uv run board-reader-edge feed --video clip.mp4 --throttle   # or a real clip
```

**The phone flow** — `make dev`, then open `/snapshot` or `/camera` on the phone. The
camera page has no controls, and the node frames the camera by itself (Auto-frame,
DESIGN D21): on connect it asks the phone for one whole frame, finds the board and sets
the rectangle, and sets it again when the phone moves. Open `/control` on the laptop to
start or stop the stream, pick the lens and resolution, or draw the rectangle by hand,
which switches Auto-frame off for that camera until you switch it on again or press
**Re-detect frame** (DESIGN D11; the walkthroughs are [[TESTING#Option J — Phone live camera, configured from the laptop (DESIGN D11)]]
and [[TESTING#Option T — Auto-frame: the node frames the camera by itself (DESIGN D21)]]).
When Auto-frame says "no board found", the board is not in the picture or too small
in it (under about a sixth of its width); it retries after 3 s, 10 s, then every 30 s.
The node log names every attempt (`auto-frame board-N: ...`).
The node assigns each phone its board id (`board-1`, `board-2`, ...; DESIGN D18) by a
device id the phone keeps, for good; `/control` lists the phones, names them (the live
page shows the name; the board id stays), forgets one that is no longer used, and shows
a QR code that opens `/camera`. `serve` keeps each board's config and each phone's board
id in `data/camera-config.json` (`BR_CAMERA_CONFIG`; empty keeps them in memory only),
so a restart gives every phone its id and framing back. A phone in a private window
keeps no id, and each reload makes it a new camera (its status line says so); forget
the old ones on `/control`. A phone page opened before the update says it is out of date: reload
it. By hand:

```bash
BR_CHESSQ=models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32-attn.onnx \
BR_SAVE_UPLOADS=data/uploads \
uv run board-reader-inference serve --host 0.0.0.0
```

Take a photo, and the position appears. No taps required: ChessQ finds the board
and its orientation in the whole frame (DESIGN D10), so corners are not needed.

**A YouTube broadcast (or any stream) as a camera** (DESIGN D19). With a node running
(`make dev`):

```bash
uv run board-reader-edge camera "https://www.youtube.com/watch?v=TDx2DGiBGhc" --start 1:00:00
uv run board-reader-edge camera "https://www.youtube.com/@<channel>/live"   # a live stream
uv run board-reader-edge camera rtsp://cam.local/stream                     # any FFmpeg URL or a file
```

It appears on `/control` like a phone, named by the stream's title, and Auto-frame sets
its crop (a broadcast's board is usually too small in the picture to watch for camera
moves: after a cut to another angle, press **Re-detect frame**); turn streaming on. A video plays once at real time from `--start`
(seconds or `h:m:s`) whether or not it streams; a live stream reconnects when it drops.
The device id sits in `~/.local/state/board-reader/edge-device-id`, so the camera keeps
its board id across restarts; run a second camera with its own `--state <file>`.
`--node ws://host:8000` points it at another node. Lens and resolution on `/control`
do nothing for a stream; `--max-height 720` makes it pick a smaller format (1080p30
costs about half a core here).

Needs **deno** for YouTube (`sudo pacman -S deno`): yt-dlp runs its challenge scripts
(the `yt-dlp-ejs` package) in it, and without it formats go missing. No ffmpeg binary
is needed; the opencv wheel's FFmpeg reads YouTube's https and HLS formats. yt-dlp
has its own seasoning date in `pyproject.toml` (`exclude-newer-package`), because
YouTube breaks old releases within weeks: when the command says yt-dlp could not
resolve a page that plays in a browser, bump that date and the floor in
`packages/edge/pyproject.toml`, then `uv sync`.

**What the compose demo reads (D3, measured 2026-11-24).** The compose edge streams
`core.sim` synthetic renders (letters on coloured squares), and ChessQ does **not**
read them: on the 14 positions of `SAMPLE_GAME_PLACEMENTS` it scores 0/14 perfect
boards and 79 % per square at the feed's 48 px squares (58 % at 96 px, 80 % on
`simulate_angled_view`). Every empty square reads empty; only 250 of 436 pieces
read right (the start position comes back `r.bk.bnr/pppppppp/8/8/8/8/P.P...PP/RNBK...R`).
So the compose demo proves the pipe — frames in, a changing position out, the
settled board, the stream bursting through each synthetic hand — but not
recognition. Use a real board on `/camera`, or replay a recorded take, to see
recognition.

## Environment

> [!important] Both entry points must agree on the recognition composition
> Two things start a real node — `docker compose up` (compose.yaml) and `make dev`
> (the Makefile) — and each picks its model out of the variables below. They must
> choose the *same* one: once a model change landed in compose alone, and every live
> phone test silently rehearsed a weaker reader than the demo. A new deploy-time variable therefore belongs in
> **both** files, or in `tests/test_deploy_parity.py`'s recorded list of
> deliberately environment-specific ones.
>
> Since DESIGN D10 both set `BR_CHESSQ` and nothing else that shapes recognition.
> `make dev` prints what it serves on startup (`recognition ChessQ Lite V6 (<file>)`,
> or `!! <file> missing — the node serves the STUB recognizer; run 'make models'`).

**Inference node**

| Variable | Effect |
|---|---|
| `BR_CHESSQ` | **The recognizer, set by both entry points** (DESIGN D10). ChessQ Lite ONNX → `ChessQRecognizer` (the `recognizer.py` seam), which reads the whole frame and finds the board and its orientation by itself. The session takes CUDA when onnxruntime can use it and the CPU otherwise; serve the fp32 file on a GPU and the int8 file on a CPU (see *Which file serves* above). Unset or a missing file degrades to the stub recognizer, loudly (startup banner, `recognizer.fallback` on `/healthz`); `make models` fetches and exports both. `GET /healthz` reports `recognizer: {name, model}`. Weights are PolyForm Noncommercial and live in `models/chessq/` (THIRD-PARTY.md). Gate result: [[2026-11-24-chessq-v6-replay-gate]]. |
| `BR_SCHEDULE` | Which arriving frames get an estimate, as a `replay --policy` spec — the same string a sweep measured. Default (set by `serve`, also `--schedule`): `burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3,exit_hold_s=0.3`, i.e. a burst ends after 0.3 s of calm (the phone holds 0.7 s and keeps sending at burst rate meanwhile) and three reads 0.3 s apart follow (enough for the 0.3 s settle window to adopt the move on its own evidence, [latency round 3](reports/2026-11-25-latency-round3.md); this relies on the energy gate, see `BR_OBSCURED_ENERGY`) plus a 2 s keepalive floor ([live latency](reports/2026-11-24-live-latency.md): 3 s left a move 3–4 s to settle whenever no burst opened). The one-read-per-move `burst-end` the 08-27 sweep measured read ~12× fewer frames than `every-frame` for −0.9 pp prior per square on the old stack ([the sweep](reports/2026-08-27-move-trigger-sweep.md)); three reads per move cost a little more. **The live event stream is sparse by design** — `GET /healthz` shows `ingest.unscheduled` climbing while `estimated` crawls, and that is the system working. `every-frame` is the way back. `ingest.backstopped > 0` means an edge's `capture_ts` stopped advancing and the node is reading on its own clock to stay alive. |
| `BR_SETTLE_SECONDS`, `BR_SETTLE_MIN_READS`, `BR_SETTLE_STALE_WINDOWS` | The settle debounce (`stabilize.SettleConfig`): a square changes after reading the same new class for `SECONDS` of capture time on at least `MIN_READS` reads; a square the board-coherence rule blocks falls through after `STALE_WINDOWS` windows. Unset keeps the defaults in `stabilize.py` (0.3 s, 2 reads, 2 windows since [latency round 3](reports/2026-11-25-latency-round3.md); the edge forwarder keeps its burst rate through its 0.7 s hold and a 1 s tail after motion stops, so those reads have frames to land on). The 0.3 s window relies on the energy gate (`BR_OBSCURED_ENERGY`); a malformed value stops the node at startup. `GET /healthz` reports them under `settle`; `replay --settle-*` takes the same knobs. |
| `BR_OBSCURED_ENERGY` | The obscured-read gate (`SettleConfig.obscured_energy`, DESIGN D16): a read whose ChessQ frame energy (minus logsumexp of each square's logits, mean of the 4 highest squares) is above this value is *obscured*, usually a hand over the board. It still counts as evidence for the settle debounce but commits no square, and its event carries `obscured: true` (the live card's "hand?" chip). Default `-12.4`, fitted for ChessQ V6 ([hand-signal probe](reports/2026-11-24-hand-signal-probe.md)). **Re-fit it for every model release**: the energy lives on the model's logit scale. `off` switches the gate off. **The shipped schedule and settle window rely on this gate**: with it off, the 0.3 s hold and 0.3 s window let hands settle (6 on the gate takes, 12 wrong boards while a hand rests on live take 2, [latency round 3](reports/2026-11-25-latency-round3.md)), and the node does not fall back by itself. With `off`, also set `BR_SETTLE_SECONDS=0.5` and `BR_SCHEDULE=burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3` (round 2's 0.7 s hold). A malformed or non-finite value stops the node at startup. Neither entry point sets it (the default is in code). `GET /healthz` reports it under `settle`; `replay --obscured-energy` takes the same value. |
| `BR_SAVE_UPLOADS` | Directory for retained phone photos + sidecars (placement, the frame it was read in). This is what turns field shots into eval data. |
| `BR_RECORD_DIR` | Directory for recorded ingress sessions. `serve` sets it to `recordings/` by default, so **every served run is replayable**; `serve --no-record` turns it off. |
| `BR_RECORD_MAX_MB` | Disk-fill stop for the recorder (default: no cap). The session truncates rather than rotating. |
| `BR_GPU_MEM_LIMIT_MB` | Ceiling on each ONNX session's CUDA arena (default `1024`). Serving builds one session, ChessQ's. `0` removes the cap. See [the GPU section](#stopping-it-and-the-gpu-it-leaves-behind) for why a display GPU needs one. |
| `BR_ALLOW_ORIGINS` | CORS origins, comma-separated (default `*`). |

### Reproducing a run

Every frame that reaches `WS /ingest` is recorded — the CLI feed, the Pi, and the
phone `/camera` page alike, since all three cross that one socket. A session is a
directory of `manifest.json` + append-only `frames.ndjson` + the JPEGs exactly as
they arrived.

```bash
curl -s localhost:8000/healthz | jq .recording    # is it recording, and where?
uv run board-reader-edge feed --recording recordings/ --throttle   # re-run it
```

Every published event is recorded beside the frames with an `evidence` digest —
the top-2 classes and their probabilities for all 64 squares — so a post-mortem
can ask how close a wrong square came without putting the model back in the loop.
Scoring a session (`eval-session`) writes `<recording>/report.json` every time, so
no number from a run exists without the artifact behind it.

`--recording` re-sends the stored bytes verbatim, so the replayed run sees the same
pixels the original did. Prefer `--throttle`: a flat-out replay reads the same
frames, but it can push the read queue into its overload rule (keepalive reads
dropped after 5 s of waiting), which the recorded run never saw. Anything that would reprocess the stream
(`--adaptive`, `--roi`, `--loop`, `--jpeg-quality`) is refused
with `--recording` rather than silently changing what goes on the wire.

### Capturing a run's positions

Recording captures a run's **input** (the frames); `positions` captures its **output** —
the L3 position stream itself, headlessly, without a browser in the loop.

```bash
uv run board-reader-inference positions                        # tail it, readably
uv run board-reader-inference positions --out run.ndjson --quiet   # capture it
uv run board-reader-inference positions --board A1 --changes-only  # one board, moves only
```

`--out` appends each event as one line of the exact protocol JSON the socket carried, so a
capture *is* the stream and two runs can be diffed — it *appends*, so give each run its own
file rather than reusing one. `--url` points it at a gateway elsewhere (default
`ws://127.0.0.1:8000/viewer`). Exit code `1` means the stream produced nothing and `2` that
no gateway answered — both scriptable, so a scripted demo fails rather than reporting
success over silence.

Read it as *positions only*: the tail diffs consecutive placements for legibility and
smooths nothing, so a flickering square shows a transition on every frame. See
[[architecture#Temporal consistency — the layer nobody has built]].

**Viewer node** — a Node server, `node packages/viewer/server/main.ts` (Node 22.18
or newer runs the TypeScript directly), serving the build `make ui` writes to
`packages/viewer/build/`. It refuses to start without that build and says what to run.

| Flag | Effect |
|---|---|
| `--host` | Bind address (default `127.0.0.1`; `0.0.0.0` to reach it from a phone). |
| `--port` | The plain http port (default `8080`). |
| `--tls-port` | Also serve https on this port, with a self-signed certificate cached in `$XDG_CACHE_HOME/board-reader/viewer-tls/` and reused until it nears expiry or stops naming this machine's addresses. `/camera` on a phone needs it. `make dev` passes `8443`. |

| Variable | Effect |
|---|---|
| `BR_INFERENCE_WS` | The node's `/viewer` socket URL; its origin is where the viewer relays every `/ws/<path>` socket (`/ws/ingest`, `/ws/viewer`, `/ws/cameras/{device}`, `/ws/cameras`, `/ws/boards/{id}/control`, and any the node adds). |
| `BR_INFERENCE_HTTP` | Where the *viewer* forwards `/api/<path>` requests. |
| `BR_CLIENT_LOG_DIR` | Where the phone's trial notes (`POST /api/client-log`) are appended, one NDJSON file per viewer start (default `client-logs`; empty drops them). |
| `BR_PUBLIC_URL` | An origin that serves the viewer over TLS from outside the LAN, such as a Cloudflare quick tunnel (`/tunnel` skill): `BR_PUBLIC_URL=https://<name>.trycloudflare.com make dev`. The camera QR code on `/control` (`GET /api/camera-links`) offers its `/camera` first, before the LAN, hostname and `localhost` addresses on the TLS port. Only the origin is kept. |

The viewer logs its startup banner and one line per relay or server error, nothing per
request. `pnpm dev` in `packages/viewer` (Vite, hot reload) relays `/api` and `/ws` to
the same two variables, so it works against a running node.

> [!note] Neither reaches the browser
> The pages are same-origin and relative throughout, so these address the viewer's own
> hop to the node. Point them at the node wherever it actually is — `127.0.0.1` for
> `make dev`, the service name under compose. To reach the stack from off the LAN,
> forward the viewer's port rather than reconfiguring these (`.claude/skills/tunnel`).

> [!warning] `models/` and `data/` are git-ignored
> Not an oversight — the ChessQ Lite V6 weights are PolyForm Noncommercial, so they are
> **not redistributable**; only their release descriptors under `models/chessq/` are
> committed (THIRD-PARTY.md). A fresh clone has no weights until `make models`; the
> server degrades to the stub and still composes up.

## Stopping it, and the GPU it leaves behind

`make stop` signals both servers, waits, escalates to SIGKILL, and prints the pids
it actually killed — it exits non-zero rather than claiming success it cannot
verify (`tools/stop.sh`). That matters more than it sounds:

> [!warning] A released port does not mean the process is gone
> An inference server can wedge in uvicorn's shutdown behind an in-flight ONNX
> call: it stops listening — so the port frees and nothing looks wrong — while
> staying resident with its **CUDA context and hundreds of MB of VRAM**. One sat
> that way for two hours on 2026-08-13; the next `make dev` inherited a card too
> full for its second ONNX session and **aborted mid-stream** (status 134),
> taking a live phone test with it. If a start fails with `BFCArena` or
> `CL_MEM_OBJECT_ALLOCATION_FAILURE`, look before restarting:
>
> ```bash
> nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
> make stop        # names what it killed; SIGKILLs what ignores SIGTERM
> ```

`make dev` also runs with `OPENCV_OPENCL_RUNTIME=disabled`. OpenCV's OpenCL
backend shares the card with onnxruntime, and it is the one that turns a tight
GPU into a *crash* rather than a degradation: a failed `warpAffine` allocation
throws a `cv::Exception` through C++ `terminate`, which no Python handler can
catch. The image work left (one resize per read) costs nothing on the CPU worth that
risk. This is **not** mirrored in `compose.yaml`, which declares no GPU device and
runs the same recognizer on CPU — the two entry points share an environment, not a
hardware envelope.

### When the card is also driving your screens

> [!warning] The display cannot fall back to CPU
> On a laptop the GPU this node uses is usually the display pipeline too, and an
> external monitor is often the worst case: with the common Optimus layout
> (`xrandr --listproviders` showing NVIDIA as **Sink Output** only) the screen is
> rendered by the iGPU and copied into a **reverse-PRIME surface in NVIDIA VRAM**
> for scanout. That surface is reallocated on DPMS wake, resolution changes and
> monitor re-probes — and if the card is full at that moment, X has nowhere to
> degrade to. On 2026-08-13 a burst of model work took the VRAM and a 4K monitor
> went dark twice, once losing its mode pool entirely:
>
> ```
> (WW) NVIDIA(G0): Failed to allocate reverse prime surface: out of memory.
> (EE) NVIDIA(G0): Failed to allocate memory for composition pipeline: out of memory..
> (EE) NVIDIA(GPU-0): Unable to add "nvidia-auto-select" mode to ModePool
> ```
>
> Recover without a reboot by forcing a smaller surface and back:
> ```bash
> xrandr --output HDMI-1-0 --mode 1920x1080 && xrandr --output HDMI-1-0 --mode 3840x2160
> ```

Two defences, and they are different in kind:

* **`BR_GPU_MEM_LIMIT_MB` (default 1024, always on).** Caps each session's CUDA
  arena so a runaway extension fails as a *catchable* ORT error rather than by
  taking the display's headroom, and pins two non-default ORT knobs that exist
  for the same reason: `arena_extend_strategy=kSameAsRequested` (the default
  rounds every extension up to the next power of two and never gives it back)
  and `cudnn_conv_algo_search=HEURISTIC` (the default benchmarks every conv
  algorithm with large transient workspaces).
* **`make dev-cpu` (opt in).** The same stack — same release, same ports — with `CUDA_VISIBLE_DEVICES=` empty, which is CUDA's "no devices", and the int8 file, the faster one on a CPU. The
  node then never touches the card at all, so the display's VRAM is not a shared
  resource. ChessQ on CPU (int8) takes ~660 ms per read against ~205 ms for fp32 on
  CUDA ([[2026-11-24-chessq-gpu-formats]]), and it is exactly what `compose.yaml` runs. Use
  it when the machine running the node is also the machine displaying it.

`make dev` prints which one you are on (`gpu visible — onnxruntime capped at …` /
`gpu hidden — the same recognizer on CPU …`).

## Verifying a restart

> [!danger] `healthz` proves nothing about recognition
> A server that silently serves the stub, or a wrong model file, is perfectly healthy and
> serves garbage. Check `recognizer` in `/healthz` (no `fallback`, the expected `model`),
> then read a known photo. Two incidents with the earlier calibration bank were missed
> exactly this way.

**Verify with a known photo.** Post a retained upload whose expected score you know and
check it still reads at that accuracy:

```bash
curl -s -X POST --data-binary @data/uploads/<known>.jpg \
  http://localhost:8000/estimate | jq .placement
```

A labeled photo has its truth in `data/uploads/<known>.label.json`; `eval` scores the
whole set at once ([[measuring-accuracy]]).

**Then check the GPU:**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

When the GPU is busy, run the test suite with `CUDA_VISIBLE_DEVICES="" uv run pytest`
(`make test` already does), or a real ONNX session fails with a BFCArena OOM that looks
exactly like a code regression.

## Related

- [[architecture]] — what each process is
- [[operating-envelope]] — whether a bad read is a bug or the framing
- [[TESTING]] — the full manual test guide
