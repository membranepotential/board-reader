---
aliases: [runbook, ops, how to run it]
tags: [board-reader/ops]
---

# Runbook

How to run the system, what to check after touching it, and how to recover when the
calibration bank goes bad. For the *test* path — automated suite and demo walkthrough —
see [[TESTING]].

## Running it

```bash
uv sync                                   # the whole workspace + dev tools
uv run pytest                             # must be green before any commit
```

**The canonical demo** — three containers, `edge → inference → viewer`:

```bash
docker compose up --build                 # then open http://localhost:8080
```

**Locally, three terminals:**

```bash
uv run board-reader-inference serve       # gateway  :8000
uv run board-reader-viewer --port 8080    # UI       :8080  -> open it
uv run board-reader-edge feed             # synthetic feed
uv run board-reader-edge feed --video clip.mp4 --throttle   # or a real clip
```

**The phone flow** — serve on the LAN and open `/snapshot` on the phone:

```bash
BR_MODEL=models/run_e/model.onnx \
BR_SESSION_BANK=models/exemplars_session.npz \
BR_SAVE_UPLOADS=data/uploads \
uv run board-reader-inference serve --host 0.0.0.0
```

Take a photo, and the position appears. No taps required — corners are auto-detected,
orientation is auto-resolved, and calibration bootstraps itself the first time you
photograph a start position.

## Environment

> [!important] Both entry points must agree on the recognition composition
> Two things start a real node — `docker compose up` (compose.yaml) and `make dev`
> (the Makefile) — and each picks its models out of the variables below. They must
> choose the *same* ones: when the cascade landed in compose alone, `make dev` went
> on serving `BR_MODEL` by itself and every live phone test silently rehearsed a
> weaker reader than the demo. A new deploy-time variable therefore belongs in
> **both** files, or in `tests/test_deploy_parity.py`'s recorded list of
> deliberately environment-specific ones. `aspects.yaml` carries the pair as the
> `deploy` aspect so `/drift` sees them together.
>
> `make dev` prints the composition it resolved on startup (`recognition BR_MODEL +
> BR_MODEL_B — the same cascade compose.yaml serves`). It withholds `BR_MODEL_B`
> when that file is absent, or when `BR_MODEL_TUNED` asks for the other
> composition — both of which the node treats as hard startup errors.

**Inference node**

| Variable | Effect |
|---|---|
| `BR_MODEL` | ONNX crop classifier → `TrainedEstimator`. The zero-shot default. Reads its crop geometry from `<model>.onnx.meta.json`. |
| `BR_MODEL_B` | A second crop classifier on a different crop geometry → `CascadeEstimator`. Squares `BR_MODEL` is unsure about are re-read by this one and the two averaged. **Both members must ship a geometry sidecar** — a missing one is a startup failure, not a warning. `GET /healthz` reports how often it fires. |
| `BR_CASCADE_TAU` | The cascade's confidence gate (default `0.95`): squares whose max probability falls below it go to `BR_MODEL_B`. Lower = cheaper and closer to `BR_MODEL` alone. |
| `BR_MODEL_TUNED` | A calibration fine-tune of `BR_MODEL`, from `tools/train/calibrate_finetune.py` → `AveragedEstimator`: both models read every square off **one** crop pass (they share a geometry) and the distributions are averaged 50/50. Measured 0.969 on the board it was calibrated for, against `BR_MODEL` alone at 0.938. Two hard startup errors, both deliberate: **without `BR_MODEL`** (the tuned model alone hallucinates pieces onto empty squares — it is the base's vote that suppresses them), and **together with `BR_MODEL_B`** (an unmeasured stack; pick one). `GET /healthz` names both members. |
| `BR_SESSION_BANK` | Path for the per-set exemplar bank. **Opt-in: unset means nothing is ever written.** With `BR_MODEL` set, the two are ensembled. |
| `BR_CENTROIDS` | An explicit curated calibration file. Outranks `BR_MODEL`. |
| `BR_OCCUPANCY_BANK` | A style-agnostic occupancy bank; enables the occupancy-only answer and start-position auto-calibration before any piece-type calibration exists. |
| `BR_SAVE_UPLOADS` | Directory for retained phone photos + sidecars (placement, `autorotated`). This is what turns field shots into eval data. |
| `BR_RECORD_DIR` | Directory for recorded ingress sessions. `serve` sets it to `recordings/` by default, so **every served run is replayable**; `serve --no-record` turns it off. |
| `BR_RECORD_MAX_MB` | Disk-fill stop for the recorder (default: no cap). The session truncates rather than rotating. |
| `BR_DINOV2_ONNX` | Backbone path (default `models/dinov2_small.onnx`). |
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

`--recording` re-sends the stored bytes verbatim, so the replayed run sees the same
pixels the original did. Prefer `--throttle`: the gateway's latest-wins pre-filter
drops frames as a function of *arrival rate*, so a flat-out replay makes it drop
more than the recorded run did. Anything that would reprocess the stream
(`--adaptive`, `--rectify`, `--corners`, `--forward-original`, `--loop`) is refused
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

**Viewer node**

| Variable | Effect |
|---|---|
| `BR_INFERENCE_WS` | Where the *viewer* opens the position stream it relays. |
| `BR_INFERENCE_HTTP` | Where the *viewer* forwards `/api/<path>` requests. |

> [!note] Neither reaches the browser
> The pages are same-origin and relative throughout, so these address the viewer's own
> hop to the node. Point them at the node wherever it actually is — `127.0.0.1` for
> `make dev`, the service name under compose. To reach the stack from off the LAN,
> forward the viewer's port rather than reconfiguring these (`.claude/skills/tunnel`).

> [!warning] `models/` and `data/` are git-ignored
> Not an oversight — the trained model derives from ChessReD (CC BY-NC-SA) and murky-license
> geometry, so it is **personal use only, not redistributable**. A fresh clone has no model;
> the server degrades to the stub and still composes up. See [[training-and-data#Licensing ledger]].

## Stopping it, and the GPU it leaves behind

`make stop` signals both servers, waits, escalates to SIGKILL, and prints the pids
it actually killed — it exits non-zero rather than claiming success it cannot
verify (`tools/stop.sh`). That matters more than it sounds:

> [!warning] A released port does not mean the process is gone
> An inference server can wedge in uvicorn's shutdown behind an in-flight ONNX
> call: it stops listening — so the port frees and nothing looks wrong — while
> staying resident with its **CUDA context and hundreds of MB of VRAM**. One sat
> that way for two hours on 2026-08-13; the next `make dev` inherited a card too
> full for the cascade's second session and **aborted mid-stream** (status 134),
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
catch. The warps are small enough that the CPU path costs nothing worth that
risk. This is **not** mirrored in `compose.yaml`, which declares no GPU device and
runs the same cascade on CPU — the two entry points share an environment, not a
hardware envelope.

## Verifying a restart

> [!danger] `healthz` proves nothing about recognition
> A server with a clobbered calibration bank is perfectly healthy and serves garbage. Two
> separate incidents were missed exactly this way.

**Verify with a known photo.** Post a retained upload whose expected score you know and
check it still reads at that accuracy:

```bash
curl -s -X POST --data-binary @data/uploads/<known>.jpg \
  "http://localhost:8000/estimate?corners=<x0,y0,...,x3,y3>" | jq .placement
```

Reference points on record: `img_0566` → 64/64, `up_20260718_0000` → 64/64,
`up_20260718_0001` → 59/64 (the steep dim board).

**Then check the GPU:**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

> [!bug] The server squats GPU memory even when you think it is CPU-only
> The DINOv2 embedder's ONNX session takes `CUDAExecutionProvider` successfully even when the
> trained model's session falls back to CPU — so "the server is CPU-only" was silently false
> after the ensemble deploy, holding 2.1 GB on a 4 GB card and poised to OOM the next training
> run. During training campaigns, serve with `CUDA_VISIBLE_DEVICES=""`; CPU reads are ~0.6 s
> per board, indistinguishable in the phone flow.

The same rule extends to the test suite: when the GPU is busy, run
`CUDA_VISIBLE_DEVICES="" uv run pytest`, or the DINOv2 session fails with a BFCArena OOM that
looks exactly like a code regression.

## Recovering a poisoned bank

Symptoms: reads that were fine degrade, often with a systematic signature — White/Black
flipped on the back ranks (a rotation-poisoned calibrate), or plausible-looking pieces in the
wrong classes (a mid-game board banked as "start"). See
[[calibration-and-the-bank#The four ways this got poisoned]].

```bash
uv run python tools/rebuild_session_bank.py
```

It rebuilds the bank from the base bank plus every retained calibrate upload, applying the
same orientation probe and piece-square agreement gate the server now enforces — so it
*reports* which calibrates it rejected rather than silently dropping them. In both real
incidents it flagged exactly the one poisoned photo and salvaged the rest.

Then re-verify with a known photo, as above.

## Manual calibration

Rarely needed — auto-calibration covers the normal case — but available when you want to
calibrate from a specific labeled position:

```bash
uv run board-reader-inference calibrate \
  --photo photo.jpg --corners x0,y0,x1,y1,x2,y2,x3,y3 \
  --backbone dinov2 --placement "<FEN field 1>" -o models/exemplars_session.npz
```

Corners are the playing-surface corners in `a8,h8,h1,a1` order. Defaults match the deployed
geometry (`--headroom 1.6 --side-pad 0.15`).

The HTTP equivalent is `POST /calibrate`, which additionally applies the orientation probe and
the agreement gate — prefer it when calibrating from the phone.

## Related

- [[architecture]] — what each process is
- [[calibration-and-the-bank]] — what the bank is and how it breaks
- [[operating-envelope]] — whether a bad read is a bug or a camera angle
- [[TESTING]] — the full manual test guide
