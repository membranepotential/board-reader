# board-reader — one command to run it locally.
#
#   make dev     start inference + both viewers; Ctrl-C stops all three
#   make dev-cpu the same stack with the GPU hidden — for when the card is
#                also driving your screens (see CUDA_ENV below)
#   make ui      build the SvelteKit viewer frontend into the Python package
#   make models  fetch the recognizer weights (pinned, SHA-256 verified)
#   make stop    stop anything left running
#   make test     the suite in parallel, minus the GPU tests
#   make test-gpu the GPU tests alone (needs an idle GPU)
#   make test-all everything, GPU included
#   make gate     replay the four gate takes and print them against cq-v6-tuned
#
# `dev` depends on `ui`, unconditionally: the viewer server (server.py) refuses to
# start if packages/viewer/src/board_reader_viewer/web/ is missing, and that
# directory is generated + gitignored, so a fresh checkout has none. Rebuilding it
# on every `make dev` also means an edit under packages/viewer/ui/ is never served
# stale — there is no separate "did the UI change" check to fall out of date.
#
# The env below is the *deployed* config, not a convenience default: BR_CHESSQ
# serves ChessQ Lite V6, the core recognizer (DESIGN D10), the same release
# compose.yaml serves (tests/test_deploy_parity.py) — as fp32 on the GPU here, as
# int8 on the CPU there (see BR_CHESSQ_GPU below). The weights are not in git —
# `make models` fetches and exports them; without them the node serves the stub
# recognizer and says so.
#
# The viewer runs TWICE, but only for convenience now — either bind serves all three
# pages. Every page is same-origin: the viewer relays the whole inference API under
# /api and the sockets under /ws (packages/viewer/.../proxy.py), so nothing the browser
# fetches names a second host, and there is no mixed content to block. (This used to be
# load-bearing: / and /snapshot were handed an absolute `http://<lan-ip>:8000` origin
# injected into their HTML, which no https origin could use.)
#
# What each bind is for:
#   * https :8443 — /camera needs a SECURE context (getUserMedia is refused over plain
#     http on a phone), so the phone uses this one, with the cached self-signed cert
#     `--ssl` mints, and warns once.
#   * http :8080 — the desktop live page, with no cert warning to click through. Also
#     the port to tunnel: `.claude/skills/tunnel` puts a real certificate in front of
#     it, which is why the tunnel needs no TLS from us.
#
# Teardown kills the three children by PID, never `kill 0`: the process group
# includes this shell, so a group kill re-enters its own TERM trap, which kills
# the group again — bash recurses until it overflows its stack and segfaults.
# The trap disarms itself first for the same reason.

.PHONY: dev dev-cpu ui models stop test test-gpu test-all test-ui test-e2e test-front lint lint-fix gate

# `wait -n` below is a bashism, and the teardown relies on bash trap semantics.
SHELL := /bin/bash

# Primary LAN address, so /snapshot works from a phone. Falls back to localhost.
HOST := $(shell ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[\d.]+' || echo 127.0.0.1)

# The recognizer, so the dev path reads boards the same way the canonical
# `docker compose up` demo does. The two entry points drifted once already (the
# cascade landed in compose.yaml alone, 2b64ada); tests/test_deploy_parity.py is
# the guard, and aspects.yaml carries the Makefile as part of `deploy`.
#
# One release, two files, chosen by hardware (docs/reports/2026-11-24-chessq-gpu-formats.md):
#   GPU  our fp32 export — ~205 ms/read on the GTX 1650. The author's int8 runs
#        600+ ms there: its integer matmuls have no CUDA kernels.
#   CPU  the author's int8 — ~660 ms/read against ~990 ms for fp32 on this machine.
# Both read every gate frame identically (cq-v6 vs cq-v6-fp32: 0 of 133 boards differ).
# `make dev` and `make gate` take the GPU file; `make dev-cpu` and compose (which
# declares no GPU device) take the CPU file.
BR_CHESSQ_GPU   := models/chessq/cq-vit-s-644-v6-seed1-onnx-fp32.onnx
BR_CHESSQ_CPU   := models/chessq/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx
BR_CHESSQ       ?= $(BR_CHESSQ_GPU)
BR_SAVE_UPLOADS ?= data/uploads

# OpenCV's OpenCL backend shares the card with onnxruntime, and it is the one
# that turns a tight GPU into a *crash*: when `warpAffine` cannot allocate, it
# throws a cv::Exception through C++ terminate, so the node aborts (status 134)
# mid-stream instead of degrading. Measured 2026-08-13 on the 4GB GTX 1650 —
# the cascade's second session plus a leaked CUDA context from a previous run
# was enough. The image work here is unremarkable on CPU, so the GPU path buys
# nothing worth an uncatchable abort. `disabled` is the strong form: the runtime
# is never loaded, so no VRAM is claimed at all (`=` to re-enable).
#
# NOTE this is deliberately NOT mirrored in compose.yaml: that stack declares no
# GPU device, so it runs the same recognizer on CPU where none of this applies.
# Env parity between the two entry points does not imply the same hardware
# envelope — see tests/test_deploy_parity.py.
OPENCV_OPENCL_RUNTIME ?= disabled

# The inference node's view of the card, empty for `make dev` and emptied *of
# devices* by `make dev-cpu` (target-specific variables apply to a target's
# prerequisites, which is how one assignment reaches the `dev` recipe).
#
# `dev-cpu` exists because on a laptop this GPU is usually also the display
# pipeline, and the display cannot fall back to anything: on 2026-08-13 a
# calibration batch took the card, the X server could no longer allocate the
# reverse-PRIME surface for a 4K external monitor, and the screen went dark
# twice ("Failed to allocate reverse prime surface: out of memory" in
# Xorg.0.log). BR_GPU_MEM_LIMIT_MB caps the arena so that fails catchably
# instead; hiding the card removes the question. It costs ~5x on embedding and
# is exactly what compose.yaml runs, which declares no GPU device at all.
#
# CUDA reads an EMPTY CUDA_VISIBLE_DEVICES as "no devices" — unset would mean
# "all of them", so the assignment below deliberately has nothing after the `=`.
CUDA_ENV :=

INFERENCE_PORT  ?= 8000
VIEWER_PORT     ?= 8080
VIEWER_TLS_PORT ?= 8443

# Prerenders the live / snapshot / camera pages (adapter-static) straight into
# packages/viewer/src/board_reader_viewer/web/, which ships as that package's data
# (packages/viewer/pyproject.toml). `--frozen-lockfile` matches `uv sync --frozen`'s
# discipline on the Python side: this installs exactly pnpm-lock.yaml, never a
# resolution that drifts from what is committed.
ui:
	cd packages/viewer/ui && pnpm install --frozen-lockfile && pnpm build

dev: ui
	@test -f "$(BR_CHESSQ)" \
	  && echo "  recognition ChessQ Lite V6 ($(BR_CHESSQ))" \
	  || echo "!! $(BR_CHESSQ) missing — the node serves the STUB recognizer; run 'make models'"
	@echo "  gpu        $(if $(CUDA_ENV),hidden — the same recognizer on CPU; nothing of ours can reach the display's card,visible — onnxruntime capped at $${BR_GPU_MEM_LIMIT_MB:-768} MiB/session so the display keeps its VRAM)"
	@echo "  camera     https://$(HOST):$(VIEWER_TLS_PORT)/camera   (accept the warning once)"
	@echo "  snapshot   http://$(HOST):$(VIEWER_PORT)/snapshot"
	@echo "  live       http://$(HOST):$(VIEWER_PORT)"
	@echo "  inference  http://$(HOST):$(INFERENCE_PORT)"
	@BR_CHESSQ=$(BR_CHESSQ) \
	BR_SAVE_UPLOADS=$(BR_SAVE_UPLOADS) \
	OPENCV_OPENCL_RUNTIME=$(OPENCV_OPENCL_RUNTIME) \
	$(CUDA_ENV) \
	  uv run board-reader-inference serve --host 0.0.0.0 --port $(INFERENCE_PORT) & \
	inference=$$!; \
	BR_INFERENCE_HTTP=http://127.0.0.1:$(INFERENCE_PORT) \
	BR_INFERENCE_WS=ws://127.0.0.1:$(INFERENCE_PORT)/viewer \
	  uv run board-reader-viewer --host 0.0.0.0 --port $(VIEWER_PORT) & \
	viewer=$$!; \
	BR_INFERENCE_HTTP=http://127.0.0.1:$(INFERENCE_PORT) \
	BR_INFERENCE_WS=ws://127.0.0.1:$(INFERENCE_PORT)/viewer \
	  uv run board-reader-viewer --host 0.0.0.0 --port $(VIEWER_TLS_PORT) --ssl & \
	secure=$$!; \
	trap 'kill $$inference $$viewer $$secure 2>/dev/null; \
	      wait $$inference $$viewer $$secure 2>/dev/null' EXIT; \
	trap 'trap - EXIT INT TERM; \
	      kill $$inference $$viewer $$secure 2>/dev/null; \
	      wait $$inference $$viewer $$secure 2>/dev/null; \
	      exit 0' INT TERM; \
	wait -n; status=$$?; \
	echo "!! a server exited on its own (status $$status) — stopping the others"; \
	exit $$status

# The whole `dev` stack with the card hidden from the inference node. One
# assignment does it: target-specific variables reach a target's prerequisites,
# so this is the value `dev`'s recipe expands. The model file follows the hardware
# (the CPU one, as compose serves); nothing else differs — same release, same
# ports — so it is a straight swap when the machine running this is also the
# machine displaying it.
dev-cpu: CUDA_ENV := CUDA_VISIBLE_DEVICES=
dev-cpu: BR_CHESSQ := $(BR_CHESSQ_CPU)
dev-cpu: dev

# `-n auto` (pytest-xdist) fans the suite across every core: ~23s -> ~10s here.
# Deliberately *not* in pyproject's addopts — worker startup makes a single
# focused `uv run pytest path::test` slower, and interleaves its output.
#
# `-m "not gpu"` deselects the tests that build a real ONNX InferenceSession.
# They pass on an idle machine and fail on a busy one — a training run or a
# second agent session exhausts the GPU arena, session creation dies, and the
# memory pressure has been seen taking unrelated OpenCV allocations down with
# it. Keeping them in the default loop means a red suite that says nothing about
# the code. The count of what was skipped is printed rather than swallowed:
# silently narrowing a suite is how a pipeline ends up reporting success over
# nothing, which this repo has been bitten by three times.
#
# CUDA is hidden here, not merely deselected. `-m "not gpu"` drops the tests that
# build a real ONNX InferenceSession *deliberately*, but other tests still
# construct sessions incidentally, and `-n auto` runs a dozen workers — a dozen
# CUDA contexts against one 4GB card. That does not fail cleanly: the workers
# block inside the driver and the run hangs forever (seen twice at ~94% of the
# suite; the identical suite passes in ~23s with CUDA hidden). The CPU provider
# is correct for everything in this selection, so hiding the card makes the
# default loop both deterministic and faster. `make test-gpu` still gets it.
test:
	CUDA_VISIBLE_DEVICES="" uv run pytest -n auto -m "not gpu"
	@echo "   (GPU tests deselected — run 'make test-gpu' on an idle machine)"
	@echo "   (frontend not covered here — run 'make test-front')"

# The frontend's two test layers. `make test` is the Python suite only, and most of
# what the viewer does now lives in packages/viewer/ui — these are not optional.
#
#   test-ui   Vitest: the tracker maths, the FEN sanitizer, the stores (~1s)
#   test-e2e  Playwright: the three pages driven in a real Chromium against a stubbed
#             inference node — no GPU, no model. It runs `vite build` first and serves
#             the result with `vite preview`, so what is driven is what ships (~15s).
test-ui:
	cd packages/viewer/ui && pnpm install --frozen-lockfile && pnpm test

# `playwright install` is a one-off on a fresh checkout: Playwright pins its own
# Chromium build and refuses to run against another.
test-e2e:
	cd packages/viewer/ui && pnpm install --frozen-lockfile && \
	  pnpm exec playwright install chromium && pnpm test:e2e

test-front: test-ui test-e2e

# The deselected remainder, serially: they contend with each other too.
test-gpu:
	uv run pytest -m gpu -v

test-all:
	uv run pytest -n auto

# The recognizer weights, from the release descriptors committed under
# models/chessq/ (pinned Hugging Face revision, size, SHA-256). Idempotent: a file
# whose SHA-256 already matches is left alone, and a bad download replaces nothing.
#
# Two steps. fetch_models downloads the author's int8 ONNX (the CPU model) and the
# FP16 safetensors; tools/chessq_export then exports those safetensors to the fp32
# ONNX the GPU serves, checks it square by square against the official package's
# own prediction, and writes its descriptor. The export runs in its own uv project
# (PyTorch + chessqueries), so the workspace and the node never depend on torch.
# The first run installs that environment (~1 GB of wheels) and takes a minute.
models:
	uv run python tools/fetch_models.py
	CUDA_VISIBLE_DEVICES= uv run --project tools/chessq_export --frozen \
	  python tools/chessq_export/export.py

# The replay gate (docs/reports/2026-11-24-chessq-v6-replay-gate.md): re-read the
# four gate takes through BR_CHESSQ alone, score each against its truth.json, and
# print the table against GATE_BASE. Serial — each replay wants the GPU. The takes
# and their replay flags live in tools/gate_table.py, so this recipe and the table
# can never disagree about what the gate is. GATE_NAME must be new for a new
# measurement: a replay overwrites its own directory. The base, cq-v6-tuned, is the
# shipped stack (fp32, serve's schedule, the 1.0 s settle): 110/119 raw, 104/119
# settled perfect (docs/reports/2026-11-24-settle-tuning.md).
GATE_NAME ?= gate
GATE_BASE ?= cq-v6-tuned

gate:
	@test -f "$(BR_CHESSQ)" || { echo "!! $(BR_CHESSQ) missing — run 'make models'"; exit 2; }
	@uv run python tools/gate_table.py --replay-args | while read -r take flags; do \
	  echo "  replaying $$take as $(GATE_NAME)"; \
	  BR_CHESSQ=$(BR_CHESSQ) uv run board-reader-inference replay "recordings/$$take" \
	    --name $(GATE_NAME) --score $$flags > /dev/null || exit 1; \
	done
	@uv run python tools/gate_table.py $(GATE_BASE) $(GATE_NAME)

# The lint gate. Ruff is pinned in the dev dependency group, not run through
# `uvx`, so every checkout and every agent lints with the same ruleset — a gate
# whose rules drift by machine is not a gate. Config and the reasoning behind
# each selected/ignored rule live in pyproject.toml's [tool.ruff.lint].
lint:
	uv run ruff check .

# The autofixable subset, applied. Deliberately a separate target: `lint` is safe
# to run anywhere, `lint-fix` rewrites your working tree and wants a clean diff
# to land on. Only the fixes ruff considers safe — `--unsafe-fixes` is never
# wired up here, because a fix that can change behaviour belongs in a review.
lint-fix:
	uv run ruff check --fix .

# The bracket keeps the pattern from matching this recipe's own command line.
# Verified, not assumed — see tools/stop.sh for why "stopped" used to be a lie.
stop:
	@tools/stop.sh "[b]oard-reader-inference serve" "[b]oard-reader-viewer"
