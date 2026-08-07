# board-reader — one command to run it locally.
#
#   make dev     start inference + both viewers; Ctrl-C stops all three
#   make stop    stop anything left running
#   make test     the suite in parallel, minus the GPU tests
#   make test-gpu the GPU tests alone (needs an idle GPU)
#   make test-all everything, GPU included
#
# The env below is the *deployed* config, not a convenience default: BR_MODEL
# alone serves the trained model (599/640 on the labeled eval), and only with
# BR_SESSION_BANK does it ensemble with the exemplar bank (638/640). The bank is
# part of the product, so it belongs here rather than in a shell snippet you have
# to remember.
#
# The viewer runs TWICE, on purpose — http and https are not interchangeable here:
#
#   * /camera needs a SECURE context (getUserMedia is refused over plain http on a
#     phone) and it is same-origin throughout: every endpoint it touches is relayed
#     by the viewer itself (packages/viewer/.../proxy.py). So it works over https,
#     with the cached self-signed cert `--ssl` mints.
#   * / and /snapshot are NOT same-origin: they talk to the inference node directly
#     (index.html's `ws://…:8000/viewer`, snapshot.html's `http://…:8000/estimate`).
#     Served over https those become blocked mixed content, and the inference node
#     has no TLS of its own to upgrade them to.
#
# One viewer therefore cannot serve all three pages. Two binds of the same process
# can, and they share the one certificate, so the phone still warns only once.
# Collapsing this back to a single https origin means relaying the rest of the
# inference API through the viewer and repointing both pages at it — a work package,
# not a Makefile line.
#
# Teardown kills the three children by PID, never `kill 0`: the process group
# includes this shell, so a group kill re-enters its own TERM trap, which kills
# the group again — bash recurses until it overflows its stack and segfaults.
# The trap disarms itself first for the same reason.

.PHONY: dev stop test test-gpu test-all lint lint-fix

# `wait -n` below is a bashism, and the teardown relies on bash trap semantics.
SHELL := /bin/bash

# Primary LAN address, so /snapshot works from a phone. Falls back to localhost.
HOST := $(shell ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[\d.]+' || echo 127.0.0.1)

BR_MODEL        ?= models/run_e/model.onnx
BR_SESSION_BANK ?= models/exemplars_session.npz
BR_SAVE_UPLOADS ?= data/uploads

INFERENCE_PORT  ?= 8000
VIEWER_PORT     ?= 8080
VIEWER_TLS_PORT ?= 8443

dev:
	@test -f "$(BR_MODEL)" || echo "!! $(BR_MODEL) missing — server will degrade to the stub"
	@test -f "$(BR_SESSION_BANK)" || echo "!! $(BR_SESSION_BANK) missing — trained model only, no ensemble"
	@echo "  camera     https://$(HOST):$(VIEWER_TLS_PORT)/camera   (accept the warning once)"
	@echo "  snapshot   http://$(HOST):$(VIEWER_PORT)/snapshot"
	@echo "  live       http://$(HOST):$(VIEWER_PORT)"
	@echo "  inference  http://$(HOST):$(INFERENCE_PORT)"
	@BR_MODEL=$(BR_MODEL) \
	BR_SESSION_BANK=$(BR_SESSION_BANK) \
	BR_SAVE_UPLOADS=$(BR_SAVE_UPLOADS) \
	  uv run board-reader-inference serve --host 0.0.0.0 --port $(INFERENCE_PORT) & \
	inference=$$!; \
	BR_INFERENCE_HTTP=http://$(HOST):$(INFERENCE_PORT) \
	BR_INFERENCE_WS=ws://$(HOST):$(INFERENCE_PORT)/viewer \
	  uv run board-reader-viewer --host 0.0.0.0 --port $(VIEWER_PORT) & \
	viewer=$$!; \
	BR_INFERENCE_HTTP=http://$(HOST):$(INFERENCE_PORT) \
	BR_INFERENCE_WS=ws://$(HOST):$(INFERENCE_PORT)/viewer \
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

# `-n auto` (pytest-xdist) fans the suite across every core: ~23s -> ~10s here.
# Deliberately *not* in pyproject's addopts — worker startup makes a single
# focused `uv run pytest path::test` slower, and interleaves its output.
#
# `-m "not gpu"` deselects the two tests that build a real ONNX InferenceSession.
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

# The deselected remainder, serially: they contend with each other too.
test-gpu:
	uv run pytest -m gpu -v

test-all:
	uv run pytest -n auto

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
stop:
	@pkill -f "[b]oard-reader-inference serve" 2>/dev/null; \
	 pkill -f "[b]oard-reader-viewer" 2>/dev/null; \
	 echo "stopped"
