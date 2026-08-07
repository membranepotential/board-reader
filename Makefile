# board-reader — one command to run it locally.
#
#   make dev     start inference + viewer; Ctrl-C stops both
#   make stop    stop anything left running
#   make test    the whole suite, in parallel
#
# The env below is the *deployed* config, not a convenience default: BR_MODEL
# alone serves the trained model (599/640 on the labeled eval), and only with
# BR_SESSION_BANK does it ensemble with the exemplar bank (638/640). The bank is
# part of the product, so it belongs here rather than in a shell snippet you have
# to remember.
#
# Teardown kills the two children by PID, never `kill 0`: the process group
# includes this shell, so a group kill re-enters its own TERM trap, which kills
# the group again — bash recurses until it overflows its stack and segfaults.
# The trap disarms itself first for the same reason.

.PHONY: dev stop test

# `wait -n` below is a bashism, and the teardown relies on bash trap semantics.
SHELL := /bin/bash

# Primary LAN address, so /snapshot works from a phone. Falls back to localhost.
HOST := $(shell ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[\d.]+' || echo 127.0.0.1)

BR_MODEL        ?= models/run_e/model.onnx
BR_SESSION_BANK ?= models/exemplars_session.npz
BR_SAVE_UPLOADS ?= data/uploads

dev:
	@test -f "$(BR_MODEL)" || echo "!! $(BR_MODEL) missing — server will degrade to the stub"
	@test -f "$(BR_SESSION_BANK)" || echo "!! $(BR_SESSION_BANK) missing — trained model only, no ensemble"
	@echo "  snapshot   http://$(HOST):8080/snapshot"
	@echo "  live       http://$(HOST):8080"
	@echo "  inference  http://$(HOST):8000"
	@BR_MODEL=$(BR_MODEL) \
	BR_SESSION_BANK=$(BR_SESSION_BANK) \
	BR_SAVE_UPLOADS=$(BR_SAVE_UPLOADS) \
	  uv run board-reader-inference serve --host 0.0.0.0 --port 8000 & \
	inference=$$!; \
	BR_INFERENCE_HTTP=http://$(HOST):8000 \
	BR_INFERENCE_WS=ws://$(HOST):8000/viewer \
	  uv run board-reader-viewer --host 0.0.0.0 --port 8080 & \
	viewer=$$!; \
	trap 'kill $$inference $$viewer 2>/dev/null; \
	      wait $$inference $$viewer 2>/dev/null' EXIT; \
	trap 'trap - EXIT INT TERM; \
	      kill $$inference $$viewer 2>/dev/null; \
	      wait $$inference $$viewer 2>/dev/null; \
	      exit 0' INT TERM; \
	wait -n; status=$$?; \
	echo "!! a server exited on its own (status $$status) — stopping the other"; \
	exit $$status

# `-n auto` (pytest-xdist) fans the suite across every core: ~23s -> ~10s here.
# Deliberately *not* in pyproject's addopts — worker startup makes a single
# focused `uv run pytest path::test` slower, and interleaves its output.
test:
	uv run pytest -n auto

# The bracket keeps the pattern from matching this recipe's own command line.
stop:
	@pkill -f "[b]oard-reader-inference serve" 2>/dev/null; \
	 pkill -f "[b]oard-reader-viewer" 2>/dev/null; \
	 echo "stopped"
