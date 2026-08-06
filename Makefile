# board-reader — one command to run it locally.
#
#   make dev     start inference + viewer; Ctrl-C stops both
#   make stop    stop anything left running
#
# The env below is the *deployed* config, not a convenience default: BR_MODEL
# alone serves the trained model (599/640 on the labeled eval), and only with
# BR_SESSION_BANK does it ensemble with the exemplar bank (638/640). The bank is
# part of the product, so it belongs here rather than in a shell snippet you have
# to remember.

.PHONY: dev stop

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
	@trap 'kill 0' EXIT INT TERM; \
	BR_MODEL=$(BR_MODEL) \
	BR_SESSION_BANK=$(BR_SESSION_BANK) \
	BR_SAVE_UPLOADS=$(BR_SAVE_UPLOADS) \
	  uv run board-reader-inference serve --host 0.0.0.0 --port 8000 & \
	BR_INFERENCE_HTTP=http://$(HOST):8000 \
	BR_INFERENCE_WS=ws://$(HOST):8000/viewer \
	  uv run board-reader-viewer --host 0.0.0.0 --port 8080 & \
	wait

# The bracket keeps the pattern from matching this recipe's own command line.
stop:
	@pkill -f "[b]oard-reader-inference serve" 2>/dev/null; \
	 pkill -f "[b]oard-reader-viewer" 2>/dev/null; \
	 echo "stopped"
