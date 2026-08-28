#!/usr/bin/env bash
# setup.sh — build the training venv ON THE POD, then prove it.
#
#   BR_VENV=/workspace/torchvenv BR_REPO=/workspace/board-reader bash setup.sh
#
# Reproduces the recipe in tools/train/README.md ("Setup") with one change: the
# CUDA wheel index is pinned to the POD image's CUDA, not the local box's cu13x.
# The local GTX 1650 on driver 610 resolves a cu13x wheel; a RunPod image ships
# a fixed CUDA (12.8.1 in the default image) and installing the wrong index
# either fails to find a wheel or lands one the driver cannot run.
#
# It ends by running tools/train/_env.sh and exits nonzero if that preflight
# fails. That is the whole point: a run once trained for 20 minutes against an
# interpreter with CUDA torch but no ONNX stack, then died at export with the
# GPU time already spent. Nothing may proceed past a red preflight.
#
# The venv is built on the network volume, so it survives `pod delete` and the
# next pod's setup is a no-op re-check rather than a 10-minute reinstall.
set -euo pipefail

BR_REPO="${BR_REPO:?set BR_REPO to the repo checkout on the volume}"
BR_VENV="${BR_VENV:?set BR_VENV to the venv path on the volume}"

# Mirrors the workspace's 7-day dependency seasoning (root pyproject.toml and
# the README recipe). Bump all three together, never just this one.
EXCLUDE_NEWER="${BR_EXCLUDE_NEWER:-2026-06-09T00:00:00Z}"
PY_VERSION="${BR_PY_VERSION:-3.12}"

log() { printf '[setup] %s\n' "$*" >&2; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

# --- CUDA detection ---------------------------------------------------------
# PyTorch publishes one wheel index per CUDA minor. Pick the newest index that
# the pod's CUDA can run; override with BR_CUDA_TAG if the list goes stale.
detect_cuda_tag() {
  if [ -n "${BR_CUDA_TAG:-}" ]; then printf '%s' "$BR_CUDA_TAG"; return 0; fi

  local ver=""
  if command -v nvcc >/dev/null 2>&1; then
    ver="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
  fi
  if [ -z "$ver" ] && command -v nvidia-smi >/dev/null 2>&1; then
    # nvidia-smi reports the driver's *maximum* supported CUDA, which is an
    # upper bound on what any runtime here can use — good enough to pick an index.
    ver="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
  fi
  [ -n "$ver" ] || die "no nvcc and no nvidia-smi — cannot tell which CUDA this pod has (set BR_CUDA_TAG)"

  python3 - "$ver" <<'PY'
import sys

# PyTorch wheel indexes, newest first (download.pytorch.org/whl/<tag>).
INDEXES = [(13, 0, "cu130"), (12, 9, "cu129"), (12, 8, "cu128"),
           (12, 6, "cu126"), (11, 8, "cu118")]
major, minor = (int(x) for x in sys.argv[1].split(".")[:2])
for imaj, imin, tag in INDEXES:
    if (imaj, imin) <= (major, minor):
        print(tag)
        break
else:
    sys.exit(f"[setup] CUDA {major}.{minor} is older than every published wheel index")
PY
}

# --- uv ---------------------------------------------------------------------
ensure_uv() {
  if command -v uv >/dev/null 2>&1; then return 0; fi
  log "installing uv"
  python3 -m pip install --quiet --disable-pip-version-check uv \
    || die "could not install uv (python deps go through uv only)"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH"
}

# --- build ------------------------------------------------------------------
main() {
  [ -d "$BR_REPO/tools/train" ] || die "no repo checkout at $BR_REPO"

  local tag
  tag="$(detect_cuda_tag)"
  log "cuda wheel index: $tag"

  ensure_uv

  if [ ! -x "$BR_VENV/bin/python" ]; then
    log "creating venv at $BR_VENV (python $PY_VERSION)"
    uv venv "$BR_VENV" --python "$PY_VERSION"
  else
    log "venv already present at $BR_VENV — re-checking the stack"
  fi

  # Same package set as README.md: onnx/onnxscript/onnxruntime are NOT optional
  # (from torch 2.9 torch.onnx.export routes through the dynamo exporter, which
  # imports onnxscript), matplotlib backs plots.py, pytest runs the harness tests.
  log "installing the training stack"
  uv pip install --python "$BR_VENV/bin/python" \
    --exclude-newer "$EXCLUDE_NEWER" \
    --index-strategy unsafe-best-match \
    --extra-index-url "https://download.pytorch.org/whl/$tag" \
    torch torchvision timm opencv-python-headless numpy pyyaml pillow \
    onnx onnxscript onnxruntime matplotlib pytest

  # The recipe's venv lives at <repo>/scratch/torchvenv, which is what _env.sh
  # defaults PY to. Keep that true even though the real venv is on the volume.
  mkdir -p "$BR_REPO/scratch"
  ln -sfn "$BR_VENV" "$BR_REPO/scratch/torchvenv"

  log "preflight"
  PY="$BR_VENV/bin/python" bash "$BR_REPO/tools/train/_env.sh" \
    || die "_env.sh preflight failed — refusing to hand this pod a training run"
  log "ready"
}

main "$@"
