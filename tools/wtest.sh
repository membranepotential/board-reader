#!/usr/bin/env bash
# Run the suite from a git WORKTREE without building it a venv of its own.
#
# `uv run pytest` in a fresh worktree resolves and builds the whole workspace
# into a worktree-local `.venv` — minutes of wall clock, repeated on every
# invocation, to test code that differs from main by a handful of files.
#
# The main checkout already has that venv, and its workspace packages are
# editable installs pointing at main's `src` trees. Putting this worktree's four
# `src` directories on PYTHONPATH shadows them (PYTHONPATH precedes the .pth
# entries site-packages adds), so main's interpreter runs THIS worktree's code
# with zero build. Verify with:
#
#   tools/wtest.sh --collect-only -q >/dev/null && \
#     PYTHONPATH=... python -c 'import board_reader_core as m; print(m.__file__)'
#
# Usage — anything after the script name goes straight to pytest:
#
#   tools/wtest.sh packages/inference/tests/test_positions.py -q
#   tools/wtest.sh -n auto -m "not gpu"          # the `make test` equivalent
#   tools/wtest.sh packages/edge -q --durations=5
#
# Set BR_MAIN_CHECKOUT to override the detected main checkout.
set -euo pipefail

wt=$(git rev-parse --show-toplevel)
main=${BR_MAIN_CHECKOUT:-$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')}

if [[ ! -x "$main/.venv/bin/pytest" ]]; then
  echo "no venv at $main/.venv — run 'uv sync' in the main checkout first" >&2
  exit 1
fi
if [[ "$main" == "$wt" ]]; then
  echo "note: already in the main checkout; this is just 'pytest'" >&2
fi

export PYTHONPATH="$wt/packages/core/src:$wt/packages/edge/src:$wt/packages/inference/src:$wt/packages/viewer/src${PYTHONPATH:+:$PYTHONPATH}"

# GPU hidden by default. Under `-n auto` this machine fans the suite across a
# dozen workers, and any of them that builds an ONNX session takes the CUDA
# provider — a dozen of those against one 4GB card is the documented contention
# hazard, and here it does not fail cleanly: the workers block in the driver and
# the run hangs indefinitely (observed twice, at ~94% of the suite, while the
# same suite passes in ~23s with CUDA hidden and the e2e tests pass 4/4 alone).
# The CPU provider is correct for everything in the default selection; the tests
# that genuinely need the GPU carry the `gpu` marker and run alone via
# `make test-gpu`. Set BR_TEST_GPU=1 to hand them the card back.
if [[ "${BR_TEST_GPU:-0}" != "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

cd "$wt"
exec "$main/.venv/bin/pytest" "$@"
