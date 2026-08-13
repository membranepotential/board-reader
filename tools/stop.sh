#!/usr/bin/env bash
# Stop the processes matching each `pgrep -f` pattern given, and *verify* it.
#
# This exists because "stopped" was a lie. `make stop` used to pkill and print
# success unconditionally; on 2026-08-13 an inference server ignored SIGTERM
# (uvicorn's shutdown wedged behind an in-flight ONNX call), released its port so
# nothing looked wrong, and sat for two hours holding a CUDA context and ~340MB
# of VRAM. The next `make dev` inherited a card too full for the two-model
# cascade and aborted mid-stream, taking a live phone test with it.
#
# So: signal, wait, escalate, and say what actually died. A stop that cannot
# confirm the process is gone exits non-zero rather than claiming success.
#
# Usage:  tools/stop.sh "[p]attern" ["[p]attern" ...]
# Env:    STOP_GRACE_SECONDS (default 10) — how long SIGTERM gets before SIGKILL.
#
# Pass patterns using the `[f]oo` bracket idiom: the literal text in this
# script's own argv then does not match the regex, so it never signals itself.
set -uo pipefail

grace_seconds=${STOP_GRACE_SECONDS:-10}
self=$$

pids=""
for pattern in "$@"; do
    for pid in $(pgrep -f "$pattern" 2>/dev/null || true); do
        [ "$pid" = "$self" ] && continue
        case " $pids " in *" $pid "*) continue ;; esac
        pids="$pids $pid"
    done
done
pids=$(echo $pids)

if [ -z "$pids" ]; then
    echo "nothing to stop"
    exit 0
fi

alive_of() {
    local out="" p state
    for p in $pids; do
        kill -0 "$p" 2>/dev/null || continue
        # A zombie has already exited — it lingers only until its parent reaps
        # it, and `kill -0` keeps succeeding the whole time. Counting it as alive
        # makes a perfectly successful stop report UNKILLABLE, which is how this
        # script would re-learn the habit of lying that it was written to break.
        state=$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')
        case "$state" in Z*) continue ;; esac
        out="$out $p"
    done
    echo "$out"
}

kill -TERM $pids 2>/dev/null || true

# Poll rather than sleep the full grace period: the common case is a clean
# shutdown in well under a second, and `make stop` sits in front of `make dev`.
deadline_ticks=$(awk "BEGIN { printf \"%d\", $grace_seconds * 2 }")
alive=""
for _ in $(seq "$deadline_ticks"); do
    alive=$(alive_of)
    [ -z "$alive" ] && break
    sleep 0.5
done

if [ -n "$alive" ]; then
    echo "!!$alive ignored SIGTERM for ${grace_seconds}s — SIGKILL (the VRAM-leak path)"
    kill -KILL $alive 2>/dev/null || true
    sleep 1
fi

still=$(alive_of)
if [ -n "$still" ]; then
    echo "!! UNKILLABLE:$still — inspect with 'ps -p $(echo $still | tr ' ' ',')'"
    exit 1
fi

echo "stopped:$pids"
