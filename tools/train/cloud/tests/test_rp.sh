#!/usr/bin/env bash
# Tests for tools/train/cloud/rp — run directly, no pytest, no network, no spend:
#
#   bash tools/train/cloud/tests/test_rp.sh
#
# Everything runs against the stubs in ./stub-bin (runpodctl, ssh, rsync,
# secret-tool), which log their invocations instead of doing anything. The
# point of the suite is the ONE property that costs money when it breaks:
# **a pod that gets created always gets deleted** — on success, on error, on
# SIGINT, on SIGTERM — and the script's real exit code survives the teardown.
#
# Deliberately NOT wired into `make test`: it shells out, sleeps, and signals
# process groups, none of which belong in a 20-second parallel suite.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
RP="$ROOT/tools/train/cloud/rp"
STUBS="$HERE/stub-bin"

pass=0
fail=0

ok()   { printf '  ok   %s\n' "$*"; pass=$((pass + 1)); }
bad()  { printf '  FAIL %s\n' "$*"; fail=$((fail + 1)); }
check() { if [ "$1" = "$2" ]; then ok "$3"; else bad "$3 (got '$1', want '$2')"; fi; }
has()  { if grep -qF -- "$2" "$1"; then ok "$3"; else bad "$3 — not in log"; fi; }
hasnt() { if grep -qF -- "$2" "$1"; then bad "$3 — unexpectedly in log"; else ok "$3"; fi; }

# Runs rp with the stubs on PATH. Exports LOG and RC.
run_rp() {
  LOG="$(mktemp)"
  RP_STUB_LOG="$LOG" PATH="$STUBS:$PATH" "$RP" "$@" >/dev/null 2>&1
  RC=$?
}

section() { printf '\n%s\n' "$*"; }

# Start rp in its own session so a signal can be delivered to the whole process
# group, exactly as a terminal Ctrl-C would. `setsid --fork --wait` propagates
# the child's exit status; the child is the session (and group) leader, so its
# pid doubles as the pgid to signal.
#
# The python shim is not decoration: a shell without job control starts every
# background job with SIGINT set to SIG_IGN, and an ignored disposition survives
# exec — so `trap ... INT` inside rp would silently never fire and the test would
# "pass" a Ctrl-C that never happened. Reset SIGINT to SIG_DFL before exec.
start_bg() {
  LOG="$(mktemp)"
  setsid --fork --wait python3 -c \
    'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
    env RP_STUB_LOG="$LOG" RP_STUB_SSH_SLEEP=15 \
    PATH="$STUBS:$PATH" "$RP" sweep configs/smoke.yaml >/dev/null 2>&1 &
  BGWRAP=$!
  BGPGID=""
  for _ in $(seq 1 200); do
    BGPGID="$(ps --ppid "$BGWRAP" -o pid= 2>/dev/null | tr -d ' ' | head -n1)"
    if [ -n "$BGPGID" ] && grep -q "MemTotal" "$LOG" 2>/dev/null; then return 0; fi
    sleep 0.05
  done
  return 0
}

signal_and_wait() {   # signal_and_wait <SIG>
  if [ -n "$BGPGID" ]; then
    kill -"$1" "-$BGPGID" 2>/dev/null || kill -"$1" "$BGPGID" 2>/dev/null || true
  else
    kill -"$1" "$BGWRAP" 2>/dev/null || true
  fi
  wait "$BGWRAP"
  RC=$?
}

# --------------------------------------------------------------------------
section "syntax"
for f in "$ROOT/tools/train/cloud/rp" \
         "$ROOT/tools/train/cloud/setup.sh" \
         "$ROOT/tools/train/cloud/remote_run.sh" \
         "$ROOT/tools/train/cloud/bench_grid.sh"; do
  if bash -n "$f" 2>/dev/null; then ok "bash -n $(basename "$f")"; else bad "bash -n $(basename "$f")"; fi
done

# --------------------------------------------------------------------------
section "dry-run: sweep spends nothing and prints the right create"
run_rp --dry-run sweep configs/smoke.yaml
check "$RC" 0 "dry-run sweep exits 0"
# The stub log only ever records commands that were actually EXECUTED, so a
# create/delete appearing here would mean --dry-run leaked a state change.
hasnt "$LOG" "pod create" "dry-run does not execute pod create"
hasnt "$LOG" "pod delete" "dry-run does not execute pod delete"
has "$LOG" "network-volume list" "dry-run still runs the read-only preflight"

DRYOUT="$(RP_STUB_LOG="$(mktemp)" PATH="$STUBS:$PATH" "$RP" --dry-run sweep configs/smoke.yaml 2>&1)"
for want in "--cloud-type COMMUNITY" "NVIDIA GeForce RTX 4090" "--data-center-ids EUR-IS-1" \
            "--network-volume-id n102g25d6l" "--public-ip" "--wait" "pod delete"; do
  if printf '%s' "$DRYOUT" | grep -qF -- "$want"; then
    ok "dry-run plan contains $want"
  else
    bad "dry-run plan is missing $want"
  fi
done
if printf '%s' "$DRYOUT" | grep -qF -- "pod stop"; then bad "plan uses pod stop"; else ok "never 'pod stop'"; fi

section "dry-run: bench spends nothing and plans the grid"
run_rp --dry-run bench
check "$RC" 0 "dry-run bench exits 0"
hasnt "$LOG" "pod create" "dry-run bench does not execute pod create"
DRYOUT="$(RP_STUB_LOG="$(mktemp)" PATH="$STUBS:$PATH" "$RP" --dry-run bench 2>&1)"
for want in "bench_grid.sh" "pod delete" "--cloud-type COMMUNITY"; do
  if printf '%s' "$DRYOUT" | grep -qF -- "$want"; then
    ok "bench plan contains $want"
  else
    bad "bench plan is missing $want"
  fi
done

section "trap: bench deletes its pod on success"
run_rp bench
check "$RC" 0 "live-ish bench exits 0"
has "$LOG" "pod delete pod-stub-1" "bench pod deleted on success"

section "dry-run: stage and reap"
run_rp --dry-run stage tools/train/configs
check "$RC" 0 "dry-run stage exits 0"
run_rp --dry-run reap
check "$RC" 0 "dry-run reap exits 0"

# --------------------------------------------------------------------------
section "trap: happy path deletes the pod"
run_rp sweep configs/smoke.yaml
check "$RC" 0 "live-ish sweep exits 0"
has "$LOG" "pod create" "pod was created"
has "$LOG" "pod delete pod-stub-1" "pod was deleted on success"

section "trap: a failing remote step still deletes the pod, and keeps its code"
LOG="$(mktemp)"
RP_STUB_LOG="$LOG" RP_STUB_SSH_FAIL="setup.sh" RP_STUB_SSH_RC=42 \
  PATH="$STUBS:$PATH" "$RP" sweep configs/smoke.yaml >/dev/null 2>&1
RC=$?
check "$RC" 42 "exit code of the failing step survives the trap"
has "$LOG" "pod delete pod-stub-1" "pod deleted after an error"

section "trap: a teardown that itself fails does not mask the real exit code"
LOG="$(mktemp)"
OUT="$(RP_STUB_LOG="$LOG" RP_STUB_SSH_FAIL="setup.sh" RP_STUB_SSH_RC=42 RP_STUB_DELETE_FAIL=1 \
  PATH="$STUBS:$PATH" "$RP" sweep configs/smoke.yaml 2>&1)"
RC=$?
check "$RC" 42 "delete failing (already-deleted pod) keeps exit 42"
if printf '%s' "$OUT" | grep -qF "verify with 'rp reap'"; then
  ok "a failed delete is reported loudly"
else
  bad "a failed delete was swallowed"
fi

section "trap: SIGINT during a remote step deletes the pod (exit 130)"
start_bg
signal_and_wait INT
check "$RC" 130 "SIGINT exits 130"
has "$LOG" "pod delete pod-stub-1" "pod deleted on SIGINT"

section "trap: SIGTERM deletes the pod (exit 143)"
start_bg
signal_and_wait TERM
check "$RC" 143 "SIGTERM exits 143"
has "$LOG" "pod delete pod-stub-1" "pod deleted on SIGTERM"

# --------------------------------------------------------------------------
section "trap: a --wait-timeout that keeps the pod is still reaped"
LOG="$(mktemp)"
RP_STUB_LOG="$LOG" RP_STUB_CREATE_TIMEOUT=1 PATH="$STUBS:$PATH" \
  "$RP" sweep configs/smoke.yaml >/dev/null 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then ok "create timeout fails the run"; else bad "create timeout did not fail"; fi
has "$LOG" "pod delete pod-stub-1" "pod id recovered from the timeout error and deleted"

section "trap: orphan matching this run's pod name is swept by name"
LOG="$(mktemp)"
RP_STUB_LOG="$LOG" RP_STUB_ORPHAN=1 PATH="$STUBS:$PATH" \
  "$RP" sweep configs/smoke.yaml >/dev/null 2>&1
has "$LOG" "pod delete pod-orphan-9" "name-matched orphan deleted"

# --------------------------------------------------------------------------
section "RAM floor"
LOG="$(mktemp)"
RP_STUB_LOG="$LOG" RP_STUB_MEM_KB=16000000 PATH="$STUBS:$PATH" \
  "$RP" sweep configs/smoke.yaml >/dev/null 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then ok "a 15 GiB pod is rejected"; else bad "a 15 GiB pod was accepted"; fi
has "$LOG" "pod delete pod-stub-1" "under-RAM pod deleted"

# --------------------------------------------------------------------------
section "GPU fallback is explicit-only"
LOG="$(mktemp)"
OUT="$(RP_STUB_LOG="$LOG" RP_STUB_CREATE_FAIL="NVIDIA GeForce RTX 4090" PATH="$STUBS:$PATH" \
  "$RP" sweep configs/smoke.yaml 2>&1)"
if printf '%s' "$OUT" | grep -qF -- "--fallback-pro6000"; then
  ok "a 4090 stockout tells the operator about the flag"
else
  bad "no mention of --fallback-pro6000 on stockout"
fi
hasnt "$LOG" "RTX PRO 6000" "no silent escalation to the \$1.69/hr GPU"

LOG="$(mktemp)"
RP_STUB_LOG="$LOG" RP_STUB_CREATE_FAIL="NVIDIA GeForce RTX 4090" PATH="$STUBS:$PATH" \
  "$RP" --fallback-pro6000 sweep configs/smoke.yaml >/dev/null 2>&1
has "$LOG" "RTX PRO 6000 Blackwell Server Edition" "the flag does escalate"
has "$LOG" "pod delete" "fallback pod is deleted too"

# --------------------------------------------------------------------------
section "preflight refuses to add a pod to a populated account"
LOG="$(mktemp)"
RP_STUB_LOG="$LOG" RP_STUB_PODS='[{"id":"pod-existing","name":"someone-elses"}]' \
  PATH="$STUBS:$PATH" "$RP" sweep configs/smoke.yaml >/dev/null 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then ok "aborts when pods are already standing"; else bad "created a pod anyway"; fi
hasnt "$LOG" "pod create" "no pod was created"

# --------------------------------------------------------------------------
section "the key never lands in a command line"
LOG="$(mktemp)"
RP_STUB_LOG="$LOG" PATH="$STUBS:$PATH" "$RP" sweep configs/smoke.yaml >/dev/null 2>&1
hasnt "$LOG" "stub-api-key" "no invocation carries the API key as an argument"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
