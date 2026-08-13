"""``tools/stop.sh`` must verify, not assume — executed, not string-asserted.

The bug it was written for: `make stop` pkill'd and printed "stopped"
unconditionally, so a server that ignored SIGTERM looked stopped while it went on
holding a CUDA context and ~340MB of VRAM, and the next `make dev` aborted on a
card it thought was empty.

Every test here drives the script against a **decoy** process with a unique
pattern of its own. Nothing invokes `make stop`, which would signal whatever real
servers the developer is running — including a live phone test.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_STOP = _ROOT / "tools" / "stop.sh"

# A decoy that catches SIGTERM and refuses to die reproduces the exact failure:
# a process that survives a graceful stop. The compliant one is the happy path.
_IGNORES_SIGTERM = 'trap "" TERM; while true; do sleep 0.2; done'
_OBEYS_SIGTERM = "while true; do sleep 0.2; done"


@pytest.fixture
def decoy(tmp_path: Path):
    """Spawn throwaway processes matching a pattern unique to this test run."""
    spawned: list[subprocess.Popen[bytes]] = []
    token = f"br-stop-decoy-{uuid.uuid4().hex[:12]}"

    def spawn(body: str) -> subprocess.Popen[bytes]:
        # The token has to appear in the command line for `pgrep -f` to find it;
        # a script named after the token is the least magical way.
        script = tmp_path / f"{token}.sh"
        script.write_text(f"#!/usr/bin/env bash\n{body}\n")
        script.chmod(0o755)
        proc = subprocess.Popen(
            ["/usr/bin/env", "bash", str(script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        spawned.append(proc)
        return proc

    yield token, spawn

    for proc in spawned:  # nothing survives the test, whatever it asserted
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=5)


def _run_stop(pattern: str, grace: str = "1") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_STOP), pattern],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "STOP_GRACE_SECONDS": grace},
    )


def _assert_exited(proc: subprocess.Popen[bytes]) -> None:
    """Reap the decoy and require that it actually died.

    ``os.kill(pid, 0)`` is the wrong check from the spawning process: until the
    parent reaps it, a killed child is a zombie that still answers signal 0. So
    wait on it — a still-running decoy raises ``TimeoutExpired`` and fails.
    """
    proc.wait(timeout=5)


def test_reports_when_there_was_nothing_to_stop() -> None:
    result = _run_stop("[b]r-stop-decoy-nothing-matches-this")
    assert result.returncode == 0
    assert "nothing to stop" in result.stdout


def test_stops_a_cooperative_process_and_names_it(decoy) -> None:
    token, spawn = decoy
    proc = spawn(_OBEYS_SIGTERM)
    time.sleep(0.3)  # let bash get far enough to be findable

    result = _run_stop(f"[b]{token[1:]}")

    assert result.returncode == 0, result.stderr
    assert str(proc.pid) in result.stdout, (
        f"stop must name what it killed, got {result.stdout!r}"
    )
    _assert_exited(proc)


def test_escalates_to_sigkill_when_sigterm_is_ignored(decoy) -> None:
    """The regression that cost a live test: SIGTERM ignored, "stopped" printed."""
    token, spawn = decoy
    proc = spawn(_IGNORES_SIGTERM)
    time.sleep(0.3)

    result = _run_stop(f"[b]{token[1:]}")

    assert result.returncode == 0, result.stderr
    assert "ignored SIGTERM" in result.stdout, (
        "a process that survived SIGTERM must be reported, not silently "
        f"declared stopped; got {result.stdout!r}"
    )
    _assert_exited(proc)  # escalation must actually have killed it


def test_does_not_signal_itself(decoy) -> None:
    """The bracket idiom must keep the script out of its own pgrep results."""
    token, _ = decoy
    result = _run_stop(f"[b]{token[1:]}")
    assert result.returncode == 0
    assert "nothing to stop" in result.stdout


def test_make_stop_delegates_to_the_verified_script() -> None:
    """`make stop` must not regrow its own unverified pkill."""
    recipe = (_ROOT / "Makefile").read_text(encoding="utf-8")
    stop_target = recipe.split("\nstop:", 1)[1].split("\n\n", 1)[0]
    assert "tools/stop.sh" in stop_target
    assert "pkill" not in stop_target, (
        "pkill alone cannot tell a stopped process from one ignoring SIGTERM"
    )
