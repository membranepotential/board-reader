"""Which command runs a stream, and the node URL it streams to.

The node runs each stream as a ``board-reader-edge camera`` process (DESIGN D22). It
never imports the edge package, which keeps the two nodes apart (``tests/
test_layer_separation.py``); it starts the edge's console script by name, the same
process-level glue as the CLI and compose. An image without the edge (the inference
Docker image) has no such command, and adding a stream is then refused with
:attr:`NoEdge.reason`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: The command that runs a stream, as a shell-quoted string, when the default
#: (``board-reader-edge`` on ``PATH`` or beside this Python) is not the right one.
EDGE_COMMAND_ENV = "BR_EDGE_COMMAND"
#: The edge's console script.
EDGE_SCRIPT = "board-reader-edge"

#: The node's own WebSocket base, as a stream's process reaches it (``serve`` sets it
#: from ``--host``/``--port``).
NODE_URL_ENV = "BR_NODE_URL"
DEFAULT_NODE_URL = "ws://127.0.0.1:8000"


@dataclass(frozen=True, slots=True)
class EdgeCommand:
    """The argv prefix that runs ``board-reader-edge``."""

    argv: tuple[str, ...]

    def camera(self, url: str, *, node: str, state: Path, start_s: float | None) -> tuple[str, ...]:
        """The whole argv of one stream's ``camera`` process."""
        start = () if not start_s else ("--start", str(start_s))
        return (*self.argv, "camera", url, "--node", node, "--state", str(state), *start)


@dataclass(frozen=True, slots=True)
class NoEdge:
    """No command can run a stream here; ``reason`` says what to do."""

    reason: str


def edge_command(
    environ: Mapping[str, str] = os.environ, python: str = sys.executable
) -> EdgeCommand | NoEdge:
    """``BR_EDGE_COMMAND`` if set, else ``board-reader-edge`` on ``PATH``, else the one
    beside ``python`` (a virtualenv's ``bin/``, when the node runs without it on PATH)."""
    raw = environ.get(EDGE_COMMAND_ENV, "").strip()
    if raw:
        argv = tuple(shlex.split(raw))
        if shutil.which(argv[0], path=environ.get("PATH")) is None:
            return NoEdge(f"{EDGE_COMMAND_ENV}={raw!r}: {argv[0]} is not a command here")
        return EdgeCommand(argv)
    found = shutil.which(EDGE_SCRIPT, path=environ.get("PATH"))
    if found is not None:
        return EdgeCommand((found,))
    beside = Path(python).parent / EDGE_SCRIPT
    if os.access(beside, os.X_OK):
        return EdgeCommand((str(beside),))
    return NoEdge(
        f"this node cannot run streams: no {EDGE_SCRIPT} command on PATH or beside "
        f"{python}. Install the board-reader-edge package, or set {EDGE_COMMAND_ENV}"
    )


def node_url(environ: Mapping[str, str] = os.environ) -> str:
    return environ.get(NODE_URL_ENV) or DEFAULT_NODE_URL
