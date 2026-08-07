"""Structured run logging: one JSONL event stream per run.

``events.jsonl`` is the machine-readable record of a run — every step-interval,
epoch, and lifecycle event as one JSON object per line, flushed immediately so a
crashed or killed run keeps everything up to the last event. It is the log an
agent (or a plotting script) reads; humans get ``val.log`` and the PNGs.

Record kinds (all carry ``t`` = unix seconds and ``kind``):

* ``meta``  — once at start: config, device, split sizes, model summary.
* ``step``  — every ``log_every`` steps: loss, lr, imgs/s, GPU mem.
* ``epoch`` — per epoch: train loss, full val stats (incl. confusion), timing.
* ``abort`` / ``end`` — how the run finished.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator


class RunLog:
    """JSONL event writer (line-buffered, one flush per event). A new RunLog
    truncates: one file == one run, so consumers never see two runs interleaved."""

    def __init__(self, path: Path) -> None:
        self._f = open(path, "w", buffering=1)

    def log(self, kind: str, **fields: Any) -> None:
        self._f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}) + "\n")

    def close(self) -> None:
        self._f.close()


def read_events(path: Path, kind: str | None = None) -> Iterator[dict]:
    """Yield events from an ``events.jsonl``, optionally filtered by kind.
    Tolerates a torn final line (the run may still be writing)."""
    with open(path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None or ev.get("kind") == kind:
                yield ev
