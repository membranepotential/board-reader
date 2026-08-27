"""Load the three scored trial takes as (reads, truth) pairs.

Everything scoring-related is imported from ``board_reader_inference.trial`` —
this module only locates the data and reconstructs the *fast* take's
capture_ts filtering (its recording dir holds two colliding sessions; the take
that was scored is the one after ``FAST_CUTOFF``).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from board_reader_core.recording import EVENTS_NAME, INDEX_NAME, MANIFEST_NAME, Recording
from board_reader_inference.trial import TRUTH_NAME, SessionReport, load_truth, score_session

REPO = Path(__file__).resolve().parents[3]
RECORDINGS = REPO / "recordings"
FAST_CUTOFF = 1787846505.0

SCRATCH = Path(
    os.environ.get("RERANK_SCRATCH", "/tmp/claude-1000/rerank_evidence")
)

#: The three takes this study fits on.
TAKES: dict[str, str] = {
    "control": "20260827-163352",
    "hard": "20260827-172736",
    "fast": "20260827-175539",
}

#: Take 5 — scored after the fitting was done, and never fitted on. A true
#: holdout for whatever the three takes above suggest.
HOLDOUT: dict[str, str] = {"take5": "20260827-181802"}
ALL = {**TAKES, **HOLDOUT}


def _filter_ndjson(src: Path, dst: Path, cutoff: float) -> int:
    kept = 0
    with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if float(d.get("capture_ts", 0.0)) > cutoff:
                out.write(json.dumps(d) + "\n")
                kept += 1
    return kept


def fast_take_dir() -> Path:
    """A capture_ts-filtered copy of the fast take, materialised on demand."""
    src = RECORDINGS / TAKES["fast"]
    dst = SCRATCH / "fast-take-only"
    dst.mkdir(parents=True, exist_ok=True)
    _filter_ndjson(src / INDEX_NAME, dst / INDEX_NAME, FAST_CUTOFF)
    _filter_ndjson(src / EVENTS_NAME, dst / EVENTS_NAME, FAST_CUTOFF)
    shutil.copy(src / MANIFEST_NAME, dst / MANIFEST_NAME)
    shutil.copy(src / TRUTH_NAME, dst / TRUTH_NAME)
    return dst


def take_dir(name: str) -> Path:
    if name == "fast":
        return fast_take_dir()
    return RECORDINGS / ALL[name]


def load(name: str) -> SessionReport:
    d = take_dir(name)
    return score_session(Recording(d), load_truth(d / TRUTH_NAME))


if __name__ == "__main__":
    for name in ALL:
        rep = load(name)
        print(f"{name:8s} events={len(rep.reads):3d} ", end="")
        for s in rep.streams:
            print(f"{s.stream.value}={s.accuracy:.4f}/{s.perfect_rate:.4f} ", end="")
        ev = rep.evidence
        print(f" wrong={ev.wrong} ranked={ev.truth_ranked} runner_up={ev.truth_runner_up}")
