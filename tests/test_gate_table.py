"""``tools/gate_table.py``: the seven-take replay gate as one base→candidate table."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "gate_table.py"


def _load():
    spec = importlib.util.spec_from_file_location("gate_table", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


gate = _load()


def _report(path: Path, rows: dict[str, tuple[int, int, int, int]]) -> None:
    """``rows``: stream -> (correct, squares, perfect, events)."""
    path.mkdir(parents=True)
    streams = [
        {"stream": s, "correct": c, "squares": q, "perfect": p, "events": e}
        for s, (c, q, p, e) in rows.items()
    ]
    (path / "report.json").write_text(json.dumps({"streams": streams}))


TAKES = (gate.GateTake("t1", restamp_stability=True), gate.GateTake("t2"))


@pytest.fixture
def recordings(tmp_path):
    # A base replayed before 2026-11-24 also scored a prior stream; it is ignored.
    same = {"raw": (60, 64, 0, 1), "prior": (64, 64, 1, 1), "settled": (64, 64, 1, 1)}
    _report(tmp_path / "t1/replays/base", same)
    _report(tmp_path / "t1/replays/cand", {"raw": (60, 64, 0, 1), "settled": (64, 64, 1, 1)})
    _report(tmp_path / "t2/replays/base", {"raw": (120, 128, 0, 2), "settled": (128, 128, 1, 2)})
    _report(tmp_path / "t2/replays/cand", {"raw": (128, 128, 2, 2), "settled": (128, 128, 2, 2)})
    return tmp_path


def test_table_shows_changes_as_arrows_and_pools_the_total(recordings):
    out = gate.table("base", "cand", recordings=recordings, takes=TAKES)
    lines = out.splitlines()
    assert lines[0] == "base → cand"
    assert lines[2] == "| take | raw per-sq | raw perfect | settled perfect |"
    # Unchanged cells print once; changed ones as base→candidate.
    assert lines[4] == "| t1 | .9375 | 0/1 | 1/1 |"
    assert lines[5] == "| t2 | .9375→1.0000 | 0/2→2/2 | 1/2→2/2 |"
    # Per-square is pooled over squares, not averaged over takes.
    assert lines[6] == "| **total** | .9375→.9792 | 0/3→2/3 | 2/3→3/3 |"


def test_a_missing_replay_is_named(recordings):
    with pytest.raises(FileNotFoundError, match=r"t1/replays/other/report\.json"):
        gate.table("base", "other", recordings=recordings, takes=TAKES)


def test_the_gate_is_the_seven_documented_takes_and_the_pre_fix_ones_restamp(capsys):
    """08-13 and the three held-out takes predate the motion-meter fix."""
    assert gate.main(["--replay-args"]) == 0
    # serve's default
    served = "--policy burst-end:keepalive_s=2,burst_end_reads=3,spacing_s=0.3,exit_hold_s=0.3"
    assert capsys.readouterr().out.splitlines() == [
        f"20260813-193308 {served} --restamp-stability",
        f"20260827-175539 {served}",
        f"20260828-111421 {served}",
        f"20260827-181802 {served}",
        f"20260827-163352 {served} --restamp-stability",
        f"20260827-170333 {served} --restamp-stability",
        f"20260827-172736 {served} --restamp-stability",
    ]
