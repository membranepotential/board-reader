"""Offline: may an obscured read commit a complete move? For
docs/reports/2026-11-25-partial-commit-under-hand.md.

    uv run python tools/probes/partial_commit_sim.py [UNTRUTHED_TAKE]

Re-runs the settle debounce over the recorded reads (raw board, capture time,
obscured flag) of every take that has ``replays/latency3`` and a ``truth.json``,
no model needed, under three variants:

  base      today's BoardStabilizer (must reproduce the recorded settled boards)
  off       the energy gate off (every ripe square commits)
  balanced  an obscured read commits a ripe piece group only when the group is
            balanced: as many squares lose piece P to "empty" as squares gain P

and prints settled perfect / scored, the settled boards that are neither the truth
nor the previous truth, and the lag from a truth boundary to the first settled
board that equals it. For UNTRUTHED_TAKE (default 20261125-024330, the live trial
that raised the question) it prints both commit timelines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from board_reader_core.control import DEFAULT_RULES
from board_reader_core.protocol import GRID, encode_placement, parse_placement

from board_reader_inference.measurement.trial.truth import load_truth
from board_reader_inference.stream.stabilize import BoardStabilizer, _Candidate, _excess

ROOT = Path(__file__).resolve().parents[2] / "recordings"
EMPTY = parse_placement("8/8/8/8/8/8/8/8")[0][0]
FILES = "abcdefgh"


class Balanced(BoardStabilizer):
    def observe(self, placement, ts, rules=DEFAULT_RULES, *, obscured=False):
        if not obscured or self._settled is None:
            return super().observe(placement, ts, rules, obscured=obscured)
        grid = parse_placement(placement)
        ripe = []
        for r in range(GRID):
            for c in range(GRID):
                read = grid[r][c]
                if read == self._settled[r][c]:
                    self._forget((r, c))
                    continue
                ch = self._candidates.get((r, c))
                if ch is None or ch.cls != read or ts < ch.first_ts:
                    self._forget((r, c))
                    ch = _Candidate(cls=read, first_ts=ts, reads=1)
                    self._candidates[(r, c)] = ch
                else:
                    ch.reads += 1
                if ch.reads >= self._min_reads and ts - ch.first_ts >= self._window:
                    ripe.append((r, c))
        lost: dict[str, list] = {}
        gained: dict[str, list] = {}
        for r, c in ripe:
            if grid[r][c] == EMPTY:
                lost.setdefault(self._settled[r][c], []).append((r, c))
            else:
                gained.setdefault(grid[r][c], []).append((r, c))
        taken = [
            sq
            for piece in lost
            if len(lost[piece]) == len(gained.get(piece, []))
            for sq in lost[piece] + gained[piece]
        ]
        if taken:
            trial = [row[:] for row in self._settled]
            for r, c in taken:
                trial[r][c] = grid[r][c]
            if _excess(trial, rules) > _excess(self._settled, rules):
                taken = []
        if taken:
            self._changed_ts = max(self._candidates[sq].first_ts for sq in taken)
        for r, c in taken:
            self._settled[r][c] = grid[r][c]
            self._forget((r, c))
        return encode_placement(self._settled)


def raw_placement(event) -> str:
    cells = [EMPTY if sq[0][0] == "empty" else sq[0][0] for sq in event["evidence"]]
    return encode_placement([cells[r * 8 : r * 8 + 8] for r in range(8)])


def load(src: Path):
    event_lines = (src / "events.ndjson").read_text().splitlines()
    events = [json.loads(line) for line in event_lines if '"placement"' in line]
    frames = [json.loads(line) for line in (src / "frames.ndjson").read_text().splitlines()]
    # keep the last session only (a seq restart starts a new one)
    restart = max(
        (
            i
            for i in range(1, len(events))
            if events[i]["frame_seq"] <= events[i - 1]["frame_seq"]
        ),
        default=0,
    )
    f_restart = max(
        (i for i in range(1, len(frames)) if frames[i]["seq"] <= frames[i - 1]["seq"]),
        default=0,
    )
    return events[restart:], {f["seq"]: f["capture_ts"] for f in frames[f_restart:]}


def run(events, variant: str):
    stab = Balanced() if variant == "balanced" else BoardStabilizer()
    out = []
    for e in events:
        obscured = e.get("obscured", False) and variant != "off"
        out.append(
            stab.observe(
                raw_placement(e),
                e["capture_ts"],
                e.get("rules", DEFAULT_RULES),
                obscured=obscured,
            )
        )
    return out


def score(events, settled, truth, seq_ts):
    perfect = scored = other = 0
    wrong = []
    for e, s in zip(events, settled):
        i = truth.index_for(e["frame_seq"])
        if i < 0 or truth.in_guard(e["frame_seq"]):
            continue
        scored += 1
        if s == truth.segments[i].placement:
            perfect += 1
        elif i == 0 or s != truth.segments[i - 1].placement:
            other += 1
            wrong.append(e["frame_seq"])
    lags = []
    never = 0
    for b in range(1, len(truth.segments)):
        start = truth.starts[b]
        end = truth.starts[b + 1] if b + 1 < len(truth.segments) else 1 << 30
        hit = next(
            (
                e["capture_ts"]
                for e, s in zip(events, settled)
                if start <= e["frame_seq"] < end and s == truth.segments[b].placement
            ),
            None,
        )
        if hit is None:
            never += 1
        elif start in seq_ts:
            lags.append(hit - seq_ts[start])
    return perfect, scored, other, lags, never, wrong


def main() -> None:
    takes = sorted(
        p.parent.parent.parent.name for p in ROOT.glob("*/replays/latency3/events.ndjson")
    )
    totals = {v: [0, 0, 0, [], 0] for v in ("base", "off", "balanced")}
    print(
        f"{'take':<17} {'variant':<9} perfect/scored  not-lag-wrong  mean lag  median lag  never"
    )
    for take in takes:
        src = ROOT / take / "replays" / "latency3"
        events, seq_ts = load(src)
        truth = load_truth(ROOT / take / "truth.json")
        for variant in totals:
            settled = run(events, variant)
            if variant == "base":
                same = sum(s == e["placement"] for s, e in zip(settled, events))
                if same != len(events):
                    print(
                        f"  !! base reproduces only {same}/{len(events)} recorded boards on {take}"
                    )
            p, n, o, lags, never, wrong = score(events, settled, truth, seq_ts)
            t = totals[variant]
            if take != "20261124-221950":
                t[0] += p
                t[1] += n
                t[2] += o
                t[3] += lags
                t[4] += never
            lags_s = sorted(lags)
            print(
                f"{take:<17} {variant:<9} {p:>4}/{n:<4}        {o:>3}           "
                f"{sum(lags) / len(lags) if lags else float('nan'):5.2f}s    {lags_s[len(lags_s) // 2] if lags_s else float('nan'):5.2f}s    {never}"
                + (f"   wrong at seq {wrong}" if wrong and variant != "base" else "")
            )
    print("\nseven gate takes pooled (221950 apart):")
    for variant, (p, n, o, lags, never) in totals.items():
        lags_s = sorted(lags)
        print(
            f"  {variant:<9} {p}/{n} perfect, {o} not-lag wrong, mean lag {sum(lags) / len(lags):.2f}s, median {lags_s[len(lags_s) // 2]:.2f}s, never {never}"
        )

    take = sys.argv[1] if len(sys.argv) > 1 else "20261125-024330"
    events, _ = load(ROOT / take)
    t0 = events[0]["capture_ts"]
    for variant in ("base", "balanced"):
        settled = run(events, variant)
        if variant == "base":
            same = sum(s == e["placement"] for s, e in zip(settled, events))
            print(f"\n{take}: base reproduces {same}/{len(events)} recorded boards")
        print(f"-- {variant}: commits")
        for prev, now, e in zip(settled, settled[1:], events[1:]):
            a, b = _flat(prev), _flat(now)
            changed = [f"{_square(k)}:{a[k]}>{b[k]}" for k in range(64) if a[k] != b[k]]
            if changed:
                flag = "obscured" if e.get("obscured") else "        "
                print(
                    f"  t={e['capture_ts'] - t0:6.1f}s seq={e['frame_seq']:4d} {flag} {' '.join(changed)}"
                )


def _flat(placement: str) -> list[str]:
    return [cell for row in parse_placement(placement) for cell in row]


def _square(k: int) -> str:
    return FILES[k % 8] + str(8 - k // 8)


if __name__ == "__main__":
    main()
