"""The seven-take replay gate as one table: a baseline replay beside a candidate.

Every model or pipeline change is gated on the same seven recorded takes: the four
of docs/reports/2026-11-24-chessq-v6-replay-gate.md and the three held-out takes
whose truth docs/reports/2026-11-24-heldout-truth-completion.md completed. Each replay leaves a scored
``recordings/<take>/replays/<name>/report.json`` (``replay --score``); this reads
two of them per take and prints, as ``base→candidate``:

    take | raw per-sq | raw perfect | prior perfect | settled perfect

with a total row (per-square pooled over every scored square, perfect boards
summed). ``make gate`` replays the takes under a new name and prints this against
the Makefile's ``GATE_BASE``.

    uv run python tools/gate_table.py sp03-settle05-7 my-candidate
    uv run python tools/gate_table.py --replay-args   # what `make gate` replays
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from board_reader_inference.stream.schedule import DEFAULT_POLICY_SPEC

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateTake:
    """One gate take and how it is replayed."""

    take: str
    #: Recorded with a broken motion meter (before the 2026-08-27 fix), so its
    #: ``stability`` must be re-metered (``replay --restamp-stability``); without
    #: it every schedule collapses to the keepalive.
    restamp_stability: bool = False

    @property
    def replay_args(self) -> str:
        """Replayed with the schedule ``serve`` ships, so the gate measures the
        served stack (until 2026-11-24 it was plain ``burst-end``, one read per
        move, which ``serve`` never ran)."""
        return f"--policy {DEFAULT_POLICY_SPEC}" + (
            " --restamp-stability" if self.restamp_stability else ""
        )


GATE_TAKES = (
    GateTake("20260813-193308", restamp_stability=True),
    GateTake("20260827-175539"),
    GateTake("20260828-111421"),
    GateTake("20260827-181802"),
    # The held-out takes, all three before the meter fix.
    GateTake("20260827-163352", restamp_stability=True),
    GateTake("20260827-170333", restamp_stability=True),
    GateTake("20260827-172736", restamp_stability=True),
)

#: The three boards a published event carries, in report order.
STREAMS = ("raw", "prior", "settled")


@dataclass(frozen=True)
class StreamRow:
    correct: int
    squares: int
    perfect: int
    events: int


@dataclass(frozen=True)
class Scored:
    """One replay's report.json, reduced to what the gate compares."""

    streams: dict[str, StreamRow]

    @classmethod
    def load(cls, path: Path) -> Scored:
        report = json.loads(path.read_text())
        rows = {
            s["stream"]: StreamRow(s["correct"], s["squares"], s["perfect"], s["events"])
            for s in report["streams"]
        }
        missing = [s for s in STREAMS if s not in rows]
        if missing:
            raise ValueError(f"{path}: no {', '.join(missing)} stream")
        return cls(rows)

    @classmethod
    def total(cls, scores: list[Scored]) -> Scored:
        return cls(
            {
                s: StreamRow(
                    correct=sum(x.streams[s].correct for x in scores),
                    squares=sum(x.streams[s].squares for x in scores),
                    perfect=sum(x.streams[s].perfect for x in scores),
                    events=sum(x.streams[s].events for x in scores),
                )
                for s in STREAMS
            }
        )

    def cells(self) -> tuple[str, str, str, str]:
        raw = self.streams["raw"]
        per_sq = f"{raw.correct / raw.squares:.4f}".lstrip("0") if raw.squares else "—"
        return (
            per_sq,
            *(f"{self.streams[s].perfect}/{self.streams[s].events}" for s in STREAMS),
        )


def report_path(recordings: Path, take: str, name: str) -> Path:
    return recordings / take / "replays" / name / "report.json"


def table(base: str, candidate: str, *, recordings: Path, takes: tuple[GateTake, ...]) -> str:
    """The gate table as Markdown; raises ``FileNotFoundError`` naming a missing report."""
    header = ("take", "raw per-sq", "raw perfect", "prior perfect", "settled perfect")
    pairs: list[tuple[str, Scored, Scored]] = []
    for t in takes:
        a, b = (report_path(recordings, t.take, n) for n in (base, candidate))
        for p in (a, b):
            if not p.exists():
                raise FileNotFoundError(f"no scored replay at {p} (replay --score)")
        pairs.append((t.take, Scored.load(a), Scored.load(b)))
    pairs.append(
        (
            "**total**",
            Scored.total([a for _, a, _ in pairs]),
            Scored.total([b for _, _, b in pairs]),
        )
    )
    lines = [
        f"{base} → {candidate}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for take, a, b in pairs:
        cells = [f"{x}→{y}" if x != y else x for x, y in zip(a.cells(), b.cells(), strict=True)]
        lines.append("| " + " | ".join([take, *cells]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", nargs="?", help="baseline replay name, e.g. cq-v6")
    ap.add_argument("candidate", nargs="?", help="candidate replay name")
    ap.add_argument("--recordings", type=Path, default=ROOT / "recordings")
    ap.add_argument(
        "--replay-args",
        action="store_true",
        help="print '<take> <replay flags>' per gate take and exit (what `make gate` runs)",
    )
    args = ap.parse_args(argv)
    if args.replay_args:
        for t in GATE_TAKES:
            print(t.take, t.replay_args)
        return 0
    if not (args.base and args.candidate):
        ap.error("base and candidate replay names are required")
    try:
        print(table(args.base, args.candidate, recordings=args.recordings, takes=GATE_TAKES))
    except (FileNotFoundError, ValueError) as e:
        print(f"gate_table: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
