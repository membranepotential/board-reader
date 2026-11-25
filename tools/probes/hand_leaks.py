"""List the shipped stack's settled and raw errors per gate take, split into
settle lag (settled board == the previous truth segment) and everything else.

    uv run python tools/probes/hand_leaks.py [replay-name]
"""

from __future__ import annotations

import sys
from pathlib import Path

from board_reader_core.recording import Recording

from board_reader_inference.measurement.evaluate.errors import square_errors
from board_reader_inference.measurement.trial.events import read_events
from board_reader_inference.measurement.trial.truth import load_truth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate_table import GATE_TAKES

ROOT = Path(__file__).resolve().parents[2]


def main(name: str) -> None:
    for take in GATE_TAKES:
        rec = ROOT / "recordings" / take.take
        truth = load_truth(rec / "truth.json")
        events = read_events(Recording(rec / "replays" / name), board_id=truth.board_id)
        lag = other = raw_bad = scored = 0
        lines = []
        for e in events:
            i = truth.index_for(e.frame_seq)
            if i < 0 or truth.in_guard(e.frame_seq):
                continue
            scored += 1
            want = truth.segments[i].placement
            raw = square_errors(want, e.read)
            if raw:
                raw_bad += 1
                lines.append(
                    f"    raw  seq {e.frame_seq}: {[(m.square, m.truth, m.predicted) for m in raw]}"
                )
            settled = e.settled or e.read
            miss = square_errors(want, settled)
            if not miss:
                continue
            prev = truth.segments[i - 1].placement if i > 0 else None
            if settled == prev:
                lag += 1
            else:
                other += 1
                lines.append(
                    f"    SETT seq {e.frame_seq}: {[(m.square, m.truth, m.predicted) for m in miss]}"
                )
        print(
            f"{take.take}: scored {scored}, raw wrong {raw_bad}, settled wrong lag {lag} other {other}"
        )
        print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "guard-only")
