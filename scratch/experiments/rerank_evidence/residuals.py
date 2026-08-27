"""Step 1 — rebuild the residual picture from the raw evidence digests.

For every scored (non-guard) square in every take: truth class, top-1 and top-2
class+prob. Errors are classified by whether the truth is *in* the digest at all
(the re-ranking ceiling) and by margin.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from board_reader_core.protocol import CLASSES, parse_placement
from board_reader_inference.trial import SQUARE_INDEX, Stream, square_errors

import takes

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Square:
    take: str
    event: int  # index into the take's scored reads
    seq: int
    segment: int
    square: str  # algebraic
    truth: str
    c1: str
    p1: float
    c2: str
    p2: float

    @property
    def margin(self) -> float:
        return self.p1 - self.p2

    @property
    def wrong(self) -> bool:
        return self.c1 != self.truth

    @property
    def fixable(self) -> bool:
        """Truth is the runner-up — a re-rank could reach it."""
        return self.wrong and self.c2 == self.truth


def squares(name: str) -> list[Square]:
    rep = takes.load(name)
    rows: list[Square] = []
    for i, read in enumerate(rep.reads):
        digest = read.event.evidence
        if digest is None:
            raise SystemExit(f"{name}: event {i} has no evidence digest")
        truth_board = rep.truth.segments[read.segment].placement
        # truth per square, via the same parser scoring uses
        t = [c for row in parse_placement(truth_board) for c in row]

        for sq, idx in SQUARE_INDEX.items():
            entry = digest[idx]
            c1, p1 = entry[0]
            c2, p2 = entry[1] if len(entry) > 1 else ("", 0.0)
            rows.append(
                Square(
                    take=name, event=i, seq=read.event.frame_seq, segment=read.segment,
                    square=sq, truth=t[idx],
                    c1=c1, p1=float(p1), c2=c2, p2=float(p2),
                )
            )
    return rows


def all_squares() -> dict[str, list[Square]]:
    return {n: squares(n) for n in takes.ALL}


def main() -> None:
    data = all_squares()
    out = {}
    for name, rows in data.items():
        wrong = [r for r in rows if r.wrong]
        fixable = [r for r in wrong if r.fixable]
        # cross-check the digest argmax against the recorded raw placement
        rep = takes.load(name)
        raw_wrong = sum(
            len(square_errors(rep.truth.segments[r.segment].placement, r.board(Stream.RAW)))
            for r in rep.reads
        )
        conf = Counter(f"{r.truth}->{r.c1}" for r in wrong)
        fixconf = Counter(f"{r.truth}->{r.c1}" for r in fixable)
        out[name] = {
            "events": len({r.event for r in rows}),
            "squares": len(rows),
            "wrong_from_digest": len(wrong),
            "wrong_from_scorer": raw_wrong,
            "truth_in_top2": len(fixable),
            "ceiling_pct": round(100 * len(fixable) / len(wrong), 1) if wrong else None,
            "margin_median_fixable": round(
                sorted(r.margin for r in fixable)[len(fixable) // 2], 4
            ) if fixable else None,
            "confusions": dict(conf.most_common()),
            "confusions_fixable": dict(fixconf.most_common()),
        }
        print(f"\n== {name} == {out[name]['squares']} squares, "
              f"{len(wrong)} wrong (scorer says {raw_wrong}), "
              f"{len(fixable)} with truth as runner-up "
              f"({out[name]['ceiling_pct']}% ceiling)")
        for pair, n in conf.most_common(12):
            print(f"   {pair:12s} n={n:3d}  fixable={fixconf.get(pair,0):3d}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "residuals.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
