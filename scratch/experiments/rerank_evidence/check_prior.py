"""Fidelity gate: does prior.decode on the RECONSTRUCTED distribution reproduce
the prior board the live gateway actually published?"""
from __future__ import annotations

from board_reader_core.protocol import placement_from_dist
from board_reader_inference.prior import decode
from board_reader_inference.trial import square_errors

import dists
import takes

for name in takes.ALL:
    rep = takes.load(name)
    same_raw = same_prior = 0
    sq_diff = 0
    for read in rep.reads:
        d = dists.dist_from_digest(read.event.evidence)
        same_raw += placement_from_dist(d) == read.raw
        got = decode(d).placement          # live path: quarter_turns=None (ingest.py:96)
        same_prior += got == read.prior
        sq_diff += len(square_errors(read.prior, got))
    n = len(rep.reads)
    print(f"{name:8s} n={n:3d}  raw argmax reproduced {same_raw}/{n}  "
          f"prior reproduced {same_prior}/{n}  ({sq_diff} squares differ in total)")

print()
import numpy as np
for name in takes.ALL:
    rep = takes.load(name)
    tail = []
    corr = wrong_r = wrong_t = 0
    sq = 0
    for read in rep.reads:
        truth = rep.truth.segments[read.segment].placement
        d = dists.dist_from_digest(read.event.evidence)
        got = decode(d).placement
        wrong_r += len(square_errors(truth, got))
        wrong_t += len(square_errors(truth, read.prior))
        sq += 64
        for e in read.event.evidence:
            tail.append(1.0 - e[0][1] - (e[1][1] if len(e) > 1 else 0.0))
    tail = np.array(tail)
    print(f"{name:8s} prior wrong: true {wrong_t:3d}/{sq}  reconstructed {wrong_r:3d}/{sq}"
          f"   tail mass median {np.median(tail):.4f} p90 {np.quantile(tail,0.9):.4f}")
