"""Pick the tail treatment for the reconstruction, by prior-reproduction fidelity."""
from __future__ import annotations
import numpy as np
from board_reader_core.protocol import CLASS_INDEX, GRID, NUM_CLASSES
from board_reader_inference.prior import decode
from board_reader_inference.trial import square_errors
import takes

TAIL = NUM_CLASSES - 2

def build(digest, share: float) -> np.ndarray:
    arr = np.empty((GRID*GRID, NUM_CLASSES))
    for i, e in enumerate(digest):
        c1, p1 = e[0]; c2, p2 = e[1] if len(e) > 1 else ("empty", 0.0)
        rest = max(0.0, 1.0 - p1 - p2)
        t = share * rest / TAIL
        top = p1 + p2 + (1.0 - share) * rest
        s = (p1 + p2) or 1.0
        arr[i, :] = t
        arr[i, CLASS_INDEX[c1]] = p1 / s * top
        arr[i, CLASS_INDEX[c2]] = p2 / s * top
    return arr.reshape(GRID, GRID, NUM_CLASSES)

reps = {n: takes.load(n) for n in takes.ALL}
print(f"{'share':>7s} " + "  ".join(f"{n:>18s}" for n in reps) + "   total_sq_diff")
for share in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0):
    cells, tot = [], 0
    for n, rep in reps.items():
        ok = 0; sd = 0
        for read in rep.reads:
            got = decode(build(read.event.evidence, share)).placement
            ok += got == read.prior
            sd += len(square_errors(read.prior, got))
        cells.append(f"{ok:3d}/{len(rep.reads):<3d} sq{sd:4d}   ")
        tot += sd
    print(f"{share:7.2f} " + "  ".join(cells) + f"  {tot}")
