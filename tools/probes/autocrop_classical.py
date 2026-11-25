"""The classical contender of the auto-frame probe: the x-corner lattice detector
that the edge ran until the dumb-camera change (``tools/video/boarddetect.py``,
first ``packages/core/.../boarddetect.py``; it still serves ``check_angle.py``).

It is imported, not copied, so the probe measures exactly the code that once
ran live. Two ladders:

* ``base``  -- the shipped retry ladder (downscale the long side to 1200, 1000,
  900, 1600, 800 px, then with CLAHE). Built for frames where the board fills
  most of the picture.
* ``wide``  -- the same rungs, then 2400 and 3200 px. On a zoomed-out frame the
  shipped ladder shrinks a small board below the corner finder's reach (a board
  at 15 % of a 5,300 px canvas is 240 px wide at 1,600); the extra rungs keep
  it at full resolution. Cost: only frames where the base ladder fails pay
  for them.

    CUDA_VISIBLE_DEVICES= uv run python tools/probes/autocrop_classical.py OUT_DIR

runs both ladders over every canvas of ``autocrop_run.conditions`` (CPU, 10
processes) and writes OUT_DIR/classical.jsonl, one line per (item, condition,
ladder): the quad (canvas pixels) or null, and the detector's wall time. On a
``prev480`` condition the detector sees the phone's 480 px preview.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video"))
import boarddetect

#: ``wide`` = ``base``, then ``extra``. It is run as base first and the extra
#: rungs only when base finds nothing, so a frame never pays for base twice.
LADDERS: dict[str, tuple[tuple[int, bool], ...]] = {
    "base": boarddetect._LADDER,
    "extra": ((2400, False), (3200, False), (2400, True)),
}


@contextmanager
def _ladder(name: str):
    saved = boarddetect._LADDER
    boarddetect._LADDER = LADDERS[name]
    try:
        yield
    finally:
        boarddetect._LADDER = saved


def detect(image: np.ndarray, ladder: str = "base") -> tuple[np.ndarray | None, float]:
    """The board quad (TL, TR, BR, BL) or None, and the wall time in ms."""
    with _ladder(ladder):
        t = time.perf_counter()
        quad = boarddetect.detect_board_quad(image)
        return quad, (time.perf_counter() - t) * 1000


def _job(args: tuple[int, str]) -> list[dict]:
    from autocrop_run import canvas_for, conditions, load_items, locator_view

    n, cond = args
    cv2.setNumThreads(1)
    item = load_items()[n]
    canvas, _, _ = canvas_for(item, n, conditions()[cond])
    view, k = locator_view(canvas, conditions()[cond])
    base, ms_base = detect(view, "base")
    wide, ms_wide = base, ms_base
    if base is None:
        wide, ms_extra = detect(view, "extra")
        ms_wide += ms_extra
    out = []
    for ladder, quad, ms in (("base", base, ms_base), ("wide", wide, ms_wide)):
        out.append(
            {
                "key": item.key,
                "cond": cond,
                "ladder": ladder,
                "quad": None if quad is None else (quad / np.float32(k)).tolist(),
                "ms": ms,
            }
        )
    return out


def main(out_dir: Path) -> None:
    from autocrop_run import conditions, in_run, load_items

    path = out_dir / "classical.jsonl"
    done = set()
    if path.exists():
        done = {
            (r["key"], r["cond"]) for r in map(json.loads, path.read_text().splitlines())
        }
    items = load_items()
    jobs = [
        (n, c)
        for c in conditions()
        for n, it in enumerate(items)
        if (it.key, c) not in done and in_run(c, it)
    ]
    print(f"{len(jobs)} canvases", flush=True)
    with Pool(10) as pool, path.open("a") as f:
        for k, rows in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
