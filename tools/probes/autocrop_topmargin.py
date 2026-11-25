"""Auto-frame probe, top margin: does a crop whose top holds a far-rank king read at
least as well as the old box-plus-10 % crop, and does it stop cutting piece tops?

    AUTOCROP_SCRATCH=<scratch> flock /tmp/claude-1000/br-gpu.lock \
        uv run python tools/probes/autocrop_topmargin.py MODEL.onnx OUT_DIR

``MODEL.onnx`` is a served ``-attn`` file (fp32). Per image and condition (the
``as-is`` images and the ``room-*`` canvases of ``autocrop_run.py``) it runs the
production :func:`~board_reader_inference.recognition.locate.autoframe`, then reads
the crop that :func:`~.locate.proposed_crop` gives for the final grid at each
``KING_SQUARES`` in :data:`VARIANTS` (0 is the old rule), and scores it.

A crop **cuts** a piece when the top of a real piece of the truth placement lies
more than :data:`CUT_TOLERANCE` squares above the crop's top edge, and not above
the delivered image's own top edge (that cut is the source's, no crop can undo it).
A piece's top is estimated from the truth outline with the same weak-perspective
geometry as ``locate._king_top``, at the Staunton heights in :data:`HEIGHTS`.
``OUT_DIR/topmargin.jsonl`` holds one line per (image, condition);
``OUT_DIR/pairs/`` shows the old crop (red) and the new one (green) where the tops differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from autocrop_lib import fen_grid
from autocrop_run import canvas_for, conditions, in_run, load_items

from board_reader_core.protocol import CLASSES

from board_reader_inference.recognition import locate
from board_reader_inference.recognition.chessq import ChessQRecognizer

#: ``KING_SQUARES`` values read per image; 0 is the old box-plus-10 % crop.
VARIANTS = (0.0, 1.5, 1.7, 1.9)
#: The value locate.py ships; the pairs compare it with the old crop.
SHIPPED = locate.KING_SQUARES
#: Staunton piece heights in squares: 95 / 85 / 70 / 60 / 55 / 50 mm on 57 mm.
HEIGHTS = {"k": 1.67, "q": 1.49, "b": 1.23, "n": 1.05, "r": 0.96, "p": 0.88}
#: A piece counts as cut when its top is this many squares above the crop.
CUT_TOLERANCE = 0.15
CONDS = ("as-is", "room-35", "room-25", "room-15")
LETTERS = ["." if c == "empty" else c for c in CLASSES]


def piece_tops(
    quad: np.ndarray, placement: str, w: int, h: int
) -> list[tuple[float, float]]:
    """(top y in pixels, one square's face-on side in pixels) of every piece."""
    homography = cv2.getPerspectiveTransform(locate.BOARD_CORNERS, quad.astype(np.float32))
    out = []
    for i, piece in enumerate(fen_grid(placement)):
        if piece == ".":
            continue
        u, v = locate.SQUARE_CENTRES[i]
        eps = 1e-3
        pts = np.float32([[u, v], [u + eps, v], [u, v + eps]])[:, None]
        p = cv2.perspectiveTransform(pts, homography)[:, 0].astype(np.float64)
        jac = np.stack([(p[1] - p[0]) / eps, (p[2] - p[0]) / eps], axis=1)
        left, s, _ = np.linalg.svd(jac)
        cos = np.sqrt(max(0.0, 1 - (s[1] / s[0]) ** 2))
        rise = HEIGHTS[piece.lower()] * s[0] * cos * abs(left[1, 1])
        out.append((float(p[0, 1] - rise), float(s[0])))
    return out


def crop_px(crop, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = round(crop.x * w), round(crop.y * h)
    return x0, y0, round((crop.x + crop.w) * w), round((crop.y + crop.h) * h)


def main(model: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs").mkdir(exist_ok=True)
    rec = ChessQRecognizer(model)
    conds = conditions()
    items = load_items()
    path = out_dir / "topmargin.jsonl"
    done = (
        {(r["key"], r["cond"]) for r in map(json.loads, path.read_text().splitlines())}
        if path.exists()
        else set()
    )
    with path.open("a") as f:
        for cname in CONDS:
            cond = conds[cname]
            for n, item in enumerate(items):
                if (item.key, cname) in done or not in_run(cname, item):
                    continue
                canvas, quad, hand = canvas_for(item, n, cond)
                h, w = canvas.shape[:2]
                line: dict = {"key": item.key, "group": item.group, "cond": cname}
                proposal = locate.autoframe(rec.read, canvas)
                if not isinstance(proposal, locate.Proposal):
                    line["found"] = False
                    f.write(json.dumps(line) + "\n")
                    continue
                line["found"] = True
                tops = piece_tops(quad, item.truth, w, h) if item.truth else []
                line["variants"] = {}
                crops: dict[float, tuple[tuple[int, int, int, int], np.ndarray]] = {}
                for k in VARIANTS:
                    locate.KING_SQUARES = k
                    box = crop_px(locate.proposed_crop(proposal.grid, w, h), w, h)
                    part = canvas[box[1] : box[3], box[0] : box[2]]
                    crops[k] = (box, part)
                    v: dict = {"box": box}
                    if item.truth:
                        pred = [
                            LETTERS[j]
                            for j in rec.read(part).square_dist.reshape(64, -1).argmax(-1)
                        ]
                        v["correct"] = sum(
                            p == t for p, t in zip(pred, fen_grid(item.truth))
                        )
                        v["cuts"] = sum(
                            1
                            for top, side in tops
                            if top < box[1] - CUT_TOLERANCE * side
                            and top >= hand[1] + CUT_TOLERANCE * side
                        )
                    line["variants"][str(k)] = v
                f.write(json.dumps(line) + "\n")
                f.flush()
                b0, b1 = crops[0.0][0], crops[SHIPPED][0]
                if b1[1] < b0[1]:
                    x0, y0 = min(b0[0], b1[0]), min(b0[1], b1[1])
                    view = canvas[y0 : max(b0[3], b1[3]), x0 : max(b0[2], b1[2])].copy()
                    for (a, b, c, d), colour in ((b0, (0, 0, 255)), (b1, (0, 255, 0))):
                        cv2.rectangle(
                            view, (a - x0, b - y0), (c - x0 - 1, d - y0 - 1), colour, 3
                        )
                    k = 900 / max(view.shape[:2])
                    name = f"{cname}_{item.key}".replace(":", "_").replace("#", "_")
                    cv2.imwrite(
                        str(out_dir / "pairs" / f"{name}.jpg"),
                        cv2.resize(view, None, fx=k, fy=k),
                    )
            print(f"  {cname} done", flush=True)
    locate.KING_SQUARES = SHIPPED


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
