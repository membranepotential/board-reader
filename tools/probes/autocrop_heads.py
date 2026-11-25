"""Which decoder cross-attention head localises the squares best? Reads images
through the all-heads export (``autocrop_model.py SOURCE OUT`` with no picks) and
scores each head's grid fit: R^2 and corner error against the board outline.

    flock /tmp/claude-1000/br-gpu.lock uv run python tools/probes/autocrop_heads.py FULL.onnx

Images: every still and one gate frame per truth segment (outline from
``autocrop_label.py``), as delivered (the hand crop)
and on a grey 16:9 canvas with the board at 35 % and 15 % of its width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from autocrop_lib import (
    GATE_TAKES,
    G,
    Read,
    corner_error_any,
    fit_grid,
    place,
    stills,
    take_frames,
)

from board_reader_inference.recognition.chessq import preprocess_frame
from board_reader_inference.recognition.onnxrt import cuda_providers


def main(model: Path) -> None:
    sess = ort.InferenceSession(str(model), providers=cuda_providers())
    items = stills() + [it for t in GATE_TAKES for it in take_frames(t, per_segment=1)]
    names = [f"L{layer}H{head}" for layer in range(4) for head in range(8)]
    combos = {
        **{n: [n] for n in names},
        "L2H0+L3H0": ["L2H0", "L3H0"],
        "L1H0+L2H0+L3H0": ["L1H0", "L2H0", "L3H0"],
    }
    rows: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for n_item, it in enumerate(items):
        img = cv2.imread(str(it.path))
        for share in (None, 0.35, 0.15):
            if share is None:
                im, quad = img, it.quad
            else:
                im, quad, _ = place(img, it.quad, share, None, seed=n_item)
            logits, *attn = sess.run(
                ["logits", "attn0", "attn1", "attn2", "attn3"],
                {"image": preprocess_frame(im)},
            )
            maps = {
                f"L{layer}H{head}": a[0, head, :, 1:].reshape(64, G, G)
                for layer, a in enumerate(attn)
                for head in range(8)
            }
            read = Read(logits[0], maps, 0.0)
            h, w = im.shape[:2]
            for name, heads in combos.items():
                g = fit_grid(read, heads, w, h)
                err = corner_error_any(g.quad, quad) if g else 999.0
                rows.setdefault(str(share), {}).setdefault(name, []).append(
                    (g.r2 if g else 0.0, err)
                )
        print(f"{n_item + 1}/{len(items)}", end="\r", flush=True)
    print()
    for share, by_head in rows.items():
        ranked = sorted(by_head.items(), key=lambda kv: np.median([e for _, e in kv[1]]))
        print(f"== board share {share} (n={len(next(iter(by_head.values())))})")
        print("head        median R2   median corner err %   p90 err %")
        for name, v in ranked[:8]:
            r2 = np.array([r for r, _ in v])
            err = np.array([e for _, e in v])
            print(
                f"{name:11s} {np.median(r2):9.4f}   {np.median(err):10.2f}"
                f"           {np.percentile(err, 90):8.2f}"
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
