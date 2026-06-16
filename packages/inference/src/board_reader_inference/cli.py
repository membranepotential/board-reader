"""``board-reader-inference`` CLI — run the gateway and calibrate centroids.

    board-reader-inference serve --host 0.0.0.0 --port 8000
    board-reader-inference serve --centroids centroids.npz      # use the M1 estimator
    board-reader-inference calibrate --image rect.png --placement <FEN1> -o centroids.npz

Imports only :mod:`board_reader_core` and this package; it never imports the edge.
Calibration takes an **already-rectified** board crop (run ``board-reader-edge
rectify`` first) — keeping board geometry on the edge and recognition here.
"""

from __future__ import annotations

import argparse
import os
import sys


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.centroids:
        # Picked up by server._default_estimator() at import time.
        os.environ["BR_CENTROIDS"] = args.centroids

    uvicorn.run(
        "board_reader_inference.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from board_reader_inference.centroid import build_centroids
    from board_reader_inference.embed import make_embedder
    from board_reader_inference.squares import crop_squares

    board = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if board is None:
        raise SystemExit(f"could not read rectified board image: {args.image}")
    crops = crop_squares(board, inset=args.inset, size=args.square_size)
    classifier = build_centroids(
        make_embedder(args.backbone),
        crops,
        args.placement,
        temperature=args.temperature,
        backbone=args.backbone,
    )
    classifier.save(args.out)
    present = int((~np.isnan(classifier.centroids).any(axis=1)).sum())
    print(
        f"calibrated {present}/13 class centroids ({args.backbone}) "
        f"from {args.image} -> {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    from board_reader_core.protocol import START_PLACEMENT

    p = argparse.ArgumentParser(prog="board-reader-inference", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the inference gateway")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument(
        "--centroids", help="path to a centroids .npz (use the M1 CentroidEstimator)"
    )
    s.set_defaults(func=cmd_serve)

    c = sub.add_parser("calibrate", help="build class centroids from a labeled position")
    c.add_argument("--image", required=True, help="a rectified top-down board crop")
    c.add_argument(
        "--placement", default=START_PLACEMENT, help="the board's true FEN field 1"
    )
    c.add_argument(
        "--backbone", default="lite", choices=["lite", "dinov2"],
        help="embedder: 'lite' (synthetic/tests) or 'dinov2' (real photos)",
    )
    c.add_argument("-o", "--out", default="centroids.npz", help="output .npz path")
    c.add_argument("--inset", type=float, default=0.1)
    c.add_argument("--square-size", dest="square_size", type=int, default=None)
    c.add_argument("--temperature", type=float, default=0.1)
    c.set_defaults(func=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
