"""``board-reader-inference`` CLI — run the gateway and calibrate centroids.

    board-reader-inference serve --host 0.0.0.0 --port 8000
    board-reader-inference serve --centroids centroids.npz      # use the M1 estimator
    board-reader-inference calibrate --image rect.png --placement <FEN1> -o centroids.npz
    board-reader-inference eval                                  # score the deployed stack

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


def _parse_corners(spec: str):
    import numpy as np

    nums = [float(x) for x in spec.replace(";", ",").split(",") if x.strip()]
    if len(nums) != 8:
        raise SystemExit("--corners needs 8 numbers: x0,y0,...,x3,y3 (a8,h8,h1,a1)")
    return np.array(nums, dtype="float32").reshape(4, 2)


def cmd_calibrate(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from board_reader_inference.centroid import build_centroids
    from board_reader_inference.embed import make_embedder
    from board_reader_inference.squares import crop_squares, crop_squares_photo

    if bool(args.image) == bool(args.photo):
        raise SystemExit("pass exactly one of --image (rectified crop) or --photo")
    if args.photo and not args.corners:
        raise SystemExit("--photo needs --corners x0,y0,...,x3,y3 (a8,h8,h1,a1)")

    if args.photo:
        photo = cv2.imread(args.photo, cv2.IMREAD_COLOR)
        if photo is None:
            raise SystemExit(f"could not read photo: {args.photo}")
        crops = crop_squares_photo(
            photo,
            _parse_corners(args.corners),
            headroom=args.headroom,
            side_pad=args.side_pad,
            size=args.square_size or 224,
        )
        crop_mode, source = "photo", args.photo
    else:
        board = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if board is None:
            raise SystemExit(f"could not read rectified board image: {args.image}")
        crops = crop_squares(board, inset=args.inset, size=args.square_size)
        crop_mode, source = "rectified", args.image

    classifier = build_centroids(
        make_embedder(args.backbone),
        crops,
        args.placement,
        temperature=args.temperature,
        backbone=args.backbone,
        crop_mode=crop_mode,
        augment=args.augment,
        box_headroom=args.headroom,
        box_side_pad=args.side_pad,
    )
    classifier.save(args.out)
    present = int((~np.isnan(classifier.centroids).any(axis=1)).sum())
    print(
        f"calibrated {present}/13 class centroids ({args.backbone}, {crop_mode}"
        f"{', augmented' if args.augment else ''}) from {source} -> {args.out}"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from board_reader_inference.evaluate import run

    try:
        report = run(
            args.paths or None,
            autorotate=args.autorotate,
            json_out=args.json,
        )
    except ValueError as e:
        # Misconfiguration (no photo-capable estimator, unreadable photo) — say so
        # in one line rather than a traceback; the message names the fix.
        print(f"cannot evaluate: {e}", file=sys.stderr)
        return 2
    print(report.render())
    # 1 when there was nothing to score: an empty eval is not a passing eval.
    return 0 if report.boards else 1


def build_parser() -> argparse.ArgumentParser:
    from board_reader_core.protocol import START_PLACEMENT

    from board_reader_inference.evaluate import DEFAULT_PATHS

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
    c.add_argument("--image", help="a rectified top-down board crop")
    c.add_argument(
        "--photo",
        help="an ORIGINAL photo (full-piece boxes; needs --corners) — preferred "
        "for real photos: warped squares decapitate pieces",
    )
    c.add_argument(
        "--corners",
        help="playing-surface corners in --photo pixels: x0,y0,...,x3,y3 (a8,h8,h1,a1)",
    )
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
    c.add_argument(
        "--headroom", type=float, default=1.6,
        help="--photo mode: upward box extension in square-widths (piece height)",
    )
    c.add_argument(
        "--side-pad", dest="side_pad", type=float, default=0.15,
        help="--photo mode: sideways box padding in square-widths",
    )
    c.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True,
        help="photometric augmentation (gamma/white-balance variants) of the "
        "calibration crops; widens lighting robustness",
    )
    c.set_defaults(func=cmd_calibrate)

    e = sub.add_parser(
        "eval",
        help="score the deployed estimator on labeled boards (perfect-board rate)",
    )
    e.add_argument(
        "paths",
        nargs="*",
        help="files/dirs holding labels (*.label.json, ground_truth.json); "
        f"default: {' '.join(DEFAULT_PATHS)}",
    )
    e.add_argument(
        "--autorotate",
        action="store_true",
        help="also let the estimator pick the board's orientation, as the "
        "snapshot page does with auto-detected corners; off by default so the "
        "score isolates recognition from orientation",
    )
    e.add_argument("--json", help="also write the full report as JSON here")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
