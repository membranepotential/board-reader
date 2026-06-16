"""``board-reader-edge`` CLI — drive frames to a running inference gateway.

Imports only :mod:`board_reader_core` and this package; it never imports the
inference node. The edge and inference processes meet only on the wire.

    board-reader-edge feed                       # synthetic start position, 30 frames
    board-reader-edge feed --video clip.mp4       # a recorded OTB clip
    board-reader-edge feed --images frames/       # a directory of stills
    board-reader-edge feed --angled --rectify     # angled synthetic board, un-warped
    board-reader-edge rectify board.jpg -o crop.png   # one-shot rectify (debug)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import cv2

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.rectify import Rectifier, find_board_quad, rectify
from board_reader_edge.source import (
    ImageDirSource,
    RepeatSource,
    VideoFileSource,
)


def _build_source(args: argparse.Namespace):
    if args.video:
        return VideoFileSource(args.video, board_id=args.board)
    if args.images:
        return ImageDirSource(args.images, board_id=args.board, fps=args.fps)
    # Default: a self-contained synthetic feed (one rendered board, streamed
    # lazily) so `feed` works with zero external data and O(1) memory.
    count = None if args.synthetic <= 0 else args.synthetic
    image = render_board(args.placement)
    if args.angled:
        # Simulate the angled broadcast view so `--rectify` has something to un-warp.
        image, _ = simulate_angled_view(image)
    return RepeatSource(image, board_id=args.board, count=count, fps=args.fps)


def cmd_feed(args: argparse.Namespace) -> int:
    source = _build_source(args)
    rectifier = Rectifier() if args.rectify else None
    adapter = FileEdgeAdapter(
        source, rectifier=rectifier, throttle=args.throttle, fps=args.fps
    )
    sent = asyncio.run(stream_to_gateway(adapter, args.url))
    print(f"sent {sent} frames to {args.url}")
    return 0


def _parse_corners(spec: str):
    nums = [float(x) for x in spec.replace(";", ",").split(",") if x.strip()]
    if len(nums) != 8:
        raise SystemExit("--corners needs 8 numbers: x0,y0,x1,y1,x2,y2,x3,y3")
    import numpy as np

    return np.array(nums, dtype="float32").reshape(4, 2)


def cmd_rectify(args: argparse.Namespace) -> int:
    """One-shot: detect (or use given corners), rectify, and write the crop.

    The observable manual test for WP-A — open the output PNG and see a canonical
    top-down board recovered from an angled photo.
    """
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")
    if args.corners:
        corners = _parse_corners(args.corners)
    else:
        corners = find_board_quad(image, min_area_frac=args.min_area_frac)
        if corners is None:
            raise SystemExit(
                "no board detected; pass --corners x0,y0,... (operator corner-tap)"
            )
    crop = rectify(image, corners, out_size=args.out_size)
    if not cv2.imwrite(args.out, crop):
        raise SystemExit(f"could not write output: {args.out}")
    print(f"rectified {args.image} -> {args.out} ({args.out_size}x{args.out_size})")
    print("corners (TL,TR,BR,BL):", corners.tolist())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="board-reader-edge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("feed", help="stream frames from a source to the gateway")
    f.add_argument("--url", default="ws://127.0.0.1:8000/ingest")
    f.add_argument("--board", default="A1")
    f.add_argument("--video", help="path to a video file")
    f.add_argument("--images", help="path to a directory of images")
    f.add_argument("--placement", default=START_PLACEMENT, help="synthetic placement")
    f.add_argument(
        "--synthetic", type=int, default=30,
        help="# synthetic frames (0 = stream forever)",
    )
    f.add_argument("--fps", type=float, default=2.0)
    f.add_argument("--throttle", action="store_true", help="pace to --fps in real time")
    f.add_argument(
        "--angled", action="store_true",
        help="warp the synthetic board to a simulated angled view (demo for --rectify)",
    )
    f.add_argument(
        "--rectify", action="store_true",
        help="detect + rectify each frame to a canonical crop before forwarding (WP-A)",
    )
    f.set_defaults(func=cmd_feed)

    r = sub.add_parser("rectify", help="detect + rectify a single image (debug)")
    r.add_argument("image", help="path to an input image (e.g. an angled board photo)")
    r.add_argument("-o", "--out", default="rectified.png", help="output PNG path")
    r.add_argument("--out-size", dest="out_size", type=int, default=384)
    r.add_argument(
        "--corners",
        help="manual corners 'x0,y0,x1,y1,x2,y2,x3,y3' (a8,h8,h1,a1); skips detection",
    )
    r.add_argument("--min-area-frac", dest="min_area_frac", type=float, default=0.1)
    r.set_defaults(func=cmd_rectify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
