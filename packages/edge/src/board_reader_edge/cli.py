"""``board-reader-edge`` CLI — drive frames to a running inference gateway.

Imports only :mod:`board_reader_core` and this package; it never imports the
inference node. The edge and inference processes meet only on the wire.

    board-reader-edge feed                       # synthetic start position, 30 frames
    board-reader-edge feed --video clip.mp4       # a recorded OTB clip
    board-reader-edge feed --images frames/       # a directory of stills
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.source import ImageDirSource, RepeatSource, VideoFileSource


def _build_source(args: argparse.Namespace):
    if args.video:
        return VideoFileSource(args.video, board_id=args.board)
    if args.images:
        return ImageDirSource(args.images, board_id=args.board, fps=args.fps)
    # Default: a self-contained synthetic feed (one rendered board, streamed
    # lazily) so `feed` works with zero external data and O(1) memory.
    count = None if args.synthetic <= 0 else args.synthetic
    image = render_board(args.placement)
    return RepeatSource(image, board_id=args.board, count=count, fps=args.fps)


def cmd_feed(args: argparse.Namespace) -> int:
    source = _build_source(args)
    adapter = FileEdgeAdapter(source, throttle=args.throttle, fps=args.fps)
    sent = asyncio.run(stream_to_gateway(adapter, args.url))
    print(f"sent {sent} frames to {args.url}")
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
    f.set_defaults(func=cmd_feed)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
