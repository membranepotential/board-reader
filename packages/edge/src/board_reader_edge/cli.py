"""``board-reader-edge`` CLI — drive frames to a running inference gateway.

Imports only :mod:`board_reader_core` and this package; it never imports the
inference node. The edge and inference processes meet only on the wire.

    board-reader-edge feed                       # synthetic start position, 30 frames
    board-reader-edge feed --video clip.mp4       # a recorded OTB clip
    board-reader-edge feed --images frames/       # a directory of stills
    board-reader-edge feed --angled --rectify     # angled synthetic board, un-warped
    board-reader-edge feed --game --adaptive --loop --throttle   # the M2 demo
    board-reader-edge feed --video otb.mp4 --adaptive --loop \
        --corners 120,80,900,80,960,860,60,860       # real footage, corners tapped
    board-reader-edge feed --video otb.mp4 --forward-original \
        --corners 120,80,900,80,960,860,60,860       # ... unwarped, corners on the wire
    board-reader-edge rectify board.jpg -o crop.png   # one-shot rectify (debug)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

from board_reader_core.boarddetect import detect_board_quad
from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view

from board_reader_edge.adapter import FileEdgeAdapter
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig
from board_reader_edge.rectify import Rectifier, rectify
from board_reader_edge.source import (
    FrameSource,
    GameSource,
    ImageDirSource,
    LoopSource,
    RepeatSource,
    VideoFileSource,
)

# A recorded game is a *capture* stream and wants a camera-like rate; the
# one-still feeds only need to tick over.
_GAME_FPS = 12.0
_STILL_FPS = 2.0


def _fps(args: argparse.Namespace) -> float:
    return args.fps if args.fps is not None else (_GAME_FPS if args.game else _STILL_FPS)


def _build_source(args: argparse.Namespace) -> FrameSource:
    source: FrameSource
    if args.video:
        source = VideoFileSource(args.video, board_id=args.board)
    elif args.images:
        source = ImageDirSource(args.images, board_id=args.board, fps=_fps(args))
    elif args.game:
        # A synthetic recorded game: settled positions, hands, transitions — the
        # M2 demo with zero external data (WP-C3, DESIGN §7 M2).
        source = GameSource(
            board_id=args.board, fps=_fps(args), angled=args.angled
        )
    else:
        # A self-contained synthetic feed (one rendered board, streamed lazily) so
        # `feed` works with zero external data and O(1) memory.
        count = None if args.synthetic <= 0 else args.synthetic
        image = render_board(args.placement)
        if args.angled:
            # Simulate the angled view so `--rectify` has something to un-warp.
            image, _ = simulate_angled_view(image)
        source = RepeatSource(
            image, board_id=args.board, count=count, fps=_fps(args)
        )
    return LoopSource(source) if args.loop else source


def _build_forwarder(args: argparse.Namespace) -> AdaptiveForwarder | None:
    """The D4 keepalive/burst gate, or ``None`` to forward every frame."""
    if not args.adaptive:
        return None
    return AdaptiveForwarder(
        ForwarderConfig(keepalive_fps=args.keepalive_fps, burst_fps=args.burst_fps)
    )


def _build_rectifier(args: argparse.Namespace) -> Rectifier | None:
    """Corners given -> manual mode; ``--rectify`` alone -> cadenced auto-detect.

    ``--forward-original`` sends the corners the other way — onto the wire — so
    nothing is warped at the edge and there is no rectifier at all.
    """
    if args.forward_original:
        return None
    if args.corners:
        return Rectifier(manual_corners=_parse_corners(args.corners))
    return Rectifier() if args.rectify else None


def _build_wire_corners(args: argparse.Namespace) -> "np.ndarray | None":
    """The quad to stamp on every wire frame, or ``None`` (rectify-at-edge path)."""
    if not args.forward_original:
        return None
    if args.rectify:
        raise SystemExit(
            "--forward-original and --rectify are exclusive: the first forwards the "
            "original frame with corners attached, the second warps it away"
        )
    if not args.corners:
        raise SystemExit(
            "--forward-original needs --corners x0,y0,...,x3,y3 (a8,h8,h1,a1)"
        )
    return _parse_corners(args.corners)


def cmd_feed(args: argparse.Namespace) -> int:
    wire_corners = _build_wire_corners(args)  # validates the mode before any I/O
    adapter = FileEdgeAdapter(
        _build_source(args),
        forwarder=_build_forwarder(args),
        rectifier=_build_rectifier(args),
        wire_corners=wire_corners,
        throttle=args.throttle,
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
        corners = detect_board_quad(image)
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
    f.add_argument(
        "--game", action="store_true",
        help="synthetic recorded game: positions, hands, transitions (the M2 demo)",
    )
    f.add_argument("--placement", default=START_PLACEMENT, help="synthetic placement")
    f.add_argument(
        "--synthetic", type=int, default=30,
        help="# synthetic frames (0 = stream forever)",
    )
    f.add_argument(
        "--fps", type=float, default=None,
        help=f"capture rate (default {_GAME_FPS:g} with --game, else {_STILL_FPS:g})",
    )
    f.add_argument(
        "--loop", action="store_true",
        help="replay the source forever, re-stamping seq/capture_ts (lazily)",
    )
    f.add_argument(
        "--throttle", action="store_true",
        help="pace output to the stream's own capture time (i.e. real time)",
    )
    f.add_argument(
        "--angled", action="store_true",
        help="warp the synthetic board to a simulated angled view (demo for --rectify)",
    )
    f.add_argument(
        "--rectify", action="store_true",
        help="detect + rectify each frame to a canonical crop before forwarding (WP-A)",
    )
    f.add_argument(
        "--corners",
        help="manual corners 'x0,y0,...,x3,y3' (a8,h8,h1,a1); rectify, never detect",
    )
    f.add_argument(
        "--forward-original", dest="forward_original", action="store_true",
        help="forward the original frames and put --corners on every wire frame "
             "(§4.5 photo path) instead of rectifying at the edge",
    )
    f.add_argument(
        "--adaptive", action="store_true",
        help="forward at a keepalive rate at rest and burst through motion (D4)",
    )
    f.add_argument(
        "--keepalive-fps", dest="keepalive_fps", type=float,
        default=ForwarderConfig.keepalive_fps, help="--adaptive at-rest forward rate",
    )
    f.add_argument(
        "--burst-fps", dest="burst_fps", type=float,
        default=ForwarderConfig.burst_fps, help="--adaptive during-motion forward rate",
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
    r.set_defaults(func=cmd_rectify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
