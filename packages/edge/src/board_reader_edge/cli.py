"""``board-reader-edge`` CLI — drive frames to a running inference gateway.

Imports only :mod:`board_reader_core` and this package; it never imports the
inference node. The edge and inference processes meet only on the wire.

    board-reader-edge feed                       # synthetic start position, 30 frames
    board-reader-edge feed --video clip.mp4       # a recorded OTB clip
    board-reader-edge feed --images frames/       # a directory of stills
    board-reader-edge feed --game --adaptive --loop --throttle   # the M2 demo
    board-reader-edge feed --video otb.mp4 --adaptive --loop \
        --roi 0.2,0.1,0.6,0.8                       # real footage, board's rectangle
    board-reader-edge feed --recording recordings/ --throttle   # re-run a recorded session
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from board_reader_core.control import CropRect
from board_reader_core.protocol import START_PLACEMENT
from board_reader_core.sim import render_board, simulate_angled_view

from board_reader_edge.adapter import (
    DEFAULT_JPEG_QUALITY,
    FileEdgeAdapter,
    RecordingReplay,
)
from board_reader_edge.client import stream_to_gateway
from board_reader_edge.forwarder import AdaptiveForwarder, ForwarderConfig
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
            # A camera looking down at an angle, as a real one does.
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


# Flags that reprocess frames on their way out. A recording already holds the
# edge's *output*, so replaying it through any of these would send something other
# than what was recorded — which is the one thing replay exists not to do.
# (``--jpeg-quality`` defaults to ``None`` rather than 90 precisely so that
# "the operator asked for a re-encode" is visible here as falsy-vs-set.)
_REPROCESSING_FLAGS = {
    "adaptive": "--adaptive",
    "roi": "--roi",
    "loop": "--loop",
    "jpeg_quality": "--jpeg-quality",
}


def _build_replay(args: argparse.Namespace) -> RecordingReplay:
    """The verbatim re-run path (``--recording``), or a clear error."""
    conflicts = [flag for attr, flag in _REPROCESSING_FLAGS.items() if getattr(args, attr)]
    if conflicts:
        raise SystemExit(
            f"--recording replays a recorded session verbatim; {', '.join(conflicts)} "
            "would change what goes on the wire and is therefore refused"
        )
    try:
        return RecordingReplay(args.recording, throttle=args.throttle)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


def cmd_feed(args: argparse.Namespace) -> int:
    if args.recording:
        replay = _build_replay(args)
        sent = asyncio.run(stream_to_gateway(replay, args.url))
        print(f"replayed {sent} frames from {replay.recording.path} to {args.url}")
        return 0
    adapter = FileEdgeAdapter(
        _build_source(args),
        roi=args.roi,
        forwarder=_build_forwarder(args),
        jpeg_quality=(
            DEFAULT_JPEG_QUALITY if args.jpeg_quality is None else args.jpeg_quality
        ),
        throttle=args.throttle,
    )
    sent = asyncio.run(stream_to_gateway(adapter, args.url))
    print(f"sent {sent} frames to {args.url}")
    return 0


def _roi(spec: str) -> CropRect:
    """``--roi x,y,w,h`` -> a :class:`CropRect`, refused at the argparse boundary."""
    try:
        return CropRect.parse(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _jpeg_quality(spec: str) -> int:
    q = int(spec)
    if not 1 <= q <= 100:
        raise argparse.ArgumentTypeError("--jpeg-quality must be between 1 and 100")
    return q


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="board-reader-edge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("feed", help="stream frames from a source to the gateway")
    f.add_argument("--url", default="ws://127.0.0.1:8000/ingest")
    f.add_argument("--board", default="A1")
    f.add_argument("--video", help="path to a video file")
    f.add_argument("--images", help="path to a directory of images")
    f.add_argument(
        "--recording",
        help="re-run a recorded session verbatim (a session dir, or a root dir "
             "whose newest session is taken); pair with --throttle (WP-H2)",
    )
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
        help="warp the synthetic board to a simulated angled camera view",
    )
    f.add_argument(
        "--roi", type=_roi, default=None,
        help="send only this rectangle of each frame: 'x,y,w,h' in fractions of the "
             "frame (top-left corner, width, height), e.g. 0.2,0.1,0.6,0.8. Motion "
             "is measured inside it too. Default: the whole frame",
    )
    f.add_argument(
        "--jpeg-quality", dest="jpeg_quality", type=_jpeg_quality, default=None,
        help="JPEG quality of the forwarded frames, 1-100 (default "
             f"{DEFAULT_JPEG_QUALITY}). Worth turning down on a metered uplink",
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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
