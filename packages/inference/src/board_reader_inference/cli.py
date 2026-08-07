"""``board-reader-inference`` CLI — run the gateway and calibrate centroids.

    board-reader-inference serve --host 0.0.0.0 --port 8000
    board-reader-inference serve --centroids centroids.npz      # use the M1 estimator
    board-reader-inference serve --record-dir runs/             # ingress recorded here
    board-reader-inference calibrate --image rect.png --placement <FEN1> -o centroids.npz
    board-reader-inference positions --changes-only              # tail the position stream
    board-reader-inference positions --out run.ndjson --quiet    # capture it to a file
    board-reader-inference eval                                  # score the deployed stack

Imports only :mod:`board_reader_core` and this package; it never imports the edge.
Calibration takes an **already-rectified** board crop (run ``board-reader-edge
rectify`` first) — keeping board geometry on the edge and recognition here.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_RECORD_DIR = "recordings"


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.centroids:
        # Picked up by server._default_estimator() at import time.
        os.environ["BR_CENTROIDS"] = args.centroids

    # Every served run records its ingress by default (WP-H2), so a session can
    # always be replayed afterwards — including one driven by the phone /camera
    # page, whose frames reach the same /ingest socket through the viewer's proxy.
    # The default lives here rather than in ``create_app`` on purpose: a real
    # server always records, a test app never writes unless it asks to.
    if args.record_dir:
        os.environ["BR_RECORD_DIR"] = args.record_dir

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
    from board_reader_inference.squares import (
        BboxGeometry,
        crop_squares,
        crop_squares_photo,
    )

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
            geometry=BboxGeometry(headroom=args.headroom, side_pad=args.side_pad),
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


def cmd_positions(args: argparse.Namespace) -> int:
    from board_reader_inference.positions import run

    try:
        stats = run(
            args.url,
            board_id=args.board,
            limit=args.limit,
            changes_only=args.changes_only,
            out_path=args.out,
            tail=None if args.quiet else sys.stdout,
        )
    except OSError as e:
        # No gateway on the other end is the overwhelmingly common case; a
        # connection refused deserves one line naming the URL, not a traceback.
        print(f"cannot follow {args.url}: {e}", file=sys.stderr)
        return 2
    if args.out:
        # Report what was WRITTEN, and name the gap when --changes-only made one:
        # "wrote N" must be a fact about the file, or `wc -l` disagrees with it.
        held = stats.consumed - stats.reported
        suffix = f" ({held} unchanged, not written)" if held else ""
        print(
            f"wrote {stats.reported} position events to {args.out}{suffix}",
            file=sys.stderr,
        )
    # 1 when the stream produced nothing: an empty stream is not a working one,
    # and a scripted demo should fail rather than report success over silence
    # (the same rule `eval` applies to an empty eval). Judged on events consumed,
    # not written — a board that never moved was still a live stream.
    return 0 if stats.consumed else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from board_reader_inference.evaluate import run

    try:
        report = run(
            args.paths or None,
            autorotate=args.autorotate,
            prior=not args.no_prior,
            temperature=args.temperature,
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
    s.add_argument(
        "--record-dir", dest="record_dir", default=DEFAULT_RECORD_DIR,
        help=f"record every ingested frame under here (default {DEFAULT_RECORD_DIR}/); "
             "replay with `board-reader-edge feed --recording <dir>`",
    )
    s.add_argument(
        "--no-record", dest="record_dir", action="store_const", const=None,
        help="do not record ingested frames (runs are then not reproducible)",
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

    from board_reader_inference.positions import DEFAULT_URL

    o = sub.add_parser(
        "positions",
        help="follow the L3 position stream headlessly (tail it, or capture NDJSON)",
    )
    o.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"the gateway's position-event socket (default {DEFAULT_URL})",
    )
    o.add_argument("--board", help="only this board_id (default: every board)")
    o.add_argument(
        "--out",
        help="append each event as one line of §5 JSON here — the captured stream "
             "itself, so two runs can be diffed",
    )
    o.add_argument(
        "--changes-only", dest="changes_only", action="store_true",
        help="report a board only when its placement differs from its previous "
             "one; an at-rest board republishes at the keepalive rate",
    )
    o.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many events (default: follow until interrupted)",
    )
    o.add_argument(
        "--quiet", action="store_true",
        help="suppress the readable tail on stdout (pair with --out)",
    )
    o.set_defaults(func=cmd_positions)

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
    e.add_argument(
        "--no-prior",
        action="store_true",
        help="isolate the raw estimator by skipping the hard-constraint decode "
        "(<=1 king per colour, no pawn on rank 1/8, <=16 pieces, promotion "
        "budget). The decode is ON by default because it is what /estimate now "
        "serves, and this harness scores what serves",
    )
    e.add_argument(
        "--temperature",
        action="store_true",
        help="also fit and report temperature scaling of the confidences "
        "(leave-one-board-out, so no board is graded by its own labels). It "
        "cannot change a single answer — only how honest the numbers are.",
    )
    e.add_argument("--json", help="also write the full report as JSON here")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
