"""``board-reader-inference`` CLI — run the gateway.

    board-reader-inference serve --host 0.0.0.0 --port 8000

Imports only :mod:`board_reader_core` and this package; it never imports the edge.
"""

from __future__ import annotations

import argparse
import sys


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "board_reader_inference.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="board-reader-inference", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="run the inference gateway")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
