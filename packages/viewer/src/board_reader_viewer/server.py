"""board-reader-viewer — serves the live-position web UI (DESIGN.md §6 viewer).

A deliberately thin static server. It renders no board logic server-side; it ships
``index.html`` to the browser, which then connects *directly* to the inference
node's ``/viewer`` WebSocket for position events. The inference WS URL is injected
at request time from ``BR_INFERENCE_WS`` so the same image works in compose and on
a LAN. This package depends on neither core, edge, nor inference — it only serves
assets, reinforcing the layer split.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

_STATIC = Path(__file__).resolve().parent / "static"
_DEFAULT_WS = "ws://localhost:8000/viewer"


def create_app() -> FastAPI:
    app = FastAPI(title="board-reader viewer")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        ws_url = os.environ.get("BR_INFERENCE_WS", _DEFAULT_WS)
        # Inject the inference WS URL the browser should connect to.
        html = html.replace("__BR_INFERENCE_WS__", ws_url)
        return HTMLResponse(html)

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="board-reader-viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    import uvicorn

    uvicorn.run("board_reader_viewer.server:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
