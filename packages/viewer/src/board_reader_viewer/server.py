"""board-reader-viewer — serves the live-position web UI (DESIGN.md §6 viewer).

A deliberately thin server. It renders no board logic: it ships ``index.html`` to the
browser, which then connects *directly* to the inference node's ``/viewer`` WebSocket
for position events. The inference WS URL is injected at request time from
``BR_INFERENCE_WS`` so the same image works in compose and on a LAN.

It also serves the two phone pages — ``/snapshot`` (one photo, read once) and
``/camera``, the DESIGN §4.1 "virtual Pi": a browser edge that streams live frames
into the node. The camera page needs one secure same-origin, so unlike the live page
it never talks to the inference node itself — :mod:`board_reader_viewer.proxy` relays
for it, and ``--ssl`` (:mod:`board_reader_viewer.tls`) makes the origin secure enough
for ``getUserMedia``.

This package depends on neither core, edge, nor inference — it serves assets and
forwards bytes, reinforcing the layer split.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from board_reader_viewer import proxy, tls

_STATIC = Path(__file__).resolve().parent / "static"
_DEFAULT_WS = proxy.DEFAULT_WS
_DEFAULT_HTTP = proxy.DEFAULT_HTTP


def create_app() -> FastAPI:
    app = FastAPI(title="board-reader viewer")
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

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

    @app.get("/snapshot", response_class=HTMLResponse)
    async def snapshot() -> HTMLResponse:
        """Phone snapshot page: photo -> corner-tap -> rectify -> POST /estimate."""
        html = (_STATIC / "snapshot.html").read_text(encoding="utf-8")
        http_url = os.environ.get("BR_INFERENCE_HTTP", _DEFAULT_HTTP)
        html = html.replace("__BR_INFERENCE_HTTP__", http_url)
        return HTMLResponse(html)

    @app.get("/camera", response_class=HTMLResponse)
    async def camera() -> HTMLResponse:
        """Phone camera page: the browser edge (DESIGN §4.1 "virtual Pi").

        Served with nothing injected — every endpoint it uses is same-origin and
        relative, which is the whole point: one origin, one certificate to accept.
        """
        return HTMLResponse((_STATIC / "camera.html").read_text(encoding="utf-8"))

    proxy.install(app)
    return app


app = create_app()


def _banner(host: str, port: int, *, cert: tls.Certificate | None) -> list[str]:
    """Where to point the phone, and why it may refuse — the operator's whole briefing."""
    scheme = "https" if cert is not None else "http"
    lines = [f"board-reader viewer on {scheme}://{host}:{port}"]
    if cert is not None:
        made = "generated" if cert.generated else "reusing"
        lines += [
            f"  TLS: {made} self-signed cert {cert.certfile}",
            f"       valid for {', '.join(cert.names)}",
            "       your phone will warn once — accept it, then the camera works",
        ]
    else:
        lines += [
            "  http: phone cameras stay blocked (getUserMedia needs a secure context)",
            f"        start with --ssl, or forward the port: adb reverse tcp:{port} tcp:{port}",
        ]
    lines.append("  camera page:")
    lines += [f"    {url}" for url in tls.camera_urls(host, port, scheme=scheme)]
    if host.startswith("127.") or host == "localhost":
        lines.append(f"  (bound to {host} — pass --host 0.0.0.0 to reach it from a phone)")
    return lines


def _announce(host: str, port: int, *, cert: tls.Certificate | None) -> None:
    # Flushed explicitly: uvicorn logs to stderr, so a piped stdout would otherwise hold
    # this back behind its buffer and the operator would see the server start with no
    # address to type.
    print("\n".join(_banner(host, port, cert=cert)), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="board-reader-viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--ssl",
        action="store_true",
        help="serve https with a cached self-signed certificate — required for the "
             "/camera page on a phone (getUserMedia needs a secure context)",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    import uvicorn

    cert = tls.ensure_certificate() if args.ssl else None
    _announce(args.host, args.port, cert=cert)
    uvicorn.run(
        "board_reader_viewer.server:app",
        host=args.host,
        port=args.port,
        ssl_certfile=str(cert.certfile) if cert else None,
        ssl_keyfile=str(cert.keyfile) if cert else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
