"""board-reader-viewer — serves the live-position web UI (DESIGN.md §6 viewer).

A deliberately thin server. It renders no board logic: it ships four pages — the live
page (``/``), ``/snapshot`` (one photo, read once), ``/camera``, the DESIGN §4.1
"virtual Pi" browser edge that streams live frames into the node, and ``/control``, the
laptop page that configures that camera — plus the hashed ``/_app/...`` assets those
pages load.

The pages themselves are a prerendered SvelteKit build (``packages/viewer/ui``, built
by ``make ui`` / ``pnpm build``) that lands in ``web/`` next to this file. This module
does not build them and injects nothing into them at serve time — it only serves the
static output. ``web/`` is generated and gitignored, so a fresh checkout will not have
it until the build has run.

None of the pages talk to the inference node directly. Every endpoint they use is
same-origin and relative, and :mod:`board_reader_viewer.proxy` relays it, so the
browser only ever knows one origin and one certificate. ``--ssl``
(:mod:`board_reader_viewer.tls`) makes that origin secure enough for ``getUserMedia``
on a LAN; behind a tunnel the edge's own certificate does it instead
(``.claude/skills/tunnel``). ``BR_INFERENCE_WS`` / ``BR_INFERENCE_HTTP`` configure the
viewer's own hop to the node and are never sent to the browser.

This package depends on neither core, edge, nor inference — it serves assets and
forwards bytes, reinforcing the layer split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from board_reader_viewer import proxy, tls

_WEB = Path(__file__).resolve().parent / "web"


def _require_web(web: Path) -> Path:
    """The prerendered SvelteKit bundle, or a clear instruction instead of a crash.

    Raising ``SystemExit`` (not a bare exception) matters here: it is the one
    exception Python's default handler prints as a bare message rather than a full
    traceback, so an operator who hasn't run the build yet gets one readable line
    telling them what to do — not a stack trace, and never a server that starts and
    then hands the browser a blank or broken page.
    """
    if not (web / "index.html").is_file():
        raise SystemExit(
            f"board_reader_viewer: no UI build at {web}\n"
            "  run `make ui`, or from packages/viewer/ui:\n"
            "    pnpm install --frozen-lockfile && pnpm build"
        )
    return web


def _page(web: Path, name: str) -> HTMLResponse:
    """One prerendered page, or a 503 that says how to build it.

    A missing bundle is an operator mistake, and :func:`main` already refuses to start
    a server without one — so in practice this is unreachable. It exists because
    ``create_app`` must not *require* the bundle: the relay
    (:mod:`board_reader_viewer.proxy`) and the certificate have nothing to do with the
    frontend, and making them unbuildable without a ``pnpm`` run would put a Node
    toolchain between a fresh checkout and testing a Python byte-forwarder.

    503 rather than 404: the page is expected to exist and the deployment is
    incomplete, which is exactly what "service unavailable" means. It carries the same
    instruction the startup error does, so whichever one you hit tells you the same
    thing.
    """
    page = web / name
    if not page.is_file():
        return HTMLResponse(
            "<h1>board-reader viewer: no UI build</h1>"
            "<p>Run <code>make ui</code>, or from <code>packages/viewer/ui</code>:<br>"
            "<code>pnpm install --frozen-lockfile &amp;&amp; pnpm build</code></p>",
            status_code=503,
        )
    return HTMLResponse(page.read_text(encoding="utf-8"))


def create_app() -> FastAPI:
    app = FastAPI(title="board-reader viewer")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    # All four pages are served with nothing injected: every endpoint they use is
    # same-origin and relative, and :mod:`board_reader_viewer.proxy` relays it. That is
    # the whole point — one origin, one certificate to accept, and an address the
    # browser can always reach because it is the one it already loaded the page from.
    # BR_INFERENCE_WS/HTTP still configure the *viewer's* hop to the node; the browser
    # no longer sees them.
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Live page: position events from the relayed /ws/viewer socket."""
        return _page(_WEB, "index.html")

    @app.get("/snapshot", response_class=HTMLResponse)
    async def snapshot() -> HTMLResponse:
        """Phone snapshot page: photo -> corner-tap -> rectify -> POST /estimate."""
        return _page(_WEB, "snapshot.html")

    @app.get("/camera", response_class=HTMLResponse)
    async def camera() -> HTMLResponse:
        """Phone camera page: the browser edge, a dumb camera (DESIGN D10)."""
        return _page(_WEB, "camera.html")

    @app.get("/control", response_class=HTMLResponse)
    async def control() -> HTMLResponse:
        """Laptop control page: sets what the phone camera films and streams."""
        return _page(_WEB, "control.html")

    # The phone's trial log sink — registered BEFORE proxy.install so the /api
    # catch-all never relays it to the inference node: these notes are about the
    # *client*, and the viewer is the machine an operator debugs a trial from.
    # Viewer-local, no board_reader_* imports — layer separation holds.
    #
    # Opt-in like every other on-disk writer in this project (twice bitten):
    # ``BR_CLIENT_LOG_DIR`` unset -> the endpoint accepts and drops, so the page
    # never sees an error either way. ``main()`` opts real servers in by default.
    log_dir = os.environ.get("BR_CLIENT_LOG_DIR")
    log_path = (
        Path(log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}.ndjson" if log_dir else None
    )

    # The sink is reachable by anything on the LAN (the viewer binds 0.0.0.0 so
    # the phone can), so it is bounded everywhere: per-batch body cap, per-batch
    # note cap, and a per-file ceiling past which batches are dropped. The page's
    # own batches are a few hundred bytes every few seconds; only abuse hits these.
    MAX_LOG_BODY = 128 * 1024
    MAX_LOG_NOTES = 500
    MAX_LOG_FILE = 50 * 1024 * 1024

    @app.post("/api/client-log")
    async def client_log(request: Request) -> dict:
        body = await request.body()
        if len(body) > MAX_LOG_BODY:
            return {"kept": 0}
        try:
            payload = json.loads(body)
        except ValueError:
            return {"kept": 0}
        notes = payload.get("notes") if isinstance(payload, dict) else None
        if not isinstance(notes, list) or log_path is None:
            return {"kept": 0}
        notes = notes[:MAX_LOG_NOTES]
        received = time.time()
        client = request.client.host if request.client else None
        lines = "".join(
            json.dumps({**note, "received_ts": received, "client": client}) + "\n"
            for note in notes
            if isinstance(note, dict)
        )
        # Open-append-close per batch: batches arrive every few seconds and this
        # keeps the handle-lifetime problem (reload, rotation, crash) nonexistent.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size >= MAX_LOG_FILE:
            return {"kept": 0}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(lines)
        return {"kept": len(notes)}

    # /ws/ingest, /ws/viewer and /api/{path} — registered before the /_app mount below
    # so a future catch-all there can never shadow them; Starlette matches routes in
    # registration order.
    proxy.install(app)

    # The hashed JS/CSS bundle the three pages above `modulepreload`/import by
    # relative `./_app/...` URL. A plain StaticFiles mount is enough — SvelteKit
    # content-hashes every filename, so there is nothing here that needs cache
    # control tuning or an index fallback. Mounted only when it exists: StaticFiles
    # checks its directory eagerly, so an unbuilt checkout would otherwise fail to
    # construct an app at all — which is the coupling this whole function avoids.
    if (_WEB / "_app").is_dir():
        app.mount("/_app", StaticFiles(directory=_WEB / "_app"), name="web-app")
    return app


# No module-level ``app``. Building one at import time would make the whole package
# unimportable without a frontend build: ``__init__`` imports ``server``, so a fresh
# checkout that has not run ``make ui`` could not even import ``proxy`` or ``tls`` — the
# two modules that have nothing to do with the bundle — and their tests died at
# collection. ``main()`` passes the factory to uvicorn instead, so ``_require_web``
# still fires, but when someone actually starts a server.


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
    lines.append(f"  control page (on this machine): {scheme}://localhost:{port}/control")
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

    # Before anything is announced or a certificate is minted: starting a server whose
    # pages 503 is never what anyone wants, and the operator should learn that from one
    # readable line rather than from a blank phone screen at the board.
    _require_web(_WEB)

    # A real viewer keeps the phone's trial notes by default (mirroring the
    # inference node's ingress recording); a test app writes nothing unless it
    # sets the env itself. ``setdefault`` so an operator can still point it
    # elsewhere — or at "" to disable.
    os.environ.setdefault("BR_CLIENT_LOG_DIR", "client-logs")

    cert = tls.ensure_certificate() if args.ssl else None
    _announce(args.host, args.port, cert=cert)
    uvicorn.run(
        # A factory, not a module-level instance: see the note above `create_app`.
        # uvicorn calls it once in the worker, so a missing bundle still fails loudly
        # at startup rather than on the first request.
        "board_reader_viewer.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        ssl_certfile=str(cert.certfile) if cert else None,
        ssl_keyfile=str(cert.keyfile) if cert else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
