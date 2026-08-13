"""What the viewer *server* is responsible for — and nothing else.

This replaces ``test_viewer.py`` and ``test_camera_page.py``, which between them made
~85 string assertions against hand-written HTML (``assert 'const STALE_MS = 5000;' in
html``). The pages are a bundle now — hashed chunks, minified identifiers — so those
assertions could only have been rewritten as greps against Svelte *source*, which tests
nothing about what actually ships. The behaviour they were guarding moved to where it
can be executed:

* the maths and the stores → Vitest, ``packages/viewer/ui/src/**/*.test.ts``
* what the three pages do in a browser → Playwright, ``packages/viewer/ui/tests/``

What is left here is what only Python can answer: which file each route serves, that the
relay still matches before the file server, that a configured node address never reaches
a browser, what a missing build says, and what the startup banner tells the operator.

One grep survives, in ``test_no_page_or_asset_names_an_absolute_origin`` — because
"every URL the pages use is relative" is a *deployment* invariant (it is what makes them
work over https and behind a tunnel), not an implementation detail, and the only place
to check it is the bytes that ship.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _bundle import PAGES, WEB
from starlette.routing import Mount
from starlette.testclient import TestClient

from board_reader_viewer import tls
from board_reader_viewer.server import _banner, _require_web, create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# --------------------------------------------------------------------------- routes
def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize(("route", "filename"), sorted(PAGES.items()))
def test_each_page_route_serves_its_own_prerendered_file(client, route, filename):
    """Byte-identical to the file on disk: the server injects nothing, and the route
    table maps each path to the right one of three near-identical documents."""
    response = client.get(route)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == (WEB / filename).read_text(encoding="utf-8")


def test_the_three_pages_are_actually_different_documents(client):
    bodies = {route: client.get(route).text for route in PAGES}
    assert len(set(bodies.values())) == 3


def test_the_hashed_assets_a_page_loads_all_resolve(client):
    """A page whose `/_app/...` chunks 404 is a blank screen with a clean 200 in the
    log, so the mount is checked against the URLs the page itself names."""
    html = client.get("/").text
    assets = sorted(set(re.findall(r'/_app/immutable/[A-Za-z0-9._/-]+', html)))
    # A prerendered SvelteKit page always modulepreloads its entry plus its chunks;
    # zero matches would mean this test had quietly stopped testing anything.
    assert len(assets) >= 2
    for url in assets:
        assert client.get(url).status_code == 200, url


def test_an_unknown_asset_is_a_404_not_a_page(client):
    """The mount must not fall back to index.html: a stale chunk URL has to fail
    loudly, not be answered with an HTML document a browser will try to parse as JS."""
    assert client.get("/_app/immutable/nope.js").status_code == 404


# ------------------------------------------------------------------- routing order
RELAY_PATHS = ("/api/{path:path}", "/ws/ingest", "/ws/viewer")


def test_the_relay_is_registered_before_any_static_mount():
    """Starlette matches in registration order, so a file server mounted first would
    shadow the relay. `proxy.install(app)` runs before the `/_app` mount for exactly
    that reason; this pins the ordering so a future catch-all cannot quietly invert it."""
    app = create_app()
    paths = [getattr(route, "path", None) for route in app.routes]
    for path in RELAY_PATHS:
        assert path in paths, path
    last_relay = max(paths.index(path) for path in RELAY_PATHS)
    mounts = [i for i, route in enumerate(app.routes) if isinstance(route, Mount)]
    assert mounts, "the app serves no static files at all"
    assert min(mounts) > last_relay


@pytest.mark.parametrize(
    "path",
    [
        "/api/estimate",
        # Paths that look exactly like something a file server would want to answer:
        # if a static route ever matched first, these are what would break.
        "/api/index.html",
        "/api/_app/version.json",
        "/api/",
    ],
)
def test_api_paths_reach_the_relay_and_never_the_file_server(monkeypatch, path):
    # Port 1 is closed, so the relay's own "node unreachable" 502 is proof that the
    # request was relayed rather than served (or 404ed) locally.
    monkeypatch.setenv("BR_INFERENCE_HTTP", "http://127.0.0.1:1")
    response = TestClient(create_app()).get(path)
    assert response.status_code == 502
    assert "inference node unreachable" in response.json()["error"]


@pytest.mark.parametrize("path", ["/ws/ingest", "/ws/viewer"])
def test_socket_paths_reach_the_relay(monkeypatch, path):
    monkeypatch.setenv("BR_INFERENCE_WS", "ws://127.0.0.1:1/viewer")
    with TestClient(create_app()).websocket_connect(path) as ws:
        message = ws.receive()
    # Accepted, then closed with the upstream's reason — a static route would have
    # rejected the handshake outright.
    assert message["type"] == "websocket.close"
    assert message["code"] == 1011


# ------------------------------------------------------------------- single origin
def _built_sources() -> dict[str, str]:
    """Every byte that reaches a browser: the three pages and the whole bundle."""
    files = [WEB / name for name in PAGES.values()]
    files += [p for p in (WEB / "_app").rglob("*") if p.is_file()]
    return {
        str(path.relative_to(WEB)): path.read_text(encoding="utf-8", errors="replace")
        for path in files
    }


# Each pattern is a way of naming a host the browser might not be able to reach.
# Prose that merely *mentions* an address is fine and does exist (the camera page's
# "adb reverse" recipe interpolates the port it was served on, so it ships as
# `http://localhost:${...}` and matches none of these); what must not exist is a
# concrete origin the code could fetch or open a socket to.
ABSOLUTE_ORIGIN = {
    # A hardcoded socket origin at all: the scheme has to be derived from the page,
    # or an https page blocks its own ws:// as mixed content.
    "a websocket origin": re.compile(r"wss?://"),
    # `http://192.168.1.7:8000` — the LAN-only address that used to be injected.
    "an IP-literal origin": re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}"),
    # `http://localhost:8000`, `https://jetson.lan:8443` — any absolute origin with a
    # port, which is what an inference-node address always looks like.
    "an origin with an explicit port": re.compile(r"https?://[A-Za-z0-9._-]*:\d+"),
}


def test_no_page_or_asset_names_an_absolute_origin():
    """The single-origin property, checked against what actually ships.

    The live page and /snapshot used to carry an absolute `http://<lan-ip>:8000`, which
    made them LAN-only: blocked mixed content over https, and a host no browser behind a
    tunnel could resolve. Every URL is relative now (`$lib/api`), and this is the test
    that a bundler, a dependency or a stray literal cannot put an origin back.
    """
    offenders = [
        f"{name}: {what} ({match.group(0)!r})"
        for name, text in _built_sources().items()
        for what, pattern in ABSOLUTE_ORIGIN.items()
        if (match := pattern.search(text)) is not None
    ]
    assert not offenders, offenders


def test_the_bundle_addresses_the_node_through_the_viewers_own_relay():
    """The negative above is only meaningful if the pages do reach the node at all —
    they must, and only by the viewer's relative prefixes."""
    joined = "\n".join(_built_sources().values())
    for literal in ("/api", "/ws/viewer", "/ws/ingest"):
        assert re.search(rf"""['"`]{re.escape(literal)}['"`]""", joined), literal


def test_the_configured_node_address_never_reaches_the_browser(monkeypatch, client):
    """BR_INFERENCE_WS/HTTP configure the *viewer's* hop. Injecting them into a page is
    what made it LAN-only; the server must render them nowhere."""
    monkeypatch.setenv("BR_INFERENCE_WS", "ws://example.test:9000/viewer")
    monkeypatch.setenv("BR_INFERENCE_HTTP", "http://example.test:9000")
    for route in PAGES:
        assert "example.test" not in client.get(route).text, route


# ---------------------------------------------------------------- no build present
def test_a_missing_build_is_one_readable_instruction(tmp_path):
    """A fresh checkout has no `web/`. The operator must get a line telling them what
    to run — not a traceback, and never a server that starts and serves a blank page.

    `SystemExit` is the load-bearing part: it is the one exception Python's default
    handler prints as a bare message.
    """
    with pytest.raises(SystemExit) as raised:
        _require_web(tmp_path)
    message = str(raised.value)
    assert str(tmp_path) in message
    assert "make ui" in message
    assert "pnpm build" in message


def test_a_directory_without_the_entry_page_counts_as_no_build(tmp_path):
    """Half a build is not a build: an empty `web/` (or one whose prerender failed) must
    take the same path as a missing one rather than mounting and 404ing per request."""
    (tmp_path / "_app").mkdir()
    with pytest.raises(SystemExit):
        _require_web(tmp_path)


def test_a_real_build_is_accepted_unchanged():
    assert _require_web(WEB) == WEB


# ----------------------------------------------------------------- startup banner
def test_startup_prints_the_https_address_to_type_on_the_phone(monkeypatch, tmp_path):
    """The whole point of --ssl is that a phone can reach /camera, so startup has to
    hand the operator the exact URL — nobody guesses their own LAN address."""
    monkeypatch.setattr(tls, "local_ipv4_addresses", lambda: ("192.168.1.7",))
    cert = tls.ensure_certificate(tmp_path)
    banner = "\n".join(_banner("0.0.0.0", 8080, cert=cert))
    assert "https://192.168.1.7:8080/camera" in banner
    assert "https://localhost:8080/camera" in banner
    assert str(cert.certfile) in banner
    assert "192.168.1.7" in banner  # what the certificate is valid for
    assert "your phone will warn once" in banner


def test_startup_without_ssl_says_why_the_camera_will_refuse():
    """Plain http is a perfectly good way to run the viewer and a dead end for the
    camera page; the operator should learn that at startup, not from the phone."""
    banner = "\n".join(_banner("127.0.0.1", 8080, cert=None))
    assert "getUserMedia needs a secure context" in banner
    assert "--ssl" in banner
    assert "adb reverse tcp:8080 tcp:8080" in banner
    assert "pass --host 0.0.0.0 to reach it from a phone" in banner


# --------------------------------------------------------------- layer separation
def test_the_viewer_imports_no_board_reader_package():
    """CLAUDE.md's separation rule: the viewer serves assets and forwards bytes. It
    depends on neither core, edge nor inference, and this is checked over every module
    in the package rather than only the one that happens to be imported here."""
    import board_reader_viewer

    package = Path(board_reader_viewer.__file__).resolve().parent
    modules = sorted(package.rglob("*.py"))
    assert len(modules) >= 4, modules
    for module in modules:
        text = module.read_text(encoding="utf-8")
        for forbidden in ("board_reader_core", "board_reader_edge", "board_reader_inference"):
            assert forbidden not in text, f"{module.name}: {forbidden}"


# ----------------------------------------------------------------- client trial log
def test_client_log_is_viewer_local_and_appends_ndjson(tmp_path, monkeypatch):
    """The phone's trial notes land on the viewer's OWN disk — the /api catch-all
    must not relay this one to the node, whose logs are about what *arrived*."""
    import json

    monkeypatch.setenv("BR_CLIENT_LOG_DIR", str(tmp_path / "logs"))
    client = TestClient(create_app())
    reply = client.post(
        "/api/client-log",
        json={"notes": [{"t": 1, "kind": "camera-stall-reopen"}, {"t": 2, "kind": "pagehide"}]},
    )
    assert reply.json() == {"kept": 2}

    (log_file,) = list((tmp_path / "logs").iterdir())
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["camera-stall-reopen", "pagehide"]
    assert all("received_ts" in line and "client" in line for line in lines)


def test_client_log_without_a_dir_accepts_and_drops(monkeypatch):
    """Unset (the test/default-app state) -> 200 and nothing written anywhere:
    the page must never see an error for logging, and a test app never writes."""
    monkeypatch.delenv("BR_CLIENT_LOG_DIR", raising=False)
    client = TestClient(create_app())
    assert client.post("/api/client-log", json={"notes": [{"t": 1, "kind": "x"}]}).json() == {
        "kept": 0
    }


def test_client_log_rejects_garbage_without_erroring(tmp_path, monkeypatch):
    monkeypatch.setenv("BR_CLIENT_LOG_DIR", str(tmp_path / "logs"))
    client = TestClient(create_app())
    assert client.post("/api/client-log", content=b"not json").json() == {"kept": 0}
    assert client.post("/api/client-log", json={"notes": "nope"}).json() == {"kept": 0}
    assert not (tmp_path / "logs").exists()


def test_client_log_bounds_the_batch_and_the_body(tmp_path, monkeypatch):
    """LAN-reachable sink: an oversized body is dropped whole, an oversized
    batch is truncated — nothing here may buffer or write without a ceiling."""
    monkeypatch.setenv("BR_CLIENT_LOG_DIR", str(tmp_path / "logs"))
    client = TestClient(create_app())

    huge = {"notes": [{"t": 1, "kind": "x", "pad": "y" * 200_000}]}
    assert client.post("/api/client-log", json=huge).json() == {"kept": 0}
    assert not (tmp_path / "logs").exists()

    many = {"notes": [{"t": i, "kind": "tick"} for i in range(600)]}
    assert client.post("/api/client-log", json=many).json() == {"kept": 500}
