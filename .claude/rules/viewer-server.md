---
paths:
  - 'packages/viewer/src/**'
  - 'packages/viewer/tests/**'
  - 'packages/viewer/pyproject.toml'
---

# The viewer server (Python)

- Keep the viewer the single origin the browser talks to: pages use relative URLs, the viewer relays `/api/<path>` (one catch-all, not a route per endpoint) and the `/ws/*` sockets to the inference node as pure network glue with no `board_reader_*` import, because an absolute node URL is blocked mixed content once the page is served over TLS, names a host no off-LAN browser reaches, and breaks the one-port tunnel. `/camera` needs a secure context, and a browser accepts no self-signed certificate for a second origin's `wss://`.
- Keep `--ssl`'s minted certificate's SANs (localhost, hostname, current LAN IPv4s) and its `serverAuth` EKU, because iOS rejects a TLS server certificate without that EKU and offers no override.
- Forward the upstream response headers in the relay and strip only the hop-by-hop ones (content-length, content-encoding, transfer-encoding, connection, keep-alive), because a page may depend on a header the node sets, and copying the hop-by-hop ones contradicts the body Starlette sends.
- Keep `create_app()` importable and buildable without the prerendered bundle in `src/board_reader_viewer/web/` (pages return 503; only `main()` refuses to start), and keep no module-level `app` (uvicorn gets `create_app` with `factory=True`), because otherwise the relay and TLS tests die at collection on a fresh clone with no `make ui`. Check it by moving `web/` away and running `uv run pytest packages/viewer`.
- Give every new SvelteKit route its own `@app.get("/<name>")` returning `_page(_WEB, "<name>.html")` in `server.py`, because the viewer serves each prerendered page by an explicit route (only `/_app` is mounted); `vite preview` and the Playwright suite serve the build directly, so they pass while the real viewer answers 404.
