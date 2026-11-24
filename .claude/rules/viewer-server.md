---
paths:
  - 'packages/viewer/ui/server/**'
  - 'packages/viewer/ui/src/routes/api/**'
  - 'packages/viewer/ui/src/hooks.server.ts'
  - 'packages/viewer/ui/src/env.ts'
  - 'packages/viewer/ui/vite.config.ts'
  - 'docker/viewer.Dockerfile'
---

# The viewer server (SvelteKit on adapter-node, plus a custom Node server)

- Keep the viewer the single origin the browser talks to: pages use relative URLs; `/api/<path>` is one catch-all route (`src/routes/api/[...path]/+server.ts`, `fallback`) and `/ws/<path>` one catch-all upgrade relay (`server/ws.ts`), both mapping to the node's `/<path>` with path and query kept percent-encoded, because an absolute node URL is blocked mixed content over TLS, names a host no off-LAN browser reaches, and breaks the one-port tunnel, and a route per endpoint makes every new node endpoint or socket a viewer change.
- Put only what SvelteKit cannot do in `server/main.ts`: the WebSocket upgrade relay and the TLS port. Every HTTP endpoint is a `+server.ts` route, configured through `src/env.ts` (`$app/env/private`), because a parallel Node app beside SvelteKit was built once and rejected as unidiomatic.
- Set adapter-node's environment in `server/main.ts` before it imports `build/handler.js` (`BODY_SIZE_LIMIT`, the `BR_CLIENT_LOG_DIR` default), because the handler reads its environment at import, and its 512 KB default body limit answers a phone photo for `/api/estimate` with a 413.
- Keep `csrf: { trustedOrigins: ['*'] }` in `vite.config.ts`, because without a reverse proxy adapter-node takes every request's origin to be `https`, so the phone's `text/plain` posts to `/api/client-log` on the plain port fail SvelteKit's cross-site check; the viewer has no cookies or session for that check to protect.
- Strip the hop-by-hop headers and `expect` from a relayed request, and drop `content-encoding` with `content-length` from the answer when present, because Node's fetch refuses an `Expect` header (curl sends one above 1 MB) and has already decoded a compressed body. Pass every other header through both ways (the node states things in `X-Seq`).
- Close a relayed browser socket with 1011 and the node's error as the reason when the node cannot be reached (accept, then close), because the pages show `event.reason` to say whose fault a dead stream is, and a refused handshake carries no reason at all.
- Generate the TLS certificate with `selfsigned`, never by hand, and keep its SANs (localhost, 127.0.0.1, the hostname, the LAN IPv4s), its `serverAuth` EKU and its cache path (`$XDG_CACHE_HOME/board-reader/viewer-tls/`), because iOS rejects a server certificate without that EKU and offers no override, and a phone that accepted the cached certificate keeps working only while the path and the names stay the same.
- Write `server/*.ts` as erasable TypeScript (no enums, namespaces or parameter properties) with `.ts` import extensions, and import only `node:` built-ins, siblings and packages under `dependencies`, because Node runs these files by stripping types and the image prunes devDependencies; `tests/test_layer_separation.py` checks the imports.
- In a test that needs an unreachable node, use a port that was just freed, never port 1, because Node's fetch refuses port 1 as a blocked port ("bad port") before it connects, so the test checks the wrong failure.
