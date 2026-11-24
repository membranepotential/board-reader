/**
 * The viewer: `node server/main.ts [--host H] [--port P] [--tls-port T]`.
 *
 * adapter-node's custom-server pattern. The SvelteKit app (`build/handler.js`) serves
 * the pages and every HTTP route, the `/api/*` relay included; this file adds only
 * what SvelteKit cannot do: the `/ws/*` WebSocket relay (`ws.ts`) and, with
 * `--tls-port`, a second port over TLS with a cached self-signed certificate
 * (`tls.ts`), which `/camera` on a phone needs. Node runs it by stripping its types.
 *
 * Environment: BR_INFERENCE_HTTP / BR_INFERENCE_WS (the node), BR_CLIENT_LOG_DIR
 * (the phone's notes; default `client-logs`, empty drops them). It logs its banner and
 * one line per error, nothing per request.
 */

import { existsSync } from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { banner } from './banner.ts';
import { cacheDir, ensureCertificate, localIPv4Addresses, subjectNames } from './tls.ts';
import { relayUpgrade, wsOrigin } from './ws.ts';

function fail(message: string): never {
  console.error(`board-reader viewer: ${message}`);
  process.exit(2);
}

function port(value: string): number {
  const n = Number(value);
  return Number.isInteger(n) && n >= 0 && n <= 65_535 ? n : fail(`not a port: ${value}`);
}

const { values: args } = parseArgs({
  options: {
    host: { type: 'string', default: '127.0.0.1' },
    port: { type: 'string', default: '8080' },
    'tls-port': { type: 'string' }
  }
});

// Read by the app when the handler is imported, so set first. adapter-node caps a
// request body at 512 KB by default; a phone photo for /api/estimate is several MB.
process.env['BODY_SIZE_LIMIT'] ??= '64M';
// A real server keeps the phone's notes; `vite dev` and the tests write nothing.
process.env['BR_CLIENT_LOG_DIR'] ??= 'client-logs';

const build = new URL('../build/handler.js', import.meta.url);
if (!existsSync(build)) {
  fail(
    `no UI build at ${fileURLToPath(new URL('.', build))}\n` +
      '  run `make ui`, or from packages/viewer/ui: pnpm install --frozen-lockfile && pnpm build'
  );
}
const { handler } = (await import(build.href)) as { handler: http.RequestListener };

const lan = await localIPv4Addresses();
const tlsPort = args['tls-port'] === undefined ? undefined : port(args['tls-port']);
const cert = tlsPort === undefined ? undefined : await ensureCertificate(cacheDir(), subjectNames(lan));
const upgrade = relayUpgrade(wsOrigin(process.env), (line) => console.error(line));

const listen = (server: http.Server, at: number): Promise<void> =>
  new Promise((resolve, reject) => {
    server.on('upgrade', upgrade).once('error', reject).listen(at, args.host, resolve);
  });

try {
  await listen(http.createServer(handler), port(args.port));
  if (cert !== undefined && tlsPort !== undefined) {
    await listen(https.createServer({ cert: cert.cert, key: cert.key }, handler), tlsPort);
  }
} catch (error) {
  fail((error as Error).message);
}

// A streaming phone holds its socket open, so a graceful close would never finish;
// the relay keeps no state to flush, and a signal simply ends the process.
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => process.exit(0));

const tls = cert && tlsPort !== undefined ? { ...cert, port: tlsPort } : undefined;
console.log(
  banner({
    host: args.host,
    port: port(args.port),
    tls,
    lan,
    node: process.env['BR_INFERENCE_HTTP'] || 'http://localhost:8000'
  }).join('\n')
);
