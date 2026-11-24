/**
 * `/ws/<path>` → the inference node's `/<path>` socket.
 *
 * SvelteKit has no WebSocket upgrade, so this is the one relay that lives in the
 * custom server (`main.ts`); `/api/*` is a SvelteKit route. It is a catch-all and pure
 * glue: every message goes across as it came, text as text and binary as binary, in
 * order, so the ingest pair (a JSON header, then its JPEG) stays a pair and a socket
 * the node adds needs nothing here.
 */

import type { IncomingMessage } from 'node:http';
import type { Duplex } from 'node:stream';
import { type RawData, WebSocket, WebSocketServer } from 'ws';

/** The close code a page gets when the node, not the page, is the problem. */
export const UPSTREAM_UNREACHABLE = 1011;

/** The node's socket origin: `BR_INFERENCE_WS` names its `/viewer` socket; the rest live beside it. */
export function wsOrigin(env: NodeJS.ProcessEnv): string {
  return new URL(env['BR_INFERENCE_WS'] || 'ws://localhost:8000/viewer').origin;
}

/** Codes a close frame may carry (RFC 6455 §7.4); 1005 and 1006 only describe a close. */
const sendable = (code: number): boolean =>
  (code >= 1000 && code <= 1014 && ![1004, 1005, 1006].includes(code)) || (code >= 3000 && code <= 4999);

/** A close reason cut to the 123 bytes a close frame holds, never inside a character. */
function closeReason(text: string): string {
  let out = '';
  for (const char of text) {
    if (Buffer.byteLength(out + char) > 123) break;
    out += char;
  }
  return out;
}

/** Pipe one accepted browser socket to `target` on the node, both ways. */
function bridge(client: WebSocket, target: string, logError: (line: string) => void): void {
  const upstream = new WebSocket(target, { perMessageDeflate: false, handshakeTimeout: 5_000 });
  // The browser may speak before the node has answered; hold its messages until then.
  const early: Array<[RawData, boolean]> = [];

  client.on('message', (data, isBinary) => {
    if (upstream.readyState === WebSocket.OPEN) upstream.send(data, { binary: isBinary });
    else early.push([data, isBinary]);
  });
  upstream.on('open', () => {
    for (const [data, isBinary] of early.splice(0)) upstream.send(data, { binary: isBinary });
  });
  upstream.on('message', (data, isBinary) => client.send(data, { binary: isBinary }));

  // Accept-then-close rather than a failed handshake: the page gets a code and the
  // node's own error to show, where a refused handshake is a bare "connection error".
  upstream.on('error', (error) => {
    if (client.readyState !== WebSocket.OPEN) return; // the browser left first
    logError(`relay ${target}: ${error.message}`);
    client.close(UPSTREAM_UNREACHABLE, closeReason(`${target}: ${error.message}`));
  });
  upstream.on('close', (code, why) => client.close(sendable(code) ? code : 1000, why));
  client.on('close', (code, why) => {
    if (upstream.readyState === WebSocket.CONNECTING) upstream.terminate();
    else upstream.close(sendable(code) ? code : 1000, why);
  });
}

/**
 * The `upgrade` listener for both of the viewer's servers: relay `/ws/<path>`, refuse
 * anything else.
 */
export function relayUpgrade(origin: string, logError: (line: string) => void) {
  const sockets = new WebSocketServer({ noServer: true });
  return (req: IncomingMessage, socket: Duplex, head: Buffer): void => {
    const url = req.url ?? '/';
    if (!url.startsWith('/ws/')) {
      socket.end('HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n');
      return;
    }
    const target = origin + url.slice('/ws'.length); // path and query, still encoded
    sockets.handleUpgrade(req, socket, head, (client) => bridge(client, target, logError));
  };
}
