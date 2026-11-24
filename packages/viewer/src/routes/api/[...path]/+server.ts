/**
 * `/api/<path>` → the inference node's `/<path>`, for every method.
 *
 * The pages talk to the viewer only, and the viewer speaks for the node: an absolute
 * node URL is blocked mixed content once a page is served over TLS, names a host no
 * browser behind a tunnel reaches, and would need a second certificate on a phone.
 * This is a catch-all and pure network glue: method, path, query, headers and body go
 * up unchanged, and status, headers and body come back unchanged, so a new node
 * endpoint needs nothing here and the node's own refusals reach the page in its words.
 * The sockets under `/ws/` are relayed by the custom server (`server/ws.ts`), because
 * SvelteKit has no WebSocket upgrade.
 */


import { BR_INFERENCE_HTTP } from '$app/env/private';

import type { RequestHandler } from './$types';

export const prerender = false;

// Calibration on the node runs a backbone over 64 squares and rebuilds the exemplar
// bank; on a cold CUDA session that is tens of seconds. The relay must outlast it, or
// the phone sees a timeout for a calibration that in fact succeeded.
const TIMEOUT_MS = 180_000;

// Headers about one hop's connection, not the message (RFC 9110 §7.6.1), plus
// `expect`: the client's 100-continue was already answered by this server, and Node's
// fetch refuses the header outright.
const HOP_BY_HOP = ['connection', 'keep-alive', 'proxy-connection', 'te', 'trailer', 'transfer-encoding', 'upgrade', 'expect'];

function endToEnd(headers: Headers): Headers {
  const kept = new Headers(headers);
  for (const name of HOP_BY_HOP) kept.delete(name);
  return kept;
}

/**
 * The failure's own words. fetch wraps the socket error ("ECONNREFUSED") in `cause`,
 * and a host with several addresses (`localhost`: ::1 and 127.0.0.1) fails with an
 * `AggregateError` whose own message is empty.
 */
function reason(error: Error): string {
  const cause = error.cause;
  if (cause instanceof AggregateError) return cause.errors.map((e: Error) => e.message).join('; ');
  return cause instanceof Error ? cause.message : error.message;
}

export const fallback: RequestHandler = async ({ request, url }) => {
  // `url.pathname` and `url.search` are still percent-encoded, so the node gets the
  // bytes the page sent: a decoded `%2F` or comma would reach it as a different value.
  const target = BR_INFERENCE_HTTP + url.pathname.slice('/api'.length) + url.search;
  const headers = endToEnd(request.headers);
  headers.delete('host');

  // Node's fetch needs `duplex: 'half'` to stream a request body; the DOM typings the
  // app compiles against do not know the field yet.
  const init: RequestInit & { duplex: 'half' } = {
    method: request.method,
    headers,
    body: request.body,
    duplex: 'half',
    redirect: 'manual',
    // The browser going away (a reload mid-calibration) cancels the node's request.
    signal: AbortSignal.any([request.signal, AbortSignal.timeout(TIMEOUT_MS)])
  };
  let answer: Response;
  try {
    answer = await fetch(target, init);
  } catch (error) {
    const why = reason(error as Error);
    console.error(`relay ${request.method} ${target}: ${why}`);
    return Response.json({ error: `inference node unreachable at ${BR_INFERENCE_HTTP}: ${why}` }, { status: 502 });
  }

  const out = endToEnd(answer.headers);
  // fetch has already decoded a compressed body, so its encoding and length no longer
  // describe the bytes we send.
  if (out.has('content-encoding')) {
    out.delete('content-encoding');
    out.delete('content-length');
  }
  return new Response(answer.body, { status: answer.status, statusText: answer.statusText, headers: out });
};
