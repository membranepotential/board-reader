/** The `/api/[...path]` relay, called directly against a real stand-in node. */

import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

const env = vi.hoisted((): Record<string, string> => ({}));
vi.mock('$app/env/private', () => ({
  get BR_INFERENCE_HTTP() {
    return env['BR_INFERENCE_HTTP'];
  }
}));

import { fallback } from './+server';

type Event = Parameters<typeof fallback>[0];

/** Call the route the way SvelteKit would for `path` on the viewer. */
function call(path: string, init?: RequestInit): Promise<Response> {
  const request = new Request(`http://viewer${path}`, init);
  return Promise.resolve(fallback({ request, url: new URL(request.url) } as Event));
}

/** Reflects each request as JSON; a camera report is refused, in the node's words. */
const node = http.createServer((req, res) => {
  const chunks: Buffer[] = [];
  req.on('data', (chunk: Buffer) => chunks.push(chunk));
  req.on('end', () => {
    const body = JSON.stringify({
      path: req.url,
      method: req.method,
      contentType: req.headers['content-type'] ?? null,
      body: Buffer.concat(chunks).toString('latin1')
    });
    res.writeHead(req.url?.endsWith('/camera/report') ? 422 : 200, {
      'content-type': 'application/json',
      'content-length': Buffer.byteLength(body),
      'x-seq': '42'
    });
    res.end(body);
  });
});

beforeAll(async () => {
  await new Promise<void>((resolve) => node.listen(0, '127.0.0.1', resolve));
  env['BR_INFERENCE_HTTP'] = `http://127.0.0.1:${(node.address() as AddressInfo).port}`;
});
afterAll(() => node.close());

describe('the /api relay', () => {
  it('forwards method, body, query and content type verbatim', async () => {
    const res = await call('/api/estimate?board_id=snapshot&note=10,20%2C30', {
      method: 'POST',
      body: Buffer.from('\xff\xd8jpeg-ish', 'latin1'),
      headers: { 'content-type': 'image/jpeg' }
    });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      // Byte for byte: a re-encoded comma or %2C would reach the node as another value.
      path: '/estimate?board_id=snapshot&note=10,20%2C30',
      method: 'POST',
      contentType: 'image/jpeg',
      body: '\xff\xd8jpeg-ish'
    });
  });

  it.each(['/api/labels/upload-7', '/api/boards/b%201/frame?seq=42', '/api/uploads/abc/photo?max=160'])(
    'relays any node path: %s',
    async (path) => {
      expect((await (await call(path)).json()).path).toBe(path.slice('/api'.length));
    }
  );

  it("returns the node's refusal, status and headers unchanged", async () => {
    const res = await call('/api/boards/phone/camera/report', { method: 'PUT', body: '{}' });
    expect(res.status).toBe(422);
    expect(res.headers.get('x-seq')).toBe('42');
    expect((await res.json()).method).toBe('PUT');
  });

  it('answers an unreachable node with a 502 that names it', async () => {
    // A port that was just free: nothing listens, so the connect is refused at once.
    const probe = http.createServer();
    await new Promise<void>((resolve) => probe.listen(0, '127.0.0.1', resolve));
    const closedPort = (probe.address() as AddressInfo).port;
    await new Promise((resolve) => probe.close(resolve));

    const configured = env['BR_INFERENCE_HTTP'];
    env['BR_INFERENCE_HTTP'] = `http://127.0.0.1:${closedPort}`;
    vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      const res = await call('/api/estimate', { method: 'POST', body: 'photo' });
      expect(res.status).toBe(502);
      expect((await res.json()).error).toBe(
        `inference node unreachable at http://127.0.0.1:${closedPort}: connect ECONNREFUSED 127.0.0.1:${closedPort}`
      );
      // `localhost` fails on each of its addresses; the reason names them, not nothing.
      env['BR_INFERENCE_HTTP'] = `http://localhost:${closedPort}`;
      const both = await (await call('/api/healthz')).json();
      expect(both.error).toContain(`ECONNREFUSED`);
    } finally {
      env['BR_INFERENCE_HTTP'] = configured as string;
    }
  });
});
