import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  apiPath,
  DETECT_TIMEOUT_MS,
  boardFrameUrl,
  detect,
  estimate,
  listUploads,
  uploadPhotoUrl,
  wsUrl
} from './api';
import type { CornerQuad } from './protocol';

const QUAD: CornerQuad = [
  [0, 0],
  [100, 0],
  [100, 100],
  [0, 100]
];

/** Everything a page could ask for must come back as a same-origin, relative URL. */
function isRelative(url: string): boolean {
  return url.startsWith('/') && !url.startsWith('//');
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('URLs stay same-origin', () => {
  it('prefixes the node path with the viewer relay', () => {
    expect(apiPath('/estimate')).toBe('/api/estimate');
    expect(apiPath('/estimate', { board_id: 'snapshot' })).toBe('/api/estimate?board_id=snapshot');
  });

  it('omits absent parameters entirely', () => {
    expect(uploadPhotoUrl('abc')).toBe('/api/uploads/abc/photo');
    expect(uploadPhotoUrl('abc', 160)).toBe('/api/uploads/abc/photo?max=160');
  });

  it('escapes an id that would otherwise change the path', () => {
    expect(uploadPhotoUrl('../secret')).toBe('/api/uploads/..%2Fsecret/photo');
    expect(boardFrameUrl('a b', 3)).toBe('/api/boards/a%20b/frame?seq=3');
  });

  it('is relative for every endpoint the pages use', () => {
    for (const url of [
      apiPath('/estimate'),
      apiPath('/detect'),
      apiPath('/uploads', { limit: 100 }),
      uploadPhotoUrl('x'),
      apiPath('/labels/x'),
      boardFrameUrl('phone', 1)
    ]) {
      expect(isRelative(url), url).toBe(true);
    }
  });

  it('refuses a path that is not a path', () => {
    expect(() => apiPath('http://example.invalid/estimate')).toThrow(TypeError);
    expect(() => apiPath('estimate')).toThrow(TypeError);
  });

  it('starts from the relay prefix, never a configured origin', () => {
    expect(API_PREFIX).toBe('/api');
  });
});

describe('wsUrl', () => {
  it('follows the page from http to ws', () => {
    expect(wsUrl('/ws/viewer', 'http://192.168.1.5:8080/')).toBe('ws://192.168.1.5:8080/ws/viewer');
  });

  it('follows the page from https to wss — the tunnel case', () => {
    expect(wsUrl('/ws/ingest', 'https://x.trycloudflare.com/camera')).toBe(
      'wss://x.trycloudflare.com/ws/ingest'
    );
  });

  it('says so rather than guessing when there is no page (prerender)', () => {
    expect(() => wsUrl('/ws/viewer')).toThrow(TypeError);
  });
});

describe('failures carry the node own wording', () => {
  it('surfaces the {error} body a refusal states', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ error: 'could not decode the image' }), {
          status: 400
        })
      )
    );
    const result = await estimate(new Blob(['x']), 'snapshot');
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.status).toBe(400);
    expect(result.error.message).toBe('could not decode the image');
  });

  it('keeps a non-JSON body rather than inventing a reason', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('upstream exploded', { status: 502 })));
    const result = await detect(new Blob(['x']));
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.message).toBe('upstream exploded');
  });

  it('reports an unreachable viewer as status 0, never as a 200 with no data', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('network down');
      })
    );
    const result = await listUploads();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.status).toBe(0);
  });

  it('sends the photo as image/jpeg with the board in the query', async () => {
    const fetchSpy = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ board_id: 'snapshot', seq: 0, utc: 'now', placement: '8/8/8/8/8/8/8/8' })
        )
    );
    vi.stubGlobal('fetch', fetchSpy);
    const result = await estimate(new Blob(['jpeg']), 'snapshot');
    expect(result.ok).toBe(true);
    const [url, init] = fetchSpy.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/estimate?board_id=snapshot');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'image/jpeg' });
  });
});

describe('a request that never answers', () => {
  it('gives /detect its own clock, so a wedged node cannot hold the camera page', async () => {
    // The camera page hands the camera and the quad to exactly one activity at a
    // time; a fetch that hangs for ever would leave every idle-gated control
    // disabled with nothing on screen to say why (review, 0ebbbd2).
    const fetchSpy = vi.fn(async () => new Response(JSON.stringify({ corners: QUAD })));
    vi.stubGlobal('fetch', fetchSpy);
    await detect(new Blob(['jpeg']));
    const [, init] = fetchSpy.mock.calls[0] as unknown as [string, RequestInit];
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(DETECT_TIMEOUT_MS).toBeGreaterThan(4000); // a MISS costs the node 2-4 s
  });

  it('reads as an answer that never came, not as an unreachable viewer', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new DOMException('signal timed out', 'TimeoutError');
      })
    );
    const result = await detect(new Blob(['jpeg']));
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.status).toBe(0);
    expect(result.error.message).toBe('the node did not answer in time');
  });
});
