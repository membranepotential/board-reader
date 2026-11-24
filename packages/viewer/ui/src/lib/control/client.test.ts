import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCamera, parseCameraView, parseHealth, previewUrl, putCamera } from './client';

/** `CameraRegistry.view` as the node spells it, with a phone that has reported. */
const VIEW = {
  board_id: 'phone',
  config: { crop: { x: 0.1, y: 0.2, w: 0.5, h: 0.6 }, device: 'Back Camera', resolution: '1080p', streaming: true },
  report: {
    devices: [{ id: 'abc', label: 'Back Camera' }],
    device: 'Back Camera',
    frame: { width: 1920, height: 1080 },
    ingest: 'streaming'
  },
  seen_at: 99.5,
  preview: { seq: 3, at: 99, width: 640, height: 360 },
  frame: { seq: 41, received_at: 99.8, width: 960, height: 648 },
  now: 100
};

function reply(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

afterEach(() => vi.unstubAllGlobals());

describe('parseCameraView', () => {
  it('reads the whole view the node serves', () => {
    const parsed = parseCameraView(VIEW);
    expect(parsed).toEqual({ ok: true, value: VIEW });
  });

  it('reads a board nothing has touched yet: no report, no preview, no frame', () => {
    const fresh = { ...VIEW, report: null, seen_at: null, preview: null, frame: null, config: { crop: null, device: null, resolution: '1080p', streaming: false } };
    expect(parseCameraView(fresh).ok).toBe(true);
  });

  it('refuses a report it cannot read rather than showing half of one', () => {
    expect(parseCameraView({ ...VIEW, report: { ...VIEW.report, ingest: 'streming' } }).ok).toBe(false);
    expect(parseCameraView({ ...VIEW, report: { ...VIEW.report, frame: { width: 0, height: 1 } } }).ok).toBe(false);
    expect(parseCameraView({ ...VIEW, now: 'soon' }).ok).toBe(false);
  });
});

describe('parseHealth', () => {
  it('takes the recognizer and the refused count from /healthz', () => {
    const health = {
      status: 'ok',
      ingest: { received: 10, refused: 2, estimated: 1 },
      recognizer: { name: 'stub', model: null, fallback: 'no weights' }
    };
    expect(parseHealth(health)).toEqual({
      ok: true,
      value: { recognizer: 'stub', fallback: 'no weights', received: 10, refused: 2 }
    });
  });
});

describe('requests', () => {
  it('PUTs the whole config as JSON to the relayed camera route', async () => {
    const fetchMock = vi.fn(async () => reply(VIEW));
    vi.stubGlobal('fetch', fetchMock);
    const result = await putCamera('my board', VIEW.config as Parameters<typeof putCamera>[1]);
    expect(result.ok).toBe(true);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/boards/my%20board/camera');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body as string)).toEqual(VIEW.config);
  });

  it("passes the node's own refusal wording through", async () => {
    vi.stubGlobal('fetch', async () => reply({ error: 'crop sides must be at least 0.05 of the frame' }, 422));
    const result = await getCamera('phone');
    expect(result).toEqual({
      ok: false,
      error: { status: 422, message: 'crop sides must be at least 0.05 of the frame' }
    });
  });

  it('builds the preview URL with a cache-busting seq', () => {
    expect(previewUrl('phone', 7)).toBe('/api/boards/phone/camera/preview?seq=7');
  });
});
