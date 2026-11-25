/**
 * The `/camera` page — the phone as a dumb camera (DESIGN D11).
 *
 * It has no controls. Which board it streams as, what it films and whether it
 * streams come from the node, which pushes them down the phone's control socket
 * (`/ws/cameras/<device id>`). So every spec here plays the node on that socket
 * (`FakeControl`): it names the board id (the device id) and the camera's name,
 * pushes a config of its choosing, says
 * whether previews are wanted, and reads what reaches it — the reports, the
 * full-frame previews, the full frames Auto-frame asks for, and the frames on
 * `/ws/ingest`, decoded down to their JPEG size, which is the only honest proof that
 * the crop the config asked for is the crop sent. Two phones against one stateful node are `cameras.spec.ts`'s.
 */

import { expect, test, type Page, type WebSocketRoute } from '@playwright/test';
import {
  FRAME,
  LANDSCAPE,
  PORTRAIT,
  cameraRequests,
  crossHand,
  drawnBoardCamera,
  jpegSize,
  liveViewfinder
} from './helpers/camera';

/** The phone's control socket, by device id, with an optional `?board=`. */
const PHONE_SOCKET = /\/ws\/cameras\/[^/?]+(\?.*)?$/;

interface Config {
  crop: { x: number; y: number; w: number; h: number } | null;
  device: string | null;
  resolution: '720p' | '1080p' | '2160p';
  streaming: boolean;
  rules: 'standard' | 'free';
  auto: boolean;
}

interface FullFrame {
  request: number;
  captureTs: number;
  size: [number, number];
}

/** The node's half of the phone's control socket, plus a counting `/ws/ingest`. */
class FakeControl {
  config: Config = { crop: null, device: null, resolution: '1080p', streaming: false, rules: 'standard', auto: true };
  watched = false;
  readonly reports: Record<string, unknown>[] = [];
  readonly previews: [number, number][] = [];
  readonly fullFrames: FullFrame[] = [];
  readonly headers: Record<string, unknown>[] = [];
  readonly frames: [number, number][] = [];
  /** The device id each phone socket connected with, in order. */
  readonly devices: string[] = [];
  /** The board id the phone streams as: its device id, said on connect. */
  boardId = '';
  private socket: WebSocketRoute | null = null;

  static async install(page: Page): Promise<FakeControl> {
    const node = new FakeControl();
    await page.routeWebSocket(PHONE_SOCKET, (ws) => {
      node.socket = ws;
      node.boardId = new URL(ws.url()).pathname.split('/').at(-1)!;
      node.devices.push(node.boardId);
      // A full frame's header, waiting for its JPEG: the next binary message.
      let header: { request: number; capture_ts: number } | null = null;
      ws.onMessage((message) => {
        if (typeof message === 'string') {
          const parsed = JSON.parse(message) as {
            type: string;
            report: Record<string, unknown>;
            request: number;
            capture_ts: number;
          };
          header = parsed.type === 'full_frame' ? parsed : null;
          if (parsed.type === 'report') node.reports.push(parsed.report);
        } else if (header !== null) {
          node.fullFrames.push({ request: header.request, captureTs: header.capture_ts, size: jpegSize(message) });
          header = null;
        } else {
          node.previews.push(jpegSize(message));
        }
      });
      // What the node says to a phone that has just connected.
      ws.send(JSON.stringify({ type: 'assigned', board_id: node.boardId, name: 'board-1' }));
      ws.send(JSON.stringify({ type: 'config', config: node.config }));
      ws.send(JSON.stringify({ type: 'previews', on: node.watched }));
    });
    await page.route('**/api/client-log', (route) => route.fulfill({ status: 204 }));
    await page.routeWebSocket('**/ws/ingest', (ws) => {
      ws.onMessage((message) => {
        if (typeof message === 'string') node.headers.push(JSON.parse(message) as Record<string, unknown>);
        else node.frames.push(jpegSize(message));
      });
    });
    return node;
  }

  /** The control UI changed the config: the node pushes it at once. */
  configure(patch: Partial<Config>): void {
    this.config = { ...this.config, ...patch };
    this.socket?.send(JSON.stringify({ type: 'config', config: this.config }));
  }

  /** The control UI named this camera: the node tells the phone at once; the id stays. */
  rename(name: string): void {
    this.socket?.send(JSON.stringify({ type: 'assigned', board_id: this.boardId, name }));
  }

  /** Another page of this phone connected: the node says so, then closes this one. */
  replace(): void {
    this.socket?.send(JSON.stringify({ type: 'replaced' }));
    void this.socket?.close({ code: 1000, reason: 'replaced' });
  }

  /** Auto-frame wants the whole picture. */
  askFullFrame(request: number): void {
    this.socket?.send(JSON.stringify({ type: 'send_full_frame', request }));
  }

  /** A control page opened (or the last one closed). */
  watch(on: boolean): void {
    this.watched = on;
    this.socket?.send(JSON.stringify({ type: 'previews', on }));
  }

  get lastReport(): Record<string, unknown> | undefined {
    return this.reports.at(-1);
  }
}

async function openCamera(page: Page): Promise<FakeControl> {
  const node = await FakeControl.install(page);
  await drawnBoardCamera(page);
  await page.goto('/camera');
  await liveViewfinder(page);
  return node;
}

test.describe('a dumb camera', () => {
  test.use({ viewport: LANDSCAPE });

  test('has nothing to press', async ({ page }) => {
    await openCamera(page);
    await expect(page.getByRole('button')).toHaveCount(0);
    await expect(page.getByTestId('camera-status')).toContainText('board-1');
  });

  test('keeps its device id across a reload, and streams under it as its board id when named', async ({
    page
  }) => {
    const node = await openCamera(page);
    const device = node.devices[0]!;
    await expect(page.getByTestId('device-id')).toHaveText(`camera ${device.slice(0, 8)}`);
    await expect(page.getByTestId('session-id')).toHaveCount(0);
    await page.reload();
    await liveViewfinder(page);
    expect(node.devices).toEqual([device, device]);

    node.configure({ streaming: true });
    await expect.poll(() => node.headers.at(-1)?.['board_id'], { timeout: 8_000 }).toBe(device);
    // Every frame names this phone, so the node can refuse one under another's board.
    expect(node.headers.at(-1)?.['device_id']).toBe(device);
    // Named on /control: the page shows the name; the frames keep the board id.
    node.rename('table');
    await expect(page.getByTestId('camera-status')).toContainText('table');
    const sent = node.headers.length;
    await expect.poll(() => node.headers.length, { timeout: 8_000 }).toBeGreaterThan(sent);
    expect(node.headers.at(-1)?.['board_id']).toBe(device);
  });

  test('stands down when another page of the same phone takes over', async ({ page }) => {
    const node = await openCamera(page);
    node.configure({ streaming: true });
    await expect.poll(() => node.frames.length, { timeout: 8_000 }).toBeGreaterThan(0);
    node.replace();
    await expect(page.getByTestId('replaced')).toBeVisible();
    await page.waitForTimeout(500); // one in the encoder may still land
    const sent = node.frames.length;
    await page.waitForTimeout(1500);
    expect(node.frames.length).toBe(sent);
    expect(node.devices).toHaveLength(1); // it did not reconnect to take the camera back
  });

  test('says when the browser keeps no id, so a reload makes a new camera', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(window, 'localStorage', {
        get: () => {
          throw new DOMException('blocked', 'SecurityError');
        }
      });
    });
    await openCamera(page);
    await expect(page.getByTestId('session-id')).toBeVisible();
  });

  test('reports its lenses, the frame it delivers and that it is not streaming', async ({ page }) => {
    const node = await openCamera(page);
    await expect.poll(() => node.lastReport?.['frame']).toEqual(FRAME);
    expect(node.lastReport).toMatchObject({
      kind: 'phone',
      device: 'Back Camera',
      ingest: 'idle',
      devices: [
        { id: 'lens-wide', label: 'Back Camera' },
        { id: 'lens-ultra', label: 'Back Ultra Wide Camera' }
      ]
    });
    await expect(page.getByTestId('camera-status')).toContainText('node connected');
    expect(node.headers).toHaveLength(0);
    // Sent when something changed, never on a timer.
    const sent = node.reports.length;
    await page.waitForTimeout(1500);
    expect(node.reports.length).toBe(sent);
  });

  test('opens the lens and resolution the config names, once', async ({ page }) => {
    const node = await openCamera(page);
    node.configure({ device: 'Back Ultra Wide Camera', resolution: '2160p' });
    await expect
      .poll(async () => (await cameraRequests(page)).at(-1)?.video)
      .toEqual({ deviceId: { exact: 'lens-ultra' }, width: { ideal: 3840 }, height: { ideal: 2160 } });
    // The same config again (another field changed, a reconnect) does not reopen it.
    const opened = (await cameraRequests(page)).length;
    node.configure({});
    await page.waitForTimeout(1500);
    expect((await cameraRequests(page)).length).toBe(opened);
  });

  test('sends a small full-frame preview only while a control page watches', async ({ page }) => {
    const node = await openCamera(page);
    node.configure({ crop: { x: 0.25, y: 0.25, w: 0.5, h: 0.5 } });
    await page.waitForTimeout(2000);
    expect(node.previews).toHaveLength(0); // nobody watches: no uplink spent

    node.watch(true);
    // 1280x720 at a 480 px long side, whatever the crop: the control UI draws the
    // rectangle on the WHOLE field of view.
    await expect.poll(() => node.previews.length, { timeout: 8_000 }).toBeGreaterThan(1);
    expect(node.previews.at(-1)).toEqual([480, 270]);

    node.watch(false);
    await page.waitForTimeout(500); // one in the encoder may still land
    const sent = node.previews.length;
    await page.waitForTimeout(2000);
    expect(node.previews.length).toBe(sent);
  });

  test('answers Auto-frame with one whole, uncropped frame, watched or not', async ({ page }) => {
    const node = await openCamera(page);
    node.configure({ crop: { x: 0.25, y: 0.25, w: 0.5, h: 0.5 } });
    const asked = Date.now() / 1000;
    node.askFullFrame(4);
    await expect.poll(() => node.fullFrames.length, { timeout: 8_000 }).toBe(1);
    const [answer] = node.fullFrames;
    expect(answer!.request).toBe(4);
    expect(answer!.size).toEqual([FRAME.width, FRAME.height]); // under the 1920 px cap: as filmed
    expect(answer!.captureTs).toBeGreaterThanOrEqual(asked - 1);
    expect(node.previews).toHaveLength(0); // nobody watches, and a full frame is no preview
  });

  test('streams the configured crop when the config says stream, and stops when it says stop', async ({
    page
  }) => {
    const node = await openCamera(page);
    node.configure({ crop: { x: 0.25, y: 0.5, w: 0.5, h: 0.5 }, streaming: true });

    await expect.poll(() => node.frames.length, { timeout: 8_000 }).toBeGreaterThan(1);
    // Half of 1280x720, under the 960 px send cap: exactly the rectangle.
    expect(node.frames.at(-1)).toEqual([640, 360]);
    const header = node.headers.at(-1)!;
    expect(header).toMatchObject({ type: 'frame', board_id: node.boardId });
    expect(header).not.toHaveProperty('corners');
    expect(typeof header['sharpness']).toBe('number');
    await expect.poll(() => node.lastReport?.['ingest']).toBe('streaming');

    // The phone shows what is sent: the outline sits at the configured fractions.
    const outline = page.getByTestId('crop-outline');
    const picture = await page.locator('.picture').boundingBox();
    const box = await outline.boundingBox();
    expect(picture).not.toBeNull();
    expect(box).not.toBeNull();
    expect((box!.x - picture!.x) / picture!.width).toBeCloseTo(0.25, 1);
    expect((box!.y - picture!.y) / picture!.height).toBeCloseTo(0.5, 1);

    node.configure({ streaming: false });
    await expect.poll(() => node.lastReport?.['ingest']).toBe('idle');
    const sent = node.frames.length;
    await page.waitForTimeout(1500);
    expect(node.frames.length).toBe(sent);
  });
});

test.describe('the motion meter', () => {
  test.use({ viewport: LANDSCAPE });

  test('a hand crossing the board opens a burst; moving the crop alone does not', async ({ page }) => {
    const node = await openCamera(page);
    node.configure({ crop: { x: 0.2, y: 0.1, w: 0.6, h: 0.8 }, streaming: true });
    await expect.poll(() => node.headers.length, { timeout: 8_000 }).toBeGreaterThan(1);

    // A new crop is a new picture, not motion: the meter rebases instead of reading
    // the switch as one frame of whole-field change.
    node.configure({ crop: { x: 0.25, y: 0.15, w: 0.55, h: 0.75 } });
    await expect.poll(() => node.frames.at(-1), { timeout: 8_000 }).toEqual([704, 540]);
    await page.waitForTimeout(1500);
    expect(node.headers.filter((header) => header['burst_id'] !== null)).toEqual([]);

    await crossHand(page);
    await expect
      .poll(() => node.headers.find((header) => header['burst_id'] !== null), { timeout: 5_000 })
      .toMatchObject({ burst_id: `${node.boardId}:b0` });
    // The burst opened below 0.7, but the frame that opened it may not be among the
    // frames sent (the page drops frames while it encodes): wait for one that is.
    const lowest = () =>
      Math.min(
        ...node.headers
          .filter((header) => header['burst_id'] === `${node.boardId}:b0`)
          .map((header) => header['stability'] as number)
      );
    await expect.poll(lowest, { timeout: 5_000 }).toBeLessThan(0.7);
  });
});

test.describe('held upright', () => {
  test.use({ viewport: PORTRAIT });

  test('says so, and still reports and streams', async ({ page }) => {
    const node = await FakeControl.install(page);
    await drawnBoardCamera(page, { width: 720, height: 1280 });
    node.config = { ...node.config, streaming: true }; // set before the phone connects
    await page.goto('/camera');
    await liveViewfinder(page);

    await expect(page.getByTestId('camera-status')).toContainText('portrait');
    await expect.poll(() => node.frames.length, { timeout: 8_000 }).toBeGreaterThan(0);
    // Polled: a frame can reach the node before the report naming the frame size.
    await expect.poll(() => node.lastReport?.['frame']).toEqual({ width: 720, height: 1280 });
  });
});

test.describe('when the camera cannot be offered', () => {
  test.use({ viewport: PORTRAIT });

  test('an insecure context is explained, with both ways out', async ({ page }) => {
    // What a phone meets on `http://<lan-ip>:8080`: the camera is never offered, and
    // nothing is wrong with the permissions.
    await page.addInitScript(() => {
      Object.defineProperty(window, 'isSecureContext', { get: () => false });
    });
    await FakeControl.install(page);
    await page.goto('/camera');

    const blocked = page.locator('.blocked');
    await expect(blocked.locator('h2')).toHaveText('Camera blocked: insecure context');
    await expect(blocked).toContainText('make dev');
    // The adb recipe names the port this page was actually served on.
    await expect(blocked).toContainText('adb reverse tcp:4173 tcp:4173');
    await expect(page.locator('video')).toHaveCount(0);
  });

  test('a browser without getUserMedia is explained too', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'mediaDevices', { get: () => undefined });
    });
    await FakeControl.install(page);
    await page.goto('/camera');

    await expect(page.locator('.blocked h2')).toHaveText('Camera unsupported');
    await expect(page.locator('.blocked')).toContainText('getUserMedia');
  });
});
