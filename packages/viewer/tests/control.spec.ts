/**
 * The camera control page (`/control`) — the laptop side of the dumb-camera phone.
 *
 * What only a laid-out page can answer: that the pushed preview and the crop
 * rectangle are drawn, that a pointer drag on the drawn picture becomes the right
 * normalized rect in the `configure` message (the pixel -> fraction step lives in
 * the component, not in the pure `dragRect`), that the stream toggle round-trips,
 * and that an upright phone is called out. The rectangle maths and the parsing are
 * Vitest's (`src/lib/control`).
 *
 * The node is a stateful fake on the control socket: it pushes its state, a
 * preview, a received frame and a health tick on connect, and takes a `configure`
 * as the new config and pushes the new state back, the way
 * `control.camera.CameraRegistry` does. Its camera list names one phone, on
 * `board-1`; two phones against one node, and renaming, are `cameras.spec.ts`'s.
 */

import { expect, test, type Page, type WebSocketRoute } from '@playwright/test';
import { checkerboardPng } from './helpers/png';

interface Crop {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Config {
  crop: Crop | null;
  device: string | null;
  resolution: '720p' | '1080p' | '2160p';
  streaming: boolean;
  rules: 'standard' | 'free';
  auto: boolean;
}

type AutoFrameStatus = 'idle' | 'locating' | 'framed' | 'no-board' | 'manual';

interface FrameSize {
  width: number;
  height: number;
}

interface Direction {
  degrees: number;
  white_side: 'bottom' | 'left' | 'top' | 'right';
  camera_moved: boolean;
}

const SETTLED: Direction = { degrees: 89, white_side: 'left', camera_moved: false };

class FakeControlNode {
  config: Config = {
    crop: { x: 0.2, y: 0.2, w: 0.5, h: 0.6 },
    device: null,
    resolution: '1080p',
    streaming: false,
    rules: 'standard',
    auto: true
  };
  autoframe: AutoFrameStatus = 'framed';
  readonly configures: Config[] = [];
  socket: WebSocketRoute | null = null;

  constructor(readonly frame: FrameSize) {}

  state(): string {
    return JSON.stringify({
      type: 'state',
      config: this.config,
      report: {
        devices: [
          { id: 'a', label: 'Back Camera' },
          { id: 'b', label: 'Back Ultra Wide Camera' }
        ],
        device: 'Back Camera',
        frame: this.frame,
        ingest: this.config.streaming ? 'streaming' : 'idle'
      },
      phone_connected: true,
      autoframe: this.autoframe
    });
  }

  /** Auto-frame's status changed on the node: it pushes the state. */
  setAutoframe(status: AutoFrameStatus): void {
    this.autoframe = status;
    this.socket?.send(this.state());
  }

  static async install(
    page: Page,
    frame: FrameSize = { width: 1920, height: 1080 },
    direction: Direction | null = SETTLED
  ): Promise<FakeControlNode> {
    const node = new FakeControlNode(frame);
    // A PNG under a JPEG data URL still draws: browsers sniff the bytes.
    const picture = checkerboardPng(320, 180).toString('base64');
    await page.routeWebSocket(/\/ws\/cameras$/, (ws) => {
      const camera = {
        device_id: 'kitchen-phone-id',
        board_id: 'board-1',
        name: null,
        connected: true,
        report: null,
        frame_age_s: 0.5
      };
      ws.send(JSON.stringify({ type: 'cameras', cameras: [camera] }));
    });
    await page.routeWebSocket('**/ws/boards/board-1/control', (ws) => {
      node.socket = ws;
      ws.onMessage((message) => {
        const parsed = JSON.parse(message as string) as { type: string; config: Config };
        if (parsed.type !== 'configure') return;
        node.config = parsed.config;
        node.configures.push(parsed.config);
        ws.send(node.state());
      });
      ws.send(node.state());
      ws.send(JSON.stringify({ type: 'preview', width: 640, height: Math.round((640 * frame.height) / frame.width), jpeg: picture }));
      ws.send(JSON.stringify({ type: 'frame', seq: 12, width: 960, height: 648, jpeg: picture }));
      ws.send(
        JSON.stringify({
          type: 'health',
          recognizer: 'chessq',
          fallback: null,
          received: 12,
          refused: 0,
          frame_age_s: 0.5,
          fps: 4.2,
          direction
        })
      );
    });
    return node;
  }
}

test('the preview, the rectangle and the received frame are drawn', async ({ page }) => {
  await FakeControlNode.install(page);
  await page.goto('/control');

  await expect(page.locator('.chip')).toHaveText('phone connected');
  await expect(page.locator('.picture img')).toHaveJSProperty('complete', true);
  await expect(page.locator('img.received')).toBeVisible();
  await expect(page.getByText('chessq')).toBeVisible();
  await expect(page.getByText('4.2')).toBeVisible(); // frames/s, measured on the node

  // The rectangle sits where the config puts it, in fractions of the drawn picture.
  const picture = await page.locator('.picture').boundingBox();
  const rect = await page.locator('.rect').boundingBox();
  if (picture === null || rect === null) throw new Error('nothing laid out');
  expect((rect.x - picture.x) / picture.width).toBeCloseTo(0.2, 1);
  expect(rect.width / picture.width).toBeCloseTo(0.5, 1);
});

test('dragging and resizing the rectangle saves the normalized rect', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  const box = await page.locator('.picture').boundingBox();
  if (box === null) throw new Error('no picture');

  // Move the body right by 10 % and down by 5 % of the frame.
  const body = await page.locator('.rect .body').boundingBox();
  if (body === null) throw new Error('no rectangle');
  const start = { x: body.x + body.width / 2, y: body.y + body.height / 2 };
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(start.x + box.width * 0.05, start.y, { steps: 3 });
  await page.mouse.move(start.x + box.width * 0.1, start.y + box.height * 0.05, { steps: 3 });
  await page.mouse.up();

  // Then pull the bottom-right corner in by 10 % of the width.
  const corner = await page.locator('.corner.se').boundingBox();
  if (corner === null) throw new Error('no corner');
  const grip = { x: corner.x + corner.width / 2, y: corner.y + corner.height / 2 };
  await page.mouse.move(grip.x, grip.y);
  await page.mouse.down();
  await page.mouse.move(grip.x - box.width * 0.1, grip.y, { steps: 4 });
  await page.mouse.up();

  await expect(page.getByText('unsaved')).toBeVisible();
  // Nothing goes to the node before Save: a pushed state must not race the gesture.
  expect(node.configures).toHaveLength(0);
  await page.getByRole('button', { name: 'Save crop' }).click();

  await expect.poll(() => node.configures.length).toBe(1);
  const crop = node.configures[0]?.crop;
  if (crop == null) throw new Error('saved without a crop');
  expect(crop.x).toBeCloseTo(0.3, 2);
  expect(crop.y).toBeCloseTo(0.25, 2);
  expect(crop.w).toBeCloseTo(0.4, 2);
  expect(crop.h).toBeCloseTo(0.6, 2);
  // The rest of the config rides along unchanged: a configure replaces the whole object.
  expect(node.configures[0]).toMatchObject({ device: null, resolution: '1080p', streaming: false });
  await expect(page.getByText('unsaved')).toBeHidden();
});

test('the stream toggle starts and stops the phone', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');

  await page.getByRole('button', { name: 'Start streaming' }).click();
  await expect(page.getByRole('button', { name: 'Stop streaming' })).toBeVisible();
  await expect.poll(() => node.configures.at(-1)?.streaming).toBe(true);

  await page.getByRole('button', { name: 'Stop streaming' }).click();
  await expect(page.getByRole('button', { name: 'Start streaming' })).toBeVisible();
  await expect.poll(() => node.configures.at(-1)?.streaming).toBe(false);
  // The crop is not touched by a toggle.
  expect(node.configures.at(-1)?.crop).toEqual({ x: 0.2, y: 0.2, w: 0.5, h: 0.6 });
});

test('the rules select switches the board to free and back', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');

  const rules = page.getByLabel('rules');
  await expect(rules).toHaveValue('standard');
  await rules.selectOption('free');
  await expect.poll(() => node.configures.at(-1)?.rules).toBe('free');
  // A configure is the whole config: the stream and the crop ride along unchanged.
  expect(node.configures.at(-1)).toMatchObject({ streaming: false, resolution: '1080p' });

  await rules.selectOption('standard');
  await expect.poll(() => node.configures.at(-1)?.rules).toBe('standard');
});

test('an upright phone is called out', async ({ page }) => {
  await FakeControlNode.install(page, { width: 1080, height: 1920 });
  await page.goto('/control');

  await expect(page.locator('[data-warning="portrait-phone"]')).toBeVisible();
  await expect(page.getByText('(portrait)')).toBeVisible();
});

test('the side White sits on is shown, and a moved camera is called out', async ({ page }) => {
  await FakeControlNode.install(page);
  await page.goto('/control');
  await expect(page.getByText('White sits on the left')).toBeVisible();
  await expect(page.locator('[data-warning="camera-moved"]')).toHaveCount(0);

  await FakeControlNode.install(page, undefined, { degrees: 1, white_side: 'bottom', camera_moved: true });
  await page.reload();
  await expect(page.getByText('White sits at the bottom')).toBeVisible();
  await expect(page.locator('[data-warning="camera-moved"]')).toContainText('The camera moved');
});

test('Auto-frame says what it is doing for the camera', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  const status = page.getByTestId('autoframe');
  await expect(status).toHaveText('on: the node set the crop');
  node.setAutoframe('no-board');
  await expect(status).toHaveText('no board found yet; trying again');
  node.setAutoframe('manual');
  await expect(status).toHaveText('off: the crop was set by hand');
});

test('the QR code that opens /camera on a phone sits next to the camera list', async ({ page }) => {
  await FakeControlNode.install(page);
  await page.route('**/api/camera-links', (route) =>
    route.fulfill({ json: { links: [{ url: 'https://192.168.1.7:8443/camera', via: 'lan' }] } })
  );
  await page.goto('/control');
  const add = page.getByRole('complementary').filter({ hasText: 'Add a camera' });
  await expect(add.locator('svg path')).toHaveCount(1);
  await expect(add).toContainText('/camera');
});
