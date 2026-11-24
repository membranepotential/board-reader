/**
 * The camera control page (`/control`) — the laptop side of the dumb-camera phone.
 *
 * What only a laid-out page can answer: that the preview and the crop rectangle are
 * drawn, that a pointer drag on the drawn picture becomes the right normalized rect in
 * the PUT body (the pixel -> fraction step lives in the component, not in the pure
 * `dragRect`), that the stream toggle round-trips, and that an upright phone is called
 * out. The rectangle maths and the parsing are Vitest's (`src/lib/control`).
 *
 * The node is a stateful fake here: it answers GET with its config and takes a PUT as
 * the new config, the way `camera_control.CameraRegistry` does.
 */

import { expect, test, type Page } from '@playwright/test';
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
}

interface FrameSize {
  width: number;
  height: number;
}

class FakeControlNode {
  config: Config = { crop: { x: 0.2, y: 0.2, w: 0.5, h: 0.6 }, device: null, resolution: '1080p', streaming: false };
  readonly puts: Config[] = [];

  constructor(readonly frame: FrameSize) {}

  view(): Record<string, unknown> {
    return {
      board_id: 'phone',
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
      seen_at: 99.6,
      preview: { seq: 1, at: 99, width: 640, height: Math.round((640 * this.frame.height) / this.frame.width) },
      frame: { seq: 12, received_at: 99.5, width: 960, height: 648 },
      now: 100
    };
  }

  static async install(page: Page, frame: FrameSize = { width: 1920, height: 1080 }): Promise<FakeControlNode> {
    const node = new FakeControlNode(frame);
    const png = checkerboardPng(320, 180);
    await page.route(/\/api\/boards\/phone\/camera$/, async (route) => {
      if (route.request().method() === 'PUT') {
        node.config = route.request().postDataJSON() as Config;
        node.puts.push(node.config);
      }
      await route.fulfill({ json: node.view() });
    });
    await page.route(/\/api\/boards\/phone\/(camera\/preview|frame)\b/, (route) =>
      route.fulfill({ body: png, contentType: 'image/png' })
    );
    await page.route(/\/api\/healthz$/, (route) =>
      route.fulfill({
        json: {
          status: 'ok',
          ingest: { received: 12, refused: 0 },
          recognizer: { name: 'chessq', model: 'models/chessq.onnx' }
        }
      })
    );
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
  // Nothing goes to the node before Save: a poll must not race the gesture.
  expect(node.puts).toHaveLength(0);
  await page.getByRole('button', { name: 'Save crop' }).click();

  await expect.poll(() => node.puts.length).toBe(1);
  const crop = node.puts[0]?.crop;
  if (crop == null) throw new Error('saved without a crop');
  expect(crop.x).toBeCloseTo(0.3, 2);
  expect(crop.y).toBeCloseTo(0.25, 2);
  expect(crop.w).toBeCloseTo(0.4, 2);
  expect(crop.h).toBeCloseTo(0.6, 2);
  // The rest of the config rides along unchanged: a PUT replaces the whole object.
  expect(node.puts[0]).toMatchObject({ device: null, resolution: '1080p', streaming: false });
  await expect(page.getByText('unsaved')).toBeHidden();
});

test('the stream toggle starts and stops the phone', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');

  await page.getByRole('button', { name: 'Start streaming' }).click();
  await expect(page.getByRole('button', { name: 'Stop streaming' })).toBeVisible();
  expect(node.puts.at(-1)?.streaming).toBe(true);

  await page.getByRole('button', { name: 'Stop streaming' }).click();
  await expect(page.getByRole('button', { name: 'Start streaming' })).toBeVisible();
  expect(node.puts.at(-1)?.streaming).toBe(false);
  // The crop is not touched by a toggle.
  expect(node.puts.at(-1)?.crop).toEqual({ x: 0.2, y: 0.2, w: 0.5, h: 0.6 });
});

test('an upright phone is called out', async ({ page }) => {
  await FakeControlNode.install(page, { width: 1080, height: 1920 });
  await page.goto('/control');

  await expect(page.locator('[data-warning="portrait-phone"]')).toBeVisible();
  await expect(page.getByText('(portrait)')).toBeVisible();
});
