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
 * `control.camera.registry.CameraRegistry` does, including its rule that a crop
 * changed through `configure` is set by hand and switches Auto-frame off. Its
 * camera list names one phone, on `board-1`, so the page shows one card; two
 * phones against one node, and renaming, are `cameras.spec.ts`'s.
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

type AutoFrameStatus = 'idle' | 'locating' | 'framed' | 'not-visible' | 'no-board' | 'manual';

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
  watching = true;
  kind: 'phone' | 'stream' = 'phone';
  readonly configures: Config[] = [];
  reframes = 0;
  socket: WebSocketRoute | null = null;
  /** Hold the state back on connect until `sendState()`: the page has no config yet. */
  holdState = false;

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
        ingest: this.config.streaming ? 'streaming' : 'idle',
        kind: this.kind
      },
      phone_connected: true,
      autoframe: this.autoframe,
      watching: this.watching
    });
  }

  /** Auto-frame's status changed on the node: it pushes the state. */
  setAutoframe(status: AutoFrameStatus): void {
    this.autoframe = status;
    this.sendState();
  }

  /** Auto-frame found the board and set the crop itself (not through `configure`). */
  autoCrop(crop: Crop): void {
    this.config = { ...this.config, crop };
    this.setAutoframe('framed');
  }

  sendState(): void {
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
        if (parsed.type === 'reframe') node.reframes += 1;
        if (parsed.type !== 'configure') return;
        // The node's rule: a crop that differs from its own was drawn by hand.
        const byHand = JSON.stringify(parsed.config.crop) !== JSON.stringify(node.config.crop);
        node.config = byHand ? { ...parsed.config, auto: false } : parsed.config;
        if (byHand) node.autoframe = 'manual';
        node.configures.push(parsed.config);
        ws.send(node.state());
      });
      if (!node.holdState) ws.send(node.state());
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

const card = (page: Page) => page.getByTestId('camera-card');

/** Switch the card to Manual and press Crop: the crop edit mode. */
async function enterCrop(page: Page): Promise<void> {
  await page.getByRole('radio', { name: 'Manual' }).check();
  await page.getByRole('button', { name: 'Crop', exact: true }).click();
}

/** Where the cropped preview's image sits, in fractions of its box. */
async function previewPlacement(page: Page): Promise<{ ar: number; scale: number; left: number }> {
  const box = await page.getByTestId('cropped-preview').boundingBox();
  const img = await page.locator('img.preview-crop').boundingBox();
  if (box === null || img === null) throw new Error('no cropped preview');
  return { ar: box.width / box.height, scale: img.width / box.width, left: (img.x - box.x) / box.width };
}

/** Where the crop rectangle sits, in fractions of the drawn picture. */
async function fractions(page: Page): Promise<{ x: number; w: number }> {
  const picture = await page.locator('.picture').boundingBox();
  const rect = await page.locator('.rect').boundingBox();
  if (picture === null || rect === null) throw new Error('nothing laid out');
  return { x: (rect.x - picture.x) / picture.width, w: rect.width / picture.width };
}

test('while streaming, the card shows what the node receives, and its stats', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  node.config = { ...node.config, streaming: true };
  await page.goto('/control');

  await expect(card(page)).toHaveCount(1);
  await expect(card(page).locator('h2')).toContainText('board-1');
  await expect(card(page).locator('h2')).toContainText('connected');
  // The cropped frame, sized to its own aspect ratio (960×648), is the card's picture.
  const received = card(page).locator('img.received');
  await expect(received).toHaveJSProperty('complete', true);
  const box = await card(page).locator('.picture').boundingBox();
  if (box === null) throw new Error('no picture');
  expect(box.width / box.height).toBeCloseTo(960 / 648, 1);
  // No crop handles outside crop mode, and no second "what the node receives" image.
  await expect(page.locator('.rect')).toHaveCount(0);
  await expect(page.getByText('What the node receives')).toHaveCount(0);
  await expect(page.locator('img')).toHaveCount(1);
  await expect(card(page).getByText('chessq')).toBeVisible();
  await expect(card(page).getByText('4.2')).toBeVisible(); // frames/s, measured on the node
});

test('while not streaming, the card crops the latest preview to the config, and follows a new crop', async ({
  page
}) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  // The 640×360 preview, cropped to x 0.2 w 0.5 (and y 0.2 h 0.6): the box takes the
  // crop's shape in pixels, and the whole preview is twice its width, shifted left.
  await expect(page.locator('img.preview-crop')).toHaveJSProperty('complete', true);
  await expect(page.locator('img.received')).toHaveCount(0);
  let shown = await previewPlacement(page);
  expect(shown.ar).toBeCloseTo((0.5 * 640) / (0.6 * 360), 1);
  expect(shown.scale).toBeCloseTo(1 / 0.5, 1);
  expect(shown.left).toBeCloseTo(-0.2 / 0.5, 1);

  // Auto-frame sets a new crop: the view follows at once, without a new frame.
  node.autoCrop({ x: 0.1, y: 0.15, w: 0.6, h: 0.7 });
  await expect.poll(async () => (await previewPlacement(page)).scale).toBeCloseTo(1 / 0.6, 1);
  shown = await previewPlacement(page);
  expect(shown.left).toBeCloseTo(-0.1 / 0.6, 1);
  expect(shown.ar).toBeCloseTo((0.6 * 640) / (0.7 * 360), 1);

  // Streaming: the real received frame again.
  await page.getByRole('button', { name: 'Start streaming' }).click();
  await expect(page.locator('img.received')).toBeVisible();
  await expect(page.locator('img.preview-crop')).toHaveCount(0);
});

test('Crop, in Manual, shows the whole frame with the rectangle, and Cancel goes back', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  // No Crop in Auto: Auto chooses the crop.
  await expect(page.getByRole('button', { name: 'Crop', exact: true })).toHaveCount(0);
  await enterCrop(page);
  await expect.poll(() => node.configures.at(-1)?.auto).toBe(false);

  await expect(page.locator('img.preview-crop')).toHaveCount(0);
  await expect(page.locator('.picture img')).toHaveJSProperty('complete', true);
  // The whole frame (1920×1080) with the rectangle where the config puts it.
  const box = await page.locator('.picture').boundingBox();
  if (box === null) throw new Error('no picture');
  expect(box.width / box.height).toBeCloseTo(16 / 9, 1);
  const { x, w } = await fractions(page);
  expect(x).toBeCloseTo(0.2, 1);
  expect(w).toBeCloseTo(0.5, 1);

  await page.getByRole('button', { name: 'Cancel' }).click();
  await expect(page.locator('img.preview-crop')).toBeVisible();
  // Only the switch to Manual went out; Cancel sent no crop.
  expect(node.configures).toHaveLength(1);
});

test('dragging and resizing the rectangle saves the normalized rect', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  await enterCrop(page);
  await expect.poll(() => node.configures.length).toBe(1);
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
  expect(node.configures).toHaveLength(1);
  await page.getByRole('button', { name: 'Save crop' }).click();

  await expect.poll(() => node.configures.length).toBe(2);
  const crop = node.configures[1]?.crop;
  if (crop == null) throw new Error('saved without a crop');
  expect(crop.x).toBeCloseTo(0.3, 2);
  expect(crop.y).toBeCloseTo(0.25, 2);
  expect(crop.w).toBeCloseTo(0.4, 2);
  expect(crop.h).toBeCloseTo(0.6, 2);
  // The rest of the config rides along unchanged: a configure replaces the whole object.
  expect(node.configures[1]).toMatchObject({ device: null, resolution: '1080p', streaming: false, auto: false });
  // Saved: back to the cropped view, still Manual.
  await expect(page.locator('img.preview-crop')).toBeVisible();
  await expect(page.getByRole('radio', { name: 'Manual' })).toBeChecked();
  await expect(page.getByText('the crop was set by hand')).toBeVisible();
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

test('toggling the stream keeps the crop Auto-frame chose, so Auto-frame stays on', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  node.holdState = true;
  await page.goto('/control');
  // Before the node's first state the page knows no config: nothing can be sent.
  const start = page.getByRole('button', { name: 'Start streaming' });
  await expect(start).toBeDisabled();
  await expect(page.getByLabel('lens')).toBeDisabled();
  node.sendState();
  await expect(start).toBeEnabled();

  // Auto-frame re-frames the board on the node, then the operator starts the stream.
  const framed = { x: 0.1, y: 0.15, w: 0.6, h: 0.7 };
  node.autoCrop(framed);
  await expect(page.getByTestId('autoframe')).toHaveText('on: the node set the crop');
  await start.click();
  await expect.poll(() => node.configures.at(-1)?.streaming).toBe(true);
  expect(node.configures.at(-1)?.crop).toEqual(framed);
  expect(node.config.auto).toBe(true);
  await expect(page.getByRole('radio', { name: 'Auto' })).toBeChecked();

  await page.getByLabel('lens').selectOption('Back Ultra Wide Camera');
  await expect.poll(() => node.configures.at(-1)?.device).toBe('Back Ultra Wide Camera');
  expect(node.config.auto).toBe(true);
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

test('the Auto | Manual toggle reaches the node; Re-detect frame shows in Auto, Crop in Manual', async ({
  page
}) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  const auto = page.getByRole('radio', { name: 'Auto' });
  const manual = page.getByRole('radio', { name: 'Manual' });
  const redetect = page.getByRole('button', { name: 'Re-detect frame' });
  const crop = page.getByRole('button', { name: 'Crop', exact: true });
  await expect(auto).toBeChecked();
  await expect(page.getByTestId('drift-off')).toBeHidden();
  await redetect.click();
  await expect.poll(() => node.reframes).toBe(1);
  await expect(crop).toHaveCount(0);

  await manual.check();
  await expect.poll(() => node.configures.at(-1)?.auto).toBe(false);
  await expect(redetect).toHaveCount(0);
  await crop.click();
  await expect(page.locator('.rect')).toBeVisible();

  // Back to Auto: the crop being edited is dropped, and the node frames afresh.
  await auto.check();
  await expect.poll(() => node.configures.at(-1)?.auto).toBe(true);
  await expect(page.locator('.rect')).toHaveCount(0);
  await expect(crop).toHaveCount(0);
  await expect(redetect).toBeVisible();
  // Switching Auto on sends the crop the node last pushed, not one drawn here.
  expect(node.configures.at(-1)?.crop).toEqual({ x: 0.2, y: 0.2, w: 0.5, h: 0.6 });
});

test('a board too small to watch on the previews says a move goes unnoticed', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  await expect(page.getByTestId('autoframe')).toHaveText('on: the node set the crop');
  node.watching = false;
  node.setAutoframe('framed');
  await expect(page.getByTestId('drift-off')).toBeVisible();
});

test('a stream offers no lens and no resolution to choose', async ({ page }) => {
  const node = await FakeControlNode.install(page);
  await page.goto('/control');
  await expect(page.getByLabel('lens')).toBeVisible();
  await expect(page.getByLabel('resolution')).toBeVisible();
  node.kind = 'stream';
  node.setAutoframe('framed'); // any push of the state
  await expect(page.getByLabel('lens')).toBeHidden();
  await expect(page.getByLabel('resolution')).toBeHidden();
  await expect(page.getByLabel('rules')).toBeVisible();
});

test('Add camera opens the QR code that opens /camera on a phone', async ({ page }) => {
  await FakeControlNode.install(page);
  await page.route('**/api/camera-links', (route) =>
    route.fulfill({ json: { links: [{ url: 'https://192.168.1.7:8443/camera', via: 'lan' }] } })
  );
  await page.goto('/control');
  const dialog = page.getByRole('dialog', { name: 'Add a camera' });
  await expect(dialog).toBeHidden();
  await page.getByRole('button', { name: 'Add camera' }).click();
  await expect(dialog.locator('svg path')).toHaveCount(1);
  await expect(dialog).toContainText('/camera');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
});

test('Add stream sends the URL and the start, and closes once the node runs it', async ({ page }) => {
  await FakeControlNode.install(page);
  const asked: Record<string, unknown>[] = [];
  let refuse = true;
  await page.routeWebSocket(/\/ws\/streams$/, (ws) => {
    ws.send(JSON.stringify({ type: 'streams', streams: [] }));
    ws.onMessage((message) => {
      const parsed = JSON.parse(message as string) as Record<string, unknown>;
      asked.push(parsed);
      if (refuse) {
        refuse = false;
        ws.send(JSON.stringify({ type: 'refused', reason: 'yt-dlp could not open it' }));
        return;
      }
      const stream = {
        id: 'stream-1',
        url: parsed['url'],
        start_s: parsed['start_s'],
        state: 'starting',
        device_id: 'stream-device-one',
        board_id: null,
        error: null
      };
      ws.send(JSON.stringify({ type: 'streams', streams: [stream] }));
    });
  });
  await page.goto('/control');
  await page.getByRole('button', { name: 'Add stream' }).click();
  const dialog = page.getByRole('dialog', { name: 'Add a video or live stream' });
  await dialog.getByLabel('video or live stream URL').fill('https://www.youtube.com/watch?v=chessgame');
  await dialog.getByLabel('start at').fill('1:61');
  await dialog.getByRole('button', { name: 'Add stream' }).click();
  await expect(dialog.getByRole('alert')).toContainText('minutes and seconds go up to 59');
  expect(asked).toHaveLength(0);

  await dialog.getByLabel('start at').fill('1:30');
  await dialog.getByRole('button', { name: 'Add stream' }).click();
  await expect(dialog.getByRole('alert')).toContainText('the node refused: yt-dlp could not open it');
  await dialog.getByRole('button', { name: 'Add stream' }).click();
  await expect(dialog).toBeHidden();
  expect(asked.at(-1)).toEqual({ type: 'add_stream', url: 'https://www.youtube.com/watch?v=chessgame', start_s: 90 });
});
