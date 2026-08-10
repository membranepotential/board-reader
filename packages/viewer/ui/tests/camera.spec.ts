/**
 * The `/camera` page — the DESIGN §4.1 "virtual Pi" in a browser.
 *
 * Everything here is a property of the page as a phone renders it, which is why it is
 * an e2e spec and not a unit test:
 *
 *  - **Both orientations.** iOS Safari implements no `screen.orientation.lock()`, so
 *    "landscape by default" can only be the base stylesheet with
 *    `@media (orientation: portrait)` as the override. Both have to be exercised, or
 *    half the layout ships untested — and it is the half a phone lands in first.
 *  - **Real touch targets.** A 31 px rail button shipped here once. A stylesheet
 *    assertion cannot catch that; a measured `boundingBox` can.
 *  - **An empty board, not the start position.** The panel must not imply a reading it
 *    has not received.
 *  - **The blocked explainer.** Without a secure context the camera is never offered at
 *    all, and the page has to say so rather than sit blank.
 */

import { expect, test, type Locator, type Page } from '@playwright/test';
import { FakeViewerSocket, positionEvent } from './helpers/node';

const LANDSCAPE = { width: 844, height: 390 };
const PORTRAIT = { width: 390, height: 844 };

/** `--rail` / `--dock` in Viewfinder.svelte: the chrome's own thickness. */
const CHROME_PX = 150;

/** The rail's six controls, in DOM order — one order serves both orientations. */
const RAIL_CONTROLS = ['more', 'position', 'detect board', 'rotate labels', 'calibrate', 'start'];

/**
 * A finger is about 9 mm across; 44 px is the smallest target Apple and Google both
 * consider reachable. The stylesheet asks for 48 — this asserts the floor, so a layout
 * that squeezes a button below it fails here rather than on a phone at a board.
 */
const MIN_TOUCH_PX = 44;

async function openCamera(page: Page): Promise<FakeViewerSocket> {
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/camera');
  return socket;
}

async function boxOf(locator: Locator): Promise<{ width: number; height: number }> {
  const box = await locator.boundingBox();
  expect(box, 'element has no layout box').not.toBeNull();
  return { width: (box as { width: number }).width, height: (box as { height: number }).height };
}

/**
 * Wait for the fake camera to actually deliver a frame.
 *
 * `.picture` is sized from the video's own dimensions, so it has no box at all until
 * `loadedmetadata` has fired — measuring before that reads 0 and proves nothing.
 */
async function liveViewfinder(page: Page): Promise<void> {
  await expect
    .poll(async () => (await page.locator('.picture').boundingBox())?.width ?? 0)
    .toBeGreaterThan(0);
}

test.describe('the viewfinder', () => {
  test.use({ viewport: LANDSCAPE });

  test('renders landscape with the rail as a right-hand column', async ({ page }) => {
    await openCamera(page);

    const stage = page.locator('.camera-stage');
    await expect(stage).toBeVisible();
    await expect(page.locator('.blocked')).toHaveCount(0);

    const rail = await boxOf(page.locator('nav.rail'));
    expect(rail.width).toBeCloseTo(CHROME_PX, 0);
    expect(rail.height).toBeCloseTo(LANDSCAPE.height, 0);

    // The camera really opened: the picture is sized to the fake device's frame and
    // inset from the rail, never running underneath it.
    await liveViewfinder(page);
    const picture = await boxOf(page.locator('.picture'));
    expect(picture.width).toBeLessThanOrEqual(LANDSCAPE.width - CHROME_PX + 1);
  });

  test('renders portrait with the rail as a bottom dock', async ({ page }) => {
    await openCamera(page);
    await liveViewfinder(page);

    await page.setViewportSize(PORTRAIT);

    const rail = await boxOf(page.locator('nav.rail'));
    expect(rail.width).toBeCloseTo(PORTRAIT.width, 0);
    expect(rail.height).toBeCloseTo(CHROME_PX, 0);

    // The picture re-fits into the new inner box rather than staying under the dock.
    const picture = await boxOf(page.locator('.picture'));
    expect(picture.height).toBeGreaterThan(0);
    expect(picture.height).toBeLessThanOrEqual(PORTRAIT.height - CHROME_PX + 1);
  });

  for (const [name, viewport] of [
    ['landscape', LANDSCAPE],
    ['portrait', PORTRAIT]
  ] as const) {
    test(`all six rail controls are present and thumb-sized in ${name}`, async ({ page }) => {
      await openCamera(page);
      await page.setViewportSize(viewport);

      const buttons = page.locator('nav.rail button');
      await expect(buttons).toHaveCount(RAIL_CONTROLS.length);
      await expect(buttons).toHaveText(RAIL_CONTROLS.map((label) => new RegExp(`^${label}`)));

      for (let i = 0; i < RAIL_CONTROLS.length; i++) {
        const box = await boxOf(buttons.nth(i));
        expect(box.height, `${RAIL_CONTROLS[i]} height`).toBeGreaterThanOrEqual(MIN_TOUCH_PX);
        expect(box.width, `${RAIL_CONTROLS[i]} width`).toBeGreaterThanOrEqual(MIN_TOUCH_PX);
      }
    });
  }
});

test.describe('the position panel', () => {
  test.use({ viewport: LANDSCAPE });

  test('shows an EMPTY board until the first event, then the reading', async ({ page }) => {
    const socket = await openCamera(page);

    const panel = page.locator('.pos-panel');
    await expect(panel).toBeVisible();
    // "waiting…", not a position: the page has been handed nothing yet.
    await expect(panel.locator('.st')).toHaveText('waiting…');
    // The board is really there — it just has nothing on it. Showing the start
    // position here would assert a reading that never arrived.
    await expect(panel.locator('cg-board')).toBeVisible();
    await expect(panel.locator('piece')).toHaveCount(0);

    // The board id defaults to `phone`; an event for it fills the same board in place.
    socket.send(positionEvent({ board_id: 'phone', seq: 4, confidence: 0.81 }));
    await expect(panel.locator('piece')).toHaveCount(32);
    await expect(panel.locator('.st')).toHaveText('ok · seq 4 · confidence 0.81');
  });

  test('the "position" control hides and shows it', async ({ page }) => {
    await openCamera(page);
    const panel = page.locator('.pos-panel');
    await expect(panel).toBeVisible();

    await page.locator('nav.rail button', { hasText: 'position' }).click();
    await expect(panel).toBeHidden();
    // Hidden with `visibility`, never `display: none` — a 0x0 box makes chessground's
    // own ResizeObserver compute NaN and the board never recovers.
    expect(await panel.evaluate((el) => getComputedStyle(el).display)).not.toBe('none');
    await expect(panel.locator('cg-board')).toHaveCount(1);
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
    await openCamera(page);

    const blocked = page.locator('.blocked');
    await expect(blocked.locator('h2')).toHaveText('Camera blocked: insecure context');
    await expect(blocked).toContainText('board-reader-viewer --host 0.0.0.0 --ssl');
    // The adb recipe names the port this page was actually served on.
    await expect(blocked).toContainText('adb reverse tcp:4173 tcp:4173');

    // With no camera there is no viewfinder at all — no dead stage, no dead rail.
    await expect(page.locator('.camera-stage')).toHaveCount(0);
    await expect(page.locator('nav.rail')).toHaveCount(0);
  });

  test('a browser without getUserMedia is explained too', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'mediaDevices', { get: () => undefined });
    });
    await openCamera(page);

    await expect(page.locator('.blocked h2')).toHaveText('Camera unsupported');
    await expect(page.locator('.blocked')).toContainText('getUserMedia');
  });
});
