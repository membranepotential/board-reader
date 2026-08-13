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
 *  - **A full-bleed picture.** The controls float ON the picture now, so "the picture
 *    fills the stage" is a measurement rather than a promise — and the arithmetic
 *    that used to inset it is exactly what regressed the day it was written.
 *  - **An empty board, not the start position.** The panel must not imply a reading it
 *    has not received.
 *  - **The blocked explainer.** Without a secure context the camera is never offered at
 *    all, and the page has to say so rather than sit blank.
 *
 * Geometry is measured, never asserted against a class name: every number below comes
 * from a `boundingBox`, so a stylesheet that says the right thing and lays out the
 * wrong thing still fails.
 */

import { expect, test, type Locator, type Page } from '@playwright/test';
import { FakeViewerSocket, positionEvent, stubJson } from './helpers/node';

const LANDSCAPE = { width: 844, height: 390 };
const PORTRAIT = { width: 390, height: 844 };

/**
 * The four controls that are always on the viewfinder, in DOM order — one order
 * serves both orientations. `detect board` and `rotate labels` live in the sheet;
 * `rotate labels` reappears here only when a re-detection may have moved the labels.
 */
const CONTROLS = ['more', 'position', 'calibrate', 'start'];

/** These two moved into the sheet, and must not be back out on the picture. */
const SHEET_ONLY = ['detect board', 'rotate labels'];

/**
 * A finger is about 9 mm across; 44 px is the smallest target Apple and Google both
 * consider reachable. The stylesheet asks for 48 — this asserts the floor, so a layout
 * that squeezes a button below it fails here rather than on a phone at a board.
 */
const MIN_TOUCH_PX = 44;

interface Box {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

async function openCamera(page: Page): Promise<FakeViewerSocket> {
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/camera');
  return socket;
}

// ---- a camera whose scene this file decides -----------------------------------
// Chromium's `--use-fake-device-for-media-stream` gives one rolling pattern and no
// way to say what is in it. The corner-pairing test needs a board that moves and
// blurs on cue, so it swaps `getUserMedia` for a canvas `captureStream` it draws.

const FRAME = { width: 1280, height: 720 };
/** Where the board is drawn, and what `/detect` is stubbed to return. */
const BOARD = { x: 240, y: 150, w: 800, h: 448 };
/** How far it slides, in frame pixels — enough to be unmistakable, small enough to track. */
const SHIFT = 40;

interface SceneWindow {
  __scene: (change: { degrade?: boolean; shiftTo?: number }) => void;
}

async function drawnBoardCamera(page: Page): Promise<void> {
  await page.addInitScript(
    ({ FRAME, BOARD }) => {
      const canvas = document.createElement('canvas');
      canvas.width = FRAME.width;
      canvas.height = FRAME.height;
      const ctx = canvas.getContext('2d');
      if (ctx === null) return;

      let shift = 0;
      let target = 0;
      let degrade = 0;

      // A bare checkerboard is periodic, and a translation of half a square is then
      // indistinguishable from one of one and a half. These give the homography fit
      // something unambiguous. Deterministic, so a failure is reproducible.
      let seed = 7;
      const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
      const marks = Array.from({ length: 24 }, () => ({ x: rnd(), y: rnd(), r: 4 + rnd() * 10 }));

      const draw = () => {
        ctx.save();
        ctx.globalAlpha = 1;
        // Mid grey is exactly the mean of the two square colours, so fading the board
        // toward it changes CONTRAST while leaving the mean intensity alone — and the
        // mean is what the sharpness score divides by. Fading toward black would
        // cancel most of the drop and leave the candidates nearly tied.
        ctx.fillStyle = 'rgb(128,128,128)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1 - 0.7 * degrade;

        const bx = BOARD.x + shift;
        const cw = BOARD.w / 8;
        const ch = BOARD.h / 8;
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            ctx.fillStyle = (r + c) % 2 === 0 ? 'rgb(51,51,51)' : 'rgb(204,204,204)';
            ctx.fillRect(bx + c * cw, BOARD.y + r * ch, cw + 1, ch + 1);
          }
        }
        ctx.fillStyle = 'rgb(240,120,40)';
        for (const mark of marks) {
          ctx.beginPath();
          ctx.arc(bx + mark.x * BOARD.w, BOARD.y + mark.y * BOARD.h, mark.r, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      };

      // The slide is gradual — 4 px a frame — because a 40 px jump in one frame is
      // outside what Lucas-Kanade will follow, and the point is a tracker that KEEPS
      // its lock while the board moves.
      const towards = () => {
        if (shift < target) shift = Math.min(target, shift + 4);
        draw();
        requestAnimationFrame(towards);
      };
      draw();
      requestAnimationFrame(towards);

      (window as unknown as SceneWindow).__scene = (change) => {
        if (change.degrade !== undefined) degrade = change.degrade ? 1 : 0;
        if (change.shiftTo !== undefined) target = change.shiftTo;
      };

      const stream = canvas.captureStream(30);
      Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {
        configurable: true,
        value: () => Promise.resolve(stream)
      });
      Object.defineProperty(navigator.mediaDevices, 'enumerateDevices', {
        configurable: true,
        value: () => Promise.resolve([])
      });
    },
    { FRAME, BOARD }
  );
}

function scene(
  page: Page,
  change: { degrade?: boolean; shiftTo?: number }
): Promise<void> {
  return page.evaluate(
    (next) => (window as unknown as SceneWindow).__scene(next),
    change
  );
}

/** The live quad's top-left x in VIDEO pixels — the polygon's viewBox is the frame. */
async function liveQuadX(page: Page): Promise<number> {
  const points = (await page.locator('polygon.quad-line').getAttribute('points')) ?? '';
  return Number(points.trim().split(/[\s,]+/)[0]);
}

/**
 * Tap calibrate, disturb the scene once the burst is confirmed to be running, and
 * return the top-left x of the corners that were actually posted.
 *
 * The disturbance is timed off the "Hold steady" banner rather than off the click,
 * because the confirm dialog sits between them: `waitForFunction` polls on animation
 * frames, so this lands inside the first sample period and the 200 ms after it leaves
 * one or two candidates on the near side of the change.
 */
async function burstCalibrate(page: Page, during: () => Promise<void>): Promise<number> {
  const posted = page.waitForRequest(
    (request) => request.url().includes('/api/calibrate') && request.method() === 'POST'
  );
  await page.locator('nav.cluster button', { hasText: 'calibrate' }).click();
  await page.waitForFunction(
    () => document.querySelector('.banner')?.textContent?.includes('Hold steady') === true
  );
  await page.waitForTimeout(200);
  await during();
  const request = await posted;
  return Number((new URL(request.url()).searchParams.get('corners') ?? '').split(',')[0]);
}

async function boxOf(locator: Locator): Promise<Box> {
  const box = await locator.boundingBox();
  expect(box, 'element has no layout box').not.toBeNull();
  return box as Box;
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

  for (const [name, viewport] of [
    ['landscape', LANDSCAPE],
    ['portrait', PORTRAIT]
  ] as const) {
    test(`the picture fills the whole stage in ${name}`, async ({ page }) => {
      await openCamera(page);
      await page.setViewportSize(viewport);
      await liveViewfinder(page);

      // The controls float over the picture rather than beside it, so the fit is
      // against the whole stage: one axis is exhausted, neither overflows, and the
      // letterbox on the other axis is CENTRED IN THE STAGE. That last one is the
      // assertion that bites — the old inset layout produced a picture of exactly
      // this size and hung it off-centre, in the stage minus the rail.
      const stage = await boxOf(page.locator('.camera-stage'));
      const picture = await boxOf(page.locator('.picture'));
      expect(picture.width).toBeLessThanOrEqual(stage.width + 1);
      expect(picture.height).toBeLessThanOrEqual(stage.height + 1);
      expect(
        Math.abs(picture.width - stage.width) < 2 || Math.abs(picture.height - stage.height) < 2,
        'the picture exhausts neither axis of the stage'
      ).toBe(true);
      expect(Math.abs(picture.x - stage.x - (stage.width - picture.width) / 2)).toBeLessThanOrEqual(
        1
      );
      expect(
        Math.abs(picture.y - stage.y - (stage.height - picture.height) / 2)
      ).toBeLessThanOrEqual(1);

      // Uniform scale: the fitted box keeps the source frame's aspect ratio. A
      // second coordinate space here would put wrong corners on the wire while the
      // preview still looked perfect.
      const frame = await page.locator('.picture video').evaluate((el) => {
        const video = el as HTMLVideoElement;
        return { w: video.videoWidth, h: video.videoHeight };
      });
      expect(picture.width / picture.height).toBeCloseTo(frame.w / frame.h, 1);
    });

    test(`the controls float on the picture and are thumb-sized in ${name}`, async ({ page }) => {
      await openCamera(page);
      await page.setViewportSize(viewport);
      await liveViewfinder(page);

      const buttons = page.locator('nav.cluster button');
      await expect(buttons).toHaveCount(CONTROLS.length);
      await expect(buttons).toHaveText(CONTROLS.map((label) => new RegExp(`^${label}`)));

      for (let i = 0; i < CONTROLS.length; i++) {
        const box = await boxOf(buttons.nth(i));
        expect(box.height, `${CONTROLS[i]} height`).toBeGreaterThanOrEqual(MIN_TOUCH_PX);
        expect(box.width, `${CONTROLS[i]} width`).toBeGreaterThanOrEqual(MIN_TOUCH_PX);
      }

      // Inside the stage and therefore on screen. (Whether a given button actually
      // covers picture rather than letterbox depends on the camera's aspect ratio —
      // the fake device is 4:3, a phone is 16:9 — so that is not the assertion. What
      // makes them floating rather than a rail is that the picture is fitted to the
      // whole stage, which the test above measures.)
      const stage = await boxOf(page.locator('.camera-stage'));
      for (let i = 0; i < CONTROLS.length; i++) {
        const box = await boxOf(buttons.nth(i));
        expect(box.x, `${CONTROLS[i]} starts off-stage`).toBeGreaterThanOrEqual(stage.x - 1);
        expect(box.x + box.width, `${CONTROLS[i]} runs off-stage`).toBeLessThanOrEqual(
          stage.x + stage.width + 1
        );
        expect(box.y + box.height, `${CONTROLS[i]} runs off-stage`).toBeLessThanOrEqual(
          stage.y + stage.height + 1
        );
      }

      const cluster = await page.locator('nav.cluster').evaluate((el) => {
        const style = getComputedStyle(el);
        return {
          pointerEvents: style.pointerEvents,
          background: style.backgroundColor,
          border: parseFloat(style.borderLeftWidth) + parseFloat(style.borderTopWidth)
        };
      });
      // The gaps between the pills are inert, or a corner handle under the cluster
      // would be unreachable now that the picture runs beneath it.
      expect(cluster.pointerEvents).toBe('none');
      // And the opaque strip is gone: it was the operator's actual complaint.
      expect(cluster.background).toBe('rgba(0, 0, 0, 0)');
      expect(cluster.border).toBe(0);

      // Translucent pills, not the page's solid accent buttons. This one is here
      // because it regressed the moment it was written: the page's `.camera button`
      // base rule out-specifies a bare `.pill`, and the whole cluster came back as
      // opaque blue rectangles while every other assertion above still passed.
      const secondary = await buttons.first().evaluate((el) => {
        const style = getComputedStyle(el);
        return { background: style.backgroundColor, radius: parseFloat(style.borderRadius) };
      });
      const alpha = Number(/rgba?\([^)]*?([\d.]+)\)$/.exec(secondary.background)?.[1] ?? '1');
      expect(alpha, 'the pills are opaque').toBeLessThan(1);
      expect(secondary.radius, 'the pills are not pill-shaped').toBeGreaterThanOrEqual(20);
      // The primary reads differently from the rest, and by colour, never by an icon.
      const primary = await buttons
        .last()
        .evaluate((el) => getComputedStyle(el).backgroundColor);
      expect(primary).not.toBe(secondary.background);
    });
  }

  test('the primary control sits at the bottom in landscape, where the thumb is', async ({
    page
  }) => {
    await openCamera(page);
    await liveViewfinder(page);

    const buttons = page.locator('nav.cluster button');
    const first = await boxOf(buttons.first());
    const primary = await boxOf(buttons.last());
    await expect(buttons.last()).toHaveText('start');
    // A right-hand column: same x, and the primary is pushed to the far end of it.
    expect(primary.x).toBeCloseTo(first.x, 0);
    expect(primary.y).toBeGreaterThan(first.y + first.height);
    expect(primary.y + primary.height).toBeGreaterThan(LANDSCAPE.height - 60);
  });

  test('portrait lays the same controls out as a bottom row', async ({ page }) => {
    await openCamera(page);
    await liveViewfinder(page);
    await page.setViewportSize(PORTRAIT);

    const buttons = page.locator('nav.cluster button');
    const first = await boxOf(buttons.first());
    const last = await boxOf(buttons.last());
    expect(first.y).toBeCloseTo(last.y, 0);
    expect(last.x).toBeGreaterThan(first.x);
    expect(last.y + last.height).toBeGreaterThan(PORTRAIT.height - 60);
  });

  test('detect board and rotate labels are in the sheet, not on the picture', async ({ page }) => {
    await openCamera(page);
    await liveViewfinder(page);

    for (const label of SHEET_ONLY) {
      await expect(page.locator('nav.cluster button', { hasText: label })).toHaveCount(0);
    }

    await page.locator('nav.cluster button', { hasText: 'more' }).click();
    for (const label of SHEET_ONLY) {
      const button = page.locator('.sheet button', { hasText: label });
      await expect(button).toBeVisible();
      // Still a thumb target down here: the sheet is used on the same phone.
      expect((await boxOf(button)).height).toBeGreaterThanOrEqual(MIN_TOUCH_PX);
    }

    // Their whole result is on the picture, so pressing one closes the sheet.
    await page.locator('.sheet button', { hasText: 'rotate labels' }).click();
    await expect(page.locator('.sheet')).not.toHaveClass(/open/);
  });

  test('calibrate is reachable handheld, and posts the best frame of a burst', async ({ page }) => {
    await openCamera(page);
    await liveViewfinder(page);
    await stubJson(page, '**/api/calibrate**', {
      classes_calibrated: 13,
      backbone: 'resnet18',
      saved: null,
      exemplars: 64,
      label_rotation: 0
    });

    // A stability gate used to sit in front of this and left it disabled FOREVER on
    // the one device the page is for: the motion meter is mirrored value-for-value
    // with the mounted Python edge, and handheld shake never clears its floor.
    const calibrate = page.locator('nav.cluster button', { hasText: 'calibrate' });
    await expect(calibrate).toBeEnabled();

    page.on('dialog', (dialog) => void dialog.accept());
    const posted = page.waitForRequest(
      (request) => request.url().includes('/api/calibrate') && request.method() === 'POST'
    );
    await calibrate.click();

    // The grab is a burst over about a second and a half, and it says so while it runs.
    await expect(page.locator('.banner')).toContainText('Hold steady');

    const request = await posted;
    // Corners ride in the query string, the winning frame in the body — and a real
    // JPEG of a 1280x720 board is kilobytes, not the empty blob a broken canvas
    // ping-pong would produce.
    expect(request.url()).toMatch(/[?&]corners=/);
    expect(request.postDataBuffer()?.byteLength ?? 0, 'no image was sent').toBeGreaterThan(2000);
    await expect(page.locator('.banner.ok')).toContainText('13/13');
  });

  test('the winning frame ships the corners from ITS OWN tick', async ({ page }) => {
    // The invariant the burst rests on, and the one thing no other test here can see:
    // the tracker keeps following the board THROUGH a calibration (only forwarding
    // pauses), so the quad moves while the candidates are being taken. Pair the
    // winning pixels with the quad read once — before the loop or after it — and the
    // node is handed a board outline that describes a different instant than the
    // image. Nothing about the request looks wrong; the bank is just built from
    // slightly misplaced squares.
    //
    // So the scene is driven directly: a drawn board on a canvas stream, moved and
    // softened on cue, twice, so that the winner is an EARLY candidate once and a
    // LATE one once. One case alone would not do — an early winner is also what
    // "read the corners before the loop" produces, and a late one what "read them
    // after" produces. Only both together say the corners follow the winner.
    await FakeViewerSocket.install(page);
    await drawnBoardCamera(page);
    // A detection that hits. It puts the quad exactly on the drawn board AND makes
    // automatic detection stand down, so nothing else owns `activity` at the tap.
    await stubJson(page, '**/api/detect**', {
      corners: [
        [BOARD.x, BOARD.y],
        [BOARD.x + BOARD.w, BOARD.y],
        [BOARD.x + BOARD.w, BOARD.y + BOARD.h],
        [BOARD.x, BOARD.y + BOARD.h]
      ]
    });
    await stubJson(page, '**/api/calibrate**', {
      classes_calibrated: 13,
      backbone: 'resnet18',
      saved: null,
      exemplars: 64,
      label_rotation: 0
    });

    await page.goto('/camera');
    await expect(page.locator('.banner')).toContainText('Board found', { timeout: 20_000 });
    await expect(page.locator('.chip', { hasText: 'tracking — locked' })).toBeVisible();
    page.on('dialog', (dialog) => void dialog.accept());

    // The board only ever moves along x, so one coordinate tells the whole story.
    const start = await liveQuadX(page);

    // (1) Sharp, then it moves and goes soft together: every late candidate scores
    // far below the first few, so the winner is early and its corners are the ones
    // from before the move — even though the live quad has followed the board.
    const sentEarly = await burstCalibrate(page, async () => {
      await scene(page, { degrade: true, shiftTo: SHIFT });
    });
    const afterEarly = await liveQuadX(page);
    expect(Math.abs(sentEarly - start), 'the early winner did not send its own corners').toBeLessThanOrEqual(8);
    expect(afterEarly - start, 'the tracker never followed the board').toBeGreaterThanOrEqual(18);

    // (2) The mirror. It starts soft, moves while still soft, and only THEN goes
    // sharp — so the winner is a late candidate and must carry the moved corners,
    // not the ones this burst opened on.
    const burstStart = await liveQuadX(page);
    const sentLate = await burstCalibrate(page, async () => {
      await scene(page, { shiftTo: SHIFT * 2 });
      await page.waitForTimeout(300);
      await scene(page, { degrade: false });
    });
    const afterLate = await liveQuadX(page);
    expect(sentLate - burstStart, 'the late winner sent the corners the burst opened on').toBeGreaterThanOrEqual(18);
    expect(Math.abs(sentLate - afterLate), 'the late winner sent a quad from no tick at all').toBeLessThanOrEqual(14);
  });

  test('the hint line clears the button column rather than running under it', async ({ page }) => {
    // The fake camera shows no board, so the page opens on its unframed hint and
    // keeps saying something for as long as this test runs.
    await openCamera(page);
    await liveViewfinder(page);

    const hint = page.locator('.hint-bar > span');
    await expect(hint).toBeVisible({ timeout: 20_000 });
    const bar = await boxOf(hint);
    const primary = await boxOf(page.locator('nav.cluster button').last());
    expect(bar.x + bar.width, 'the hint runs under the landscape button column').toBeLessThanOrEqual(
      primary.x + 1
    );
  });
});

/**
 * The notch, driven for real.
 *
 * `viewport-fit=cover` (app.html) is what makes `env(safe-area-inset-*)` non-zero on a
 * phone, and it is also what puts the controls under the notch if nothing budgets for
 * it. Chromium reports zero insets by default — but its device toolbar can override
 * them, and the same CDP command is reachable from here, so these are measurements of
 * the real layout rather than a reading of the stylesheet.
 *
 * In landscape the notch is on the LEFT or the RIGHT depending on which way the phone
 * was turned, so both are exercised. The home indicator is always at the bottom.
 */
const NOTCH_PX = 44;
const INDICATOR_PX = 21;

test.describe('safe areas', () => {
  test.use({ viewport: LANDSCAPE });

  async function withNotch(page: Page, side: 'left' | 'right'): Promise<void> {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Emulation.setSafeAreaInsetsOverride', {
      insets: {
        top: 0,
        bottom: INDICATOR_PX,
        left: side === 'left' ? NOTCH_PX : 0,
        right: side === 'right' ? NOTCH_PX : 0
      }
    });
  }

  test('the stage keeps the whole viewport — the picture is meant to go under it', async ({
    page
  }) => {
    await openCamera(page);
    await withNotch(page, 'right');
    await liveViewfinder(page);

    // Nothing is carved off the stage for the inset: only the chrome inside it moves.
    const stage = await boxOf(page.locator('.camera-stage'));
    expect(stage.width).toBeCloseTo(LANDSCAPE.width, 0);
    expect(stage.height).toBeCloseTo(LANDSCAPE.height, 0);

    const picture = await boxOf(page.locator('.picture'));
    expect(Math.abs(picture.x - stage.x - (stage.width - picture.width) / 2)).toBeLessThanOrEqual(
      1
    );
  });

  test('the buttons and the hint line move in off a right-hand notch', async ({ page }) => {
    await openCamera(page);
    await withNotch(page, 'right');
    await liveViewfinder(page);

    const stage = await boxOf(page.locator('.camera-stage'));
    const buttons = page.locator('nav.cluster button');
    for (let i = 0; i < CONTROLS.length; i++) {
      const box = await boxOf(buttons.nth(i));
      expect(box.x + box.width, `${CONTROLS[i]} is under the notch`).toBeLessThanOrEqual(
        stage.x + stage.width - NOTCH_PX + 1
      );
    }
    // The home indicator, at the bottom of the same column.
    const primary = await boxOf(buttons.last());
    expect(primary.y + primary.height).toBeLessThanOrEqual(
      stage.y + stage.height - INDICATOR_PX + 1
    );

    const hint = await boxOf(page.locator('.hint-bar > span'));
    expect(hint.y + hint.height).toBeLessThanOrEqual(stage.y + stage.height - INDICATOR_PX + 1);
  });

  test('the chips move in off a left-hand notch', async ({ page }) => {
    await openCamera(page);
    await withNotch(page, 'left');
    await liveViewfinder(page);

    const stage = await boxOf(page.locator('.camera-stage'));
    const chip = await boxOf(page.locator('.chip').first());
    expect(chip.x).toBeGreaterThanOrEqual(stage.x + NOTCH_PX);

    // The hint line is centred, so what is asserted about it is its own gutter.
    const gutter = await page
      .locator('.hint-bar')
      .evaluate((el) => parseFloat(getComputedStyle(el).paddingLeft));
    expect(gutter).toBeGreaterThanOrEqual(NOTCH_PX);
  });

  test('the position panel parks clear of a left-hand notch, and drags from there', async ({
    page
  }) => {
    await openCamera(page);
    await withNotch(page, 'left');
    await liveViewfinder(page);
    await page.locator('nav.cluster button', { hasText: 'position' }).click();

    // `placePanel` is pure and DOM-free — it parks at 12 px and cannot know about the
    // sensor housing. The rendered left is a CSS `max()` the browser resolves against
    // the real inset, which is the only place that fact exists.
    const stage = await boxOf(page.locator('.camera-stage'));
    const panel = page.locator('.pos-panel');
    const parked = await boxOf(panel);
    expect(parked.x, 'the drag strip is behind the notch').toBeGreaterThanOrEqual(
      stage.x + NOTCH_PX
    );

    // And the first drag must not jump. The handler measures the box rather than
    // trusting the number behind the `max()`; taking the number would teleport the
    // panel left by exactly the difference the moment the pointer moved.
    const head = await boxOf(page.locator('.pos-head'));
    await page.mouse.move(head.x + 40, head.y + 10);
    await page.mouse.down();
    await page.mouse.move(head.x + 140, head.y + 60, { steps: 5 });
    await page.mouse.up();

    const moved = await boxOf(panel);
    expect(Math.abs(moved.x - parked.x - 100), 'the panel jumped on grab').toBeLessThanOrEqual(2);
    expect(Math.abs(moved.y - parked.y - 50), 'the panel jumped on grab').toBeLessThanOrEqual(2);
  });

  test('portrait budgets the home indicator ONCE, not twice', async ({ page }) => {
    // `--cluster-h` already contains the inset in portrait, so the hint line must not
    // add it again — the sentence would float a second indicator's height above the
    // buttons for no reason, and the bug is invisible on a phone without one.
    await openCamera(page);
    await page.setViewportSize(PORTRAIT);
    await liveViewfinder(page);

    // The gap between the sentence and the buttons, which is the cluster's own
    // padding and nothing else. An inset is budgeted by `--cluster-h`, which moves
    // BOTH of them together — so the gap is the thing that must not change.
    const gap = async (): Promise<number> => {
      const hint = await boxOf(page.locator('.hint-bar > span'));
      const row = await boxOf(page.locator('nav.cluster button').first());
      return row.y - (hint.y + hint.height);
    };

    const flat = await gap();
    await withNotch(page, 'right');
    expect(await gap(), 'the hint counts the home indicator a second time').toBeCloseTo(flat, 0);
  });

  test('the sheet clears the home indicator', async ({ page }) => {
    await openCamera(page);
    await liveViewfinder(page);
    await page.locator('nav.cluster button', { hasText: 'more' }).click();

    // The sheet is taller than the screen and scrolls, so its clearance is not a
    // position but the gutter under its last line. Measured as the DIFFERENCE the
    // emulated indicator makes: a sheet whose padding merely happens to exceed 21 px
    // is not budgeting for anything.
    const gutter = (): Promise<number> =>
      page.locator('.sheet').evaluate((el) => parseFloat(getComputedStyle(el).paddingBottom));
    const flat = await gutter();
    await withNotch(page, 'right');
    expect(await gutter()).toBeCloseTo(flat + INDICATOR_PX, 0);
  });
});

test.describe('the position panel', () => {
  test.use({ viewport: LANDSCAPE });

  test('starts hidden and is opened by the "position" control', async ({ page }) => {
    const socket = await openCamera(page);

    // It used to open over the picture the operator was aiming; now it is asked for.
    const panel = page.locator('.pos-panel');
    await expect(panel).toBeHidden();
    // Hidden with `visibility`, never `display: none` — a 0x0 box makes chessground's
    // own ResizeObserver compute NaN and the board never recovers.
    expect(await panel.evaluate((el) => getComputedStyle(el).display)).not.toBe('none');
    await expect(panel.locator('cg-board')).toHaveCount(1);

    await page.locator('nav.cluster button', { hasText: 'position' }).click();
    await expect(panel).toBeVisible();

    // Parked top-left: the button column owns the right-hand edge now.
    const box = await boxOf(panel);
    expect(box.x).toBeLessThan(LANDSCAPE.width / 2);
    expect(box.y).toBeLessThan(LANDSCAPE.height / 2);

    // "waiting…", not a position: the page has been handed nothing yet. Showing the
    // start position here would assert a reading that never arrived.
    await expect(panel.locator('.st')).toHaveText('waiting…');
    await expect(panel.locator('piece')).toHaveCount(0);

    // The board id defaults to `phone`; an event for it fills the same board in place.
    socket.send(positionEvent({ board_id: 'phone', seq: 4, confidence: 0.81 }));
    await expect(panel.locator('piece')).toHaveCount(32);
    await expect(panel.locator('.st')).toHaveText('ok · seq 4 · confidence 0.81');

    await page.locator('nav.cluster button', { hasText: 'position' }).click();
    await expect(panel).toBeHidden();
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

    // With no camera there is no viewfinder at all — no dead stage, no dead controls.
    await expect(page.locator('.camera-stage')).toHaveCount(0);
    await expect(page.locator('nav.cluster')).toHaveCount(0);
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
