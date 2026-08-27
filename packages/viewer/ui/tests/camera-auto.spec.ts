/**
 * The `/camera` page with nobody touching the phone.
 *
 * Everything in `camera.spec.ts` is a property of the page as a phone RENDERS it.
 * Everything here is a property of the page as a phone LEAVES it: propped on a
 * tripod beside a board, streaming, for two hours, while the operator plays. Four
 * mechanisms run in that stretch with no tap behind any of them —
 *
 *  - **automatic detection**, which frames the board on first sight and verifies a
 *    restored crop at page open (a crop saved last session may be a rank out);
 *  - **automatic calibration**, which fires once the reading looks like a start
 *    position sitting still;
 *  - **periodic re-detection**, which asks the detector every 10 s whether the
 *    tracked quad has walked, and re-locks it when it has
 *    (`docs/reports/2026-08-27-framing-drift.md`);
 *  - **the portrait hard block**, which is what gates the other three.
 *
 * `state.ts` is pure and its verdicts are unit-tested to the decimal. What cannot be
 * unit-tested is the WIRING: whether the timer runs, whether the request carries the
 * live frame, whether the verdict reaches the quad, whether `rot` survives the trip,
 * and — the one that costs a whole session when it is wrong — whether the single
 * `activity` slot is handed back when the node does not answer. All five have to be
 * driven in a browser to be seen at all.
 *
 * **Time is faked, the camera is not.** `page.clock.install()` leaves the clock
 * running in real time and lets a spec jump it forward, which is the only way to
 * cross a 10 s interval and an 80 s back-off without a spec that takes three
 * minutes. See `advanceTo` for the one thing that is not free about doing so.
 */

import { expect, test, type Page } from '@playwright/test';
import {
  EMPTY_PLACEMENT,
  FakeIngestSocket,
  FakeViewerSocket,
  START_PLACEMENT,
  positionEvent,
  stubJson
} from './helpers/node';
import {
  BOARD,
  CORNERS_KEY,
  FRAME,
  LANDSCAPE,
  boardQuad,
  drawnBoardCamera,
  liveQuadX,
  liveViewfinder,
  seedFraming
} from './helpers/camera';

// ---- what the page believes, spelled here so a spec states its own numbers ------
// Every constant below is `$lib/components/camera/state.ts`'s, copied rather than
// imported: a spec that imported them would still pass if one were changed to
// nonsense, and these are the numbers the mechanism IS.

/** `REDETECT_INTERVAL_MS` — how often a resting, streaming page checks itself. */
const REDETECT_INTERVAL_MS = 10_000;
/** `REDETECT_BACKOFF_MS` — the gaps after one, two and three consecutive misses. */
const REDETECT_BACKOFF_MS = [20_000, 40_000, 80_000] as const;
/** `DETECT_TIMEOUT_MS` (`$lib/api`) — the page's own bound on a silent node. */
const DETECT_TIMEOUT_MS = 15_000;
/** `CORNER_ORDER` (`$lib/geometry/corners`): slot `i` reads `CORNER_ORDER[i + rot]`. */
const CORNER_ORDER = ['a8', 'h8', 'h1', 'a1'] as const;

/**
 * One square of the drawn board, in frame pixels.
 *
 * `squareWidth` is the mean side over eight: this board is 800 x 448, so
 * (800 + 448) / 4 / 8 = 78. Every shift below is quoted as a multiple of it,
 * because that is the unit `ADOPT_SHIFT_SQUARES` (0.45) and
 * `ADOPT_MAX_SHIFT_SQUARES` (2.0) are measured in.
 */
const SQUARE_PX = 78;
/**
 * The drift the specs make the page live through: 0.7 square. Comfortably over the
 * adoption threshold, comfortably under the reposition cap, and a pure translation,
 * so every corner pairing but the right one is a whole board's side out and
 * `ADOPT_RELABEL_MARGIN` is never the thing that decides.
 */
const DRIFT_PX = Math.round(0.7 * SQUARE_PX);

// ---- driving the page ----------------------------------------------------------

/**
 * A quad the page refuses — `checkQuadShape` rejects a sliver — which is what every
 * kind of detector miss looks like from up here.
 */
const SLIVER: [number, number][] = [
  [10, 10],
  [1200, 12],
  [1201, 14],
  [11, 12]
];

interface DetectStub {
  /** What the next `/detect` answers with, or `null` to accept and never answer. */
  answer: readonly (readonly [number, number])[] | null;
  /** How many detections the page has asked for. Counted in node, so it is exact. */
  calls: number;
}

async function stubDetect(page: Page, answer: DetectStub['answer']): Promise<DetectStub> {
  const stub: DetectStub = { answer, calls: 0 };
  await page.route('**/api/detect**', async (route) => {
    stub.calls++;
    const corners = stub.answer;
    // A node that took the request and then said nothing at all. Nothing here ends
    // it; the page's own `AbortSignal.timeout` is the only thing that can.
    if (corners === null) await new Promise(() => {});
    else await route.fulfill({ json: { corners } });
  });
  return stub;
}

interface Counted {
  calls: number;
}

/** Answer an endpoint with a fixed body, and count what asked for it. */
async function countingStub(page: Page, pattern: string, body: unknown): Promise<Counted> {
  const seen: Counted = { calls: 0 };
  await page.route(pattern, async (route) => {
    seen.calls++;
    await route.fulfill({ json: body });
  });
  return seen;
}

/**
 * The endpoints these specs do not assert on but the page keeps asking for.
 *
 * The trial log posts a batch every 3 s; against the preview server that is a 404
 * per batch, and these specs fast-forward through a lot of batches.
 */
async function stubQuietNode(page: Page): Promise<void> {
  await stubJson(page, '**/api/client-log**', { ok: true });
}

/** The page's own `performance.now()` — the clock every gate here is written against. */
function pageNow(page: Page): Promise<number> {
  return page.evaluate(() => performance.now());
}

/**
 * How far the clock may jump at once, and how long real time is then given to catch up.
 *
 * `fastForward` fires each due timer ONCE, at the FAR END of the jump, so the frame
 * loop's next tick reads a `performance.now()` that moved by the whole window while
 * `video.currentTime` did not move at all. That is precisely the reading the camera
 * stall watchdog exists to catch (`STALL_PLAY_MS` 2 s → `play()`, `STALL_REOPEN_MS`
 * 6 s → reopen the camera), and a page that reopens its camera in the middle of a
 * drift test has thrown away the thing being measured. So the clock is crossed in
 * steps well under `STALL_PLAY_MS`, each followed by a moment of real time in which
 * the camera delivers frames and the watchdog resets itself. The margin between the
 * step and the threshold is what the whole suite's contention gets to eat: a dozen
 * workers each drawing a canvas can starve a frame loop for a few hundred ms.
 */
const CLOCK_STEP_MS = 1500;
const REAL_FRAMES_MS = 120;

/**
 * Move the page's clock to `mark` — an absolute `performance.now()` reading — or
 * stop early once `until` says the thing being waited for has happened.
 *
 * Absolute rather than a duration on purpose: the clock is still running, so every
 * assertion between two jumps spends the very budget being measured. A spec that
 * said "advance 19 s" would really advance 19 s plus however long its own
 * `expect`s took, which is exactly how a 20 s back-off test passes on a fast
 * machine and fails on a loaded one.
 */
async function advanceTo(page: Page, mark: number, until: () => boolean = () => false): Promise<void> {
  for (;;) {
    if (until()) return;
    const left = mark - (await pageNow(page));
    if (left <= 0) return;
    await page.clock.fastForward(Math.min(CLOCK_STEP_MS, left));
    await page.waitForTimeout(REAL_FRAMES_MS);
  }
}

/**
 * Drive the page to its next drift check and wait for the verdict to be folded in.
 *
 * The window is the interval plus a cushion, and it stops the moment the request
 * appears — advancing blindly for the whole window could carry the page past a
 * SECOND check and make the count say something nobody meant.
 */
async function nextDriftCheck(page: Page, detect: DetectStub): Promise<void> {
  const expected = detect.calls + 1;
  const answered = page.waitForResponse((response) => response.url().includes('/api/detect'));
  const deadline = (await pageNow(page)) + REDETECT_INTERVAL_MS + 3_000;
  await advanceTo(page, deadline, () => detect.calls >= expected);
  expect(detect.calls, 'the page never asked for its next drift check').toBe(expected);
  await answered;
  // The verdict lands a microtask after the body; a real tick settles the quad.
  await page.waitForTimeout(200);
}

/**
 * A page that has found its board, been given a labelling, started streaming, and
 * had its first drift check answered — the state the mechanism runs in, and the
 * state a phone spends a game in.
 *
 * The labels are turned once deliberately: `rot` is the one thing an adoption could
 * quietly lose, and it can only be seen to survive if it was something other than
 * the default to begin with.
 *
 * Waiting out that first check is the whole reason this is a helper. It costs no
 * time at all — `REDETECT_START.nextAt` is 0 — so it goes out the moment streaming
 * does, and on a loaded machine it can be on the wire BEFORE the pill has repainted
 * to `stop`. A spec that counted detections from after the click was therefore
 * racing it, and read one or two depending on the weather.
 */
async function streamingFramedPage(page: Page, detect: DetectStub): Promise<void> {
  await page.goto('/camera');
  await liveViewfinder(page);
  await expect(page.locator('.banner.ok')).toContainText('Board found', { timeout: 20_000 });
  await expect(page.locator('.chip', { hasText: 'tracking — locked' })).toBeVisible();

  await page.locator('nav.cluster button', { hasText: 'more' }).click();
  await page.locator('.sheet button', { hasText: 'rotate labels' }).click();
  await expect(page.locator('.corner-handle').first()).toHaveText(CORNER_ORDER[1]);

  const framed = detect.calls;
  await page.locator('nav.cluster button', { hasText: 'start' }).click();
  await expect(page.locator('nav.cluster button', { hasText: 'stop' })).toBeVisible();
  await expect.poll(() => detect.calls, { timeout: 20_000 }).toBe(framed + 1);
  await page.waitForTimeout(200);
}

test.describe('finding the board without being asked', () => {
  test.use({ viewport: LANDSCAPE });

  test('the board is framed on first sight, and the labelling is left alone', async ({ page }) => {
    // The common case should need no framing step at all: point the phone at the
    // board and the page finds it. Everything downstream — the drift check, the
    // automatic calibration, a streamable crop — waits on this one detection.
    await FakeViewerSocket.install(page);
    await stubQuietNode(page);
    const detect = await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);

    await page.goto('/camera');
    await liveViewfinder(page);

    // `AUTO_SETTLE_MS` first: a detection on frame one is spent on a frame taken
    // before autoexposure and autofocus have settled.
    await expect(page.locator('.banner.ok')).toContainText('Board found', { timeout: 20_000 });
    expect(detect.calls, 'one detection framed it, and it was not asked twice').toBe(1);
    await expect(page.locator('.chip', { hasText: 'tracking — locked' })).toBeVisible();

    // On the board that was drawn, in the frame's own pixels.
    expect(Math.abs((await liveQuadX(page)) - BOARD.x)).toBeLessThanOrEqual(8);

    // And geometry cannot know which corner is a8 (DESIGN decision #15), so the
    // labelling is untouched and the page says so rather than implying a reading.
    await expect(page.locator('.banner')).toContainText('rotate labels');
    await expect(page.locator('.corner-handle').first()).toHaveText(CORNER_ORDER[0]);
  });

  test('a restored crop is verified at page open, and a confirmation does not nag', async ({
    page
  }) => {
    // 2026-08-27: a stale restored quad read the board a full rank off and starved
    // the zero-tap calibration, silently, because a restored framing IS a framing
    // and automatic detection used to stand down for it. Now it stands down only
    // after one detection has agreed with it.
    await FakeViewerSocket.install(page);
    await stubQuietNode(page);
    const detect = await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);
    await seedFraming(page, boardQuad(), 2);

    await page.goto('/camera');
    await liveViewfinder(page);

    // The labelling came back with the crop: two turns put h1 on storage slot 0.
    await expect(page.locator('.corner-handle').first()).toHaveText(CORNER_ORDER[2]);

    await expect.poll(() => detect.calls, { timeout: 20_000 }).toBe(1);
    await page.waitForTimeout(500);

    // It landed on the same quad (inside `QUAD_VERIFY_SHIFT`), so nothing moved,
    // and a verify that moved nothing must not nag — this runs on every page open.
    await expect(page.locator('nav.cluster button', { hasText: 'rotate labels' })).toHaveCount(0);
    await expect(page.locator('.banner')).toHaveCount(0);
    // And the restored labelling survived the detection that confirmed it.
    await expect(page.locator('.corner-handle').first()).toHaveText(CORNER_ORDER[2]);
  });

  test('a restored crop the detector disagrees with warns that the labels may have moved', async ({
    page
  }) => {
    await FakeViewerSocket.install(page);
    await stubQuietNode(page);
    // The board is where it always was; last session's crop is 200 px to the right
    // of it — 0.14 of the frame diagonal, four times `QUAD_VERIFY_SHIFT`.
    const detect = await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);
    await seedFraming(page, boardQuad(200, 0), 0);

    await page.goto('/camera');
    await liveViewfinder(page);

    // A detection replaces the quad but never `rot`, so the labels can now sit on a
    // different physical corner than the operator chose. That is the one case where
    // the page moves a quad somebody else authored, and it stays on screen until
    // the operator answers it.
    await expect(page.locator('.banner.warn')).toContainText(
      'The board was found again after being lost',
      { timeout: 20_000 }
    );
    const rotate = page.locator('nav.cluster button', { hasText: 'rotate labels' });
    await expect(rotate).toBeVisible();
    expect(await rotate.evaluate((el) => getComputedStyle(el).boxShadow)).toContain('255, 207, 51');

    // The quad itself did move onto the board.
    expect(Math.abs((await liveQuadX(page)) - BOARD.x)).toBeLessThanOrEqual(8);
  });
});

test.describe('calibrating without being asked', () => {
  test.use({ viewport: LANDSCAPE });

  test('a start position at rest calibrates itself, with no tap and no confirm', async ({
    page
  }) => {
    // "detect board -> calibrate should be fully automatic" (operator, 2026-08-27).
    // The trigger is the READING, not the picture: positions have to be flowing and
    // the node's own answer has to look like a start position sitting still.
    const socket = await FakeViewerSocket.install(page);
    await FakeIngestSocket.install(page);
    await stubQuietNode(page);
    await stubDetect(page, boardQuad());
    const calibrate = await countingStub(page, '**/api/calibrate**', {
      classes_calibrated: 13,
      backbone: 'resnet18',
      saved: null,
      exemplars: 64,
      label_rotation: 0
    });
    await drawnBoardCamera(page);

    // Nothing accepts a dialog here. Playwright DISMISSES an unhandled one, so a
    // calibration that got as far as asking would be refused and never reach the
    // node — which makes a successful reply proof that nothing asked.
    let dialogs = 0;
    page.on('dialog', (dialog) => {
      dialogs++;
      void dialog.dismiss();
    });

    await page.goto('/camera');
    await liveViewfinder(page);
    await expect(page.locator('.banner.ok')).toContainText('Board found', { timeout: 20_000 });

    // Framed and streaming, and still nothing: no reading has arrived.
    await page.locator('nav.cluster button', { hasText: 'start' }).click();
    await expect(page.locator('nav.cluster button', { hasText: 'stop' })).toBeVisible();
    await page.waitForTimeout(1500);
    expect(calibrate.calls, 'calibrated with no reading to justify it').toBe(0);

    // A reading that is not a start position is not a trigger either. Occupancy is
    // the test — two opposite full bands, middle empty — and an empty board has none.
    socket.send(positionEvent({ seq: 2, placement: EMPTY_PLACEMENT }));
    await expect(page.locator('.pos-panel .st')).toContainText('seq 2');
    await page.waitForTimeout(1500);
    expect(calibrate.calls, 'calibrated from a board with nothing on it').toBe(0);

    // And now the one that is.
    const posted = page.waitForRequest(
      (request) => request.url().includes('/api/calibrate') && request.method() === 'POST'
    );
    socket.send(positionEvent({ seq: 3, placement: START_PLACEMENT }));
    await posted;

    await expect(page.locator('.banner.ok')).toContainText(
      'Calibrated automatically from the start position, 13/13 classes',
      { timeout: 20_000 }
    );
    // The count on the pill is the operator's record that it happened.
    await expect(page.locator('nav.cluster button', { hasText: 'calibrate' })).toHaveText(
      'calibrate · 1'
    );
    expect(dialogs, 'the automatic path stopped to ask').toBe(0);

    // Automation covers the FIRST calibration only; extra angles stay deliberate.
    socket.send(positionEvent({ seq: 4, placement: START_PLACEMENT }));
    await page.waitForTimeout(1500);
    expect(calibrate.calls, 'a second calibration fired by itself').toBe(1);
  });
});

/**
 * The tracker DRIFTS. It is a Lucas-Kanade plane tracker, so every frame's
 * homography is measured against the last one and the errors accumulate: on a board
 * standing still, on a shaky holder, the quad walks about a square in two minutes
 * (`docs/reports/2026-08-27-framing-drift.md`, take 5 — 58 % of that session's wrong
 * squares). Nobody sees it happen, because the overlay is drawn where the tracker
 * THINKS the board is.
 *
 * So a resting, streaming page asks the detector where the board really is every
 * 10 s. What follows drives the whole mechanism — the per-session baseline, the
 * threshold, the consecutive-reading counter, the back-off, the timeout — through a
 * real page, at the one thing a unit test cannot see: the wiring.
 */
test.describe('the framing re-detect', () => {
  test.use({ viewport: LANDSCAPE });

  test('a frame that has drifted is re-locked, and the labelling rides along', async ({ page }) => {
    test.setTimeout(120_000);
    await page.clock.install();
    await FakeViewerSocket.install(page);
    await FakeIngestSocket.install(page);
    await stubQuietNode(page);
    const detect = await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);
    await streamingFramedPage(page, detect);

    // Three successful checks are spent MEASURING and judge nothing — the first of
    // them is the one `streamingFramedPage` waited out. They fix R0, the constant
    // offset between where the detector puts a corner and where the tracked quad
    // does, without which every hand-framed session false-alarms.
    await nextDriftCheck(page, detect);
    await nextDriftCheck(page, detect);
    await expect(page.locator('.banner'), 'baselining spoke').toHaveCount(0);

    // Now the tracked quad has walked: the detector keeps finding the board where it
    // really is, 0.7 square from where the page thinks it is. Over
    // `ADOPT_SHIFT_SQUARES` (0.45), far under `ADOPT_MAX_SHIFT_SQUARES` (2.0).
    detect.answer = boardQuad(DRIFT_PX, 0);
    const before = await liveQuadX(page);

    // One reading over the threshold arms; it does not act. At K = 1 the sweep in
    // the report fires on a control take that mis-addressed nothing.
    await nextDriftCheck(page, detect);
    await expect(page.locator('.banner'), 'the page acted on one reading').toHaveCount(0);

    // Two consecutive, and it re-locks.
    await nextDriftCheck(page, detect);
    await expect(page.locator('.banner.ok')).toHaveText(
      'The frame had drifted — re-locked onto the board.'
    );

    const after = await liveQuadX(page);
    expect(after - before, 'the quad was not moved onto the board').toBeGreaterThan(
      0.45 * SQUARE_PX
    );
    expect(after - before, 'the quad overshot the detector').toBeLessThan(DRIFT_PX + 20);

    // `rot` is the one thing an adoption must not touch: the correction is the
    // detector's quad, and the detector has no idea which corner is a8. A page that
    // dropped it here would re-crop correctly and read the board a quarter turn out.
    await expect(page.locator('.corner-handle').first()).toHaveText(CORNER_ORDER[1]);
    // Saved with it, since that is what the next page open restores.
    const saved = await page.evaluate(
      (key) => JSON.parse(localStorage.getItem(key) ?? 'null') as { pts: number[][]; rot: number },
      CORNERS_KEY
    );
    expect(saved.rot, 'the saved labelling was lost with the re-lock').toBe(1);
    expect(
      (saved.pts[0]?.[0] ?? 0) * FRAME.width,
      'the corrected quad was never saved'
    ).toBeGreaterThan(BOARD.x + 0.45 * SQUARE_PX);

    // And nothing about a re-lock is the operator's to do, so the stream never stopped.
    await expect(page.locator('nav.cluster button', { hasText: 'stop' })).toBeVisible();
  });

  test('a detector that cannot see the board is asked less and less, then not at all', async ({
    page
  }) => {
    // Nobody asked for this check. A page pointed at a board the detector cannot
    // see would otherwise post a full-frame JPEG every 10 s for a whole session, on
    // a phone battery — so the misses buy silence, and buy it permanently.
    test.setTimeout(180_000);
    await page.clock.install();
    await FakeViewerSocket.install(page);
    await FakeIngestSocket.install(page);
    await stubQuietNode(page);
    const detect = await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);
    await streamingFramedPage(page, detect);

    // The check the page has already made was answered, so the budget is whole.
    // From here the detector misses every time.
    detect.answer = SLIVER;
    await nextDriftCheck(page, detect);
    const misses = detect.calls;
    const firstMiss = await pageNow(page);

    // Silent: the framing this could not confirm is still the best one anybody has.
    await expect(page.locator('.banner')).toHaveCount(0);
    // And the slot is handed back. A miss that kept `activity` would leave detect
    // and calibrate dead for the rest of the session with nothing saying why —
    // which is the whole reason `verifyFraming` clears it in a `finally`.
    await expect(page.locator('nav.cluster button', { hasText: 'detect board' })).toBeEnabled();
    await expect(page.locator('nav.cluster button', { hasText: 'calibrate' })).toBeEnabled();

    // One miss buys 20 s, not the usual 10.
    await advanceTo(page, firstMiss + REDETECT_BACKOFF_MS[0] - 2_000);
    expect(detect.calls, 'the back-off after one miss was not taken').toBe(misses);
    await advanceTo(page, firstMiss + REDETECT_BACKOFF_MS[0] + 3_000, () => detect.calls > misses);
    expect(detect.calls).toBe(misses + 1);
    const secondMiss = await pageNow(page);

    // Two, and it is 40.
    await advanceTo(page, secondMiss + REDETECT_BACKOFF_MS[1] - 2_000);
    expect(detect.calls, 'the back-off did not grow with the second miss').toBe(misses + 1);
    await advanceTo(page, secondMiss + REDETECT_BACKOFF_MS[1] + 3_000, () => detect.calls > misses + 1);
    expect(detect.calls).toBe(misses + 2);
    const thirdMiss = await pageNow(page);

    // Three is the budget. The 80 s that would have come next never arrives, and
    // neither does anything after it: the page has stopped asking for good.
    await advanceTo(page, thirdMiss + REDETECT_BACKOFF_MS[2] + 20_000);
    expect(detect.calls, 'the page went on asking after it had given up').toBe(misses + 2);
    await expect(page.locator('.banner'), 'giving up quietly is the point').toHaveCount(0);
    // Given up on its own check, not on the session: the controls are still live.
    await expect(page.locator('nav.cluster button', { hasText: 'detect board' })).toBeEnabled();
  });

  test('a node that takes the request and never answers does not wedge the controls', async ({
    page
  }) => {
    // `fetch` has no default timeout, and all three callers of `/detect` claim the
    // page's single `activity` slot for the duration. A node that accepts and then
    // says nothing would therefore park the page in `verifying` FOREVER — detect
    // and calibrate both dead, no banner, nothing to press. The bound lives at the
    // endpoint (`$lib/api`) precisely because all three need it.
    test.setTimeout(120_000);
    await page.clock.install();
    await FakeViewerSocket.install(page);
    await FakeIngestSocket.install(page);
    await stubQuietNode(page);
    const detect = await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);
    await streamingFramedPage(page, detect);

    const answered = detect.calls;
    detect.answer = null; // accepted, and then nothing at all
    // The next check is a full interval away — the one before it came back.
    await advanceTo(
      page,
      (await pageNow(page)) + REDETECT_INTERVAL_MS + 3_000,
      () => detect.calls > answered
    );
    expect(detect.calls).toBe(answered + 1);
    const asked = await pageNow(page);

    // While it is out there the slot is claimed — and the page says so, which is
    // the difference between a disabled control and a broken one.
    const calibrate = page.locator('nav.cluster button', { hasText: 'calibrate' });
    await expect(calibrate).toBeDisabled();
    await expect(calibrate).toHaveAttribute('title', 'checking the framing…');
    await expect(page.locator('nav.cluster button', { hasText: 'detect board' })).toBeDisabled();

    // 15 s, and the slot comes back on its own.
    await advanceTo(page, asked + DETECT_TIMEOUT_MS + 3_000);
    await expect(calibrate).toBeEnabled();
    await expect(calibrate).toHaveAttribute(
      'title',
      'rebuild the piece bank from the live camera (the board must show the starting position)'
    );
    await expect(page.locator('nav.cluster button', { hasText: 'detect board' })).toBeEnabled();
    // A timeout is a miss like any other: it spends the budget and says nothing.
    await expect(page.locator('.banner')).toHaveCount(0);
  });
});

/**
 * Piece crops extend toward the top of the image, so a sideways board reads every
 * piece into its neighbour. It cost an afternoon of reads that looked like a
 * position problem (2026-08-27), and it is now a hard block rather than a warning.
 *
 * Two signals feed it: the frame's own aspect — the ground truth for what would
 * actually be sent — and `screen.orientation`, which is the early one, available
 * before the camera has even opened. The camera here is landscape throughout, so
 * these drive the screen and nothing else.
 */
test.describe('the portrait hard block', () => {
  test.use({ viewport: LANDSCAPE });

  interface TurnWindow {
    __turn: (type: string) => void;
  }

  /** Take over `screen.orientation`, and offer a way to turn the phone mid-test. */
  async function fakeOrientation(page: Page, initial: string): Promise<void> {
    await page.addInitScript((start: string) => {
      const target = new EventTarget();
      let current = start;
      Object.defineProperty(target, 'type', { get: () => current });
      Object.defineProperty(target, 'angle', { get: () => 0 });
      Object.defineProperty(screen, 'orientation', { configurable: true, get: () => target });
      (window as unknown as TurnWindow).__turn = (next: string) => {
        current = next;
        target.dispatchEvent(new Event('change'));
      };
    }, initial);
  }

  function turn(page: Page, type: string): Promise<void> {
    return page.evaluate((next) => (window as unknown as TurnWindow).__turn(next), type);
  }

  test('a phone held upright explains itself and gates everything that would send', async ({
    page
  }) => {
    await fakeOrientation(page, 'portrait-primary');
    await FakeViewerSocket.install(page);
    await stubQuietNode(page);
    const detect = await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);

    await page.goto('/camera');
    await liveViewfinder(page);

    const block = page.locator('.portrait-block');
    await expect(block).toBeVisible();
    await expect(block.locator('.head')).toHaveText('turn the phone sideways');
    // Not streaming, so it says what turning the phone would unlock rather than
    // what it has paused.
    await expect(block).toContainText('detect, calibrate and start unlock in landscape');

    await expect(page.locator('nav.cluster button', { hasText: 'start' })).toBeDisabled();
    await expect(page.locator('nav.cluster button', { hasText: 'detect board' })).toBeDisabled();
    const calibrate = page.locator('nav.cluster button', { hasText: 'calibrate' });
    await expect(calibrate).toBeDisabled();
    await expect(calibrate).toHaveAttribute(
      'title',
      'the frame is portrait — turn the phone sideways'
    );

    // Framing is NOT gated, and deliberately: finding the board costs nothing that
    // a sideways crop could spoil, and the page is then ready the moment the phone
    // turns. Only what would be READ from the frame is blocked.
    await expect.poll(() => detect.calls, { timeout: 20_000 }).toBe(1);
  });

  test('turning the phone mid-stream pauses the wire without dropping the stream', async ({
    page
  }) => {
    await fakeOrientation(page, 'landscape-primary');
    await FakeViewerSocket.install(page);
    const ingest = await FakeIngestSocket.install(page);
    await stubQuietNode(page);
    await stubDetect(page, boardQuad());
    await drawnBoardCamera(page);

    await page.goto('/camera');
    await liveViewfinder(page);
    await expect(page.locator('.banner.ok')).toContainText('Board found', { timeout: 20_000 });
    await page.locator('nav.cluster button', { hasText: 'start' }).click();
    // A still scene forwards at `KEEPALIVE_FPS` — one frame a second, each a header
    // and then its JPEG.
    await expect.poll(() => ingest.frames, { timeout: 20_000 }).toBeGreaterThan(1);

    await turn(page, 'portrait-primary');
    const block = page.locator('.portrait-block');
    await expect(block).toBeVisible();
    await expect(block).toContainText('stream paused —');
    await expect(block).toContainText('sending resumes in landscape');

    // The stream was starved, not stopped: the frames are still metered (stability
    // has to keep meaning something) and the pill still offers to stop it.
    await expect(page.locator('nav.cluster button', { hasText: 'stop' })).toBeVisible();
    const paused = ingest.frames;
    await page.waitForTimeout(3_000);
    expect(ingest.frames, 'a portrait frame was put on the wire').toBe(paused);

    // And it resumes by itself, with no tap.
    await turn(page, 'landscape-primary');
    await expect(block).toHaveCount(0);
    await expect.poll(() => ingest.frames, { timeout: 20_000 }).toBeGreaterThan(paused);
  });
});
