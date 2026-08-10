/**
 * The live page (`/`) — one card per board, fed by `/ws/viewer`.
 *
 * What is pinned here is what only a laid-out, running page can answer: that the socket
 * is opened and its events become cards, that a card visibly *decays* once the stream
 * goes quiet while the one row that reports the decay stays legible, that the calibrate
 * button explains its own refusal, and that a collapsed preview costs the node nothing.
 * The arithmetic behind all of it (`$lib/stores/liveness`, `$lib/board/fen`) is Vitest's.
 */

import { expect, test } from '@playwright/test';
import { FakeViewerSocket, positionEvent, START_PLACEMENT } from './helpers/node';

/** `STALE_MS` in `$lib/stores/liveness`, plus room for the 200 ms ticker. */
const PAST_STALE_MS = 5000 + 1500;

test('the page opens the viewer socket and renders a card per board', async ({ page }) => {
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/');

  await expect(page.locator('.conn')).toHaveText('live');
  await expect(page.locator('.source code')).toHaveText(/^ws:\/\/localhost:\d+\/ws\/viewer$/);

  socket.send(positionEvent({ board_id: 'alpha' }));
  socket.send(positionEvent({ board_id: 'beta', seq: 2 }));

  await expect(page.locator('.board-card')).toHaveCount(2);
  await expect(page.locator('.board-card h2 > span').first()).toHaveText('board alpha');
  // The wire string is rendered as a real board, not just echoed into the meta row.
  await expect(page.locator('.board-card').first().locator('piece')).toHaveCount(32);
  await expect(page.locator('.board-card').first().locator('code.pl')).toHaveText(
    START_PLACEMENT
  );
});

/**
 * The one invariant `$lib/board/Board.svelte` exists to hold, and the one no assertion
 * about *rendered content* can catch: a position change must `set()` the live
 * chessground instance, never destroy and rebuild it.
 *
 * A rebuild renders a perfectly correct board, so piece counts and FEN text stay green
 * while `api.destroy()` leaks the `ResizeObserver` chessground bound to the surviving
 * host element — once per event, at up to `/camera`'s 12 fps burst rate. So this asserts
 * on *node identity* instead: tag the live `<cg-board>`, push a new placement, and
 * require the same DOM node to still be there wearing the tag.
 */
test('a new position updates the board in place instead of rebuilding it', async ({
  page
}) => {
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/');

  socket.send(positionEvent({ board_id: 'alpha' }));
  const board = page.locator('.board-card cg-board');
  await expect(board).toHaveCount(1);
  await board.evaluate((el) => el.setAttribute('data-identity', 'original'));

  // One pawn short of the start position, so the board genuinely has to repaint.
  socket.send(
    positionEvent({
      board_id: 'alpha',
      seq: 2,
      placement: 'rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    })
  );
  await expect(page.locator('.board-card piece')).toHaveCount(31);

  // Survives => same instance. A rebuild would have replaced the element and the
  // attribute with it.
  await expect(page.locator('.board-card cg-board')).toHaveAttribute(
    'data-identity',
    'original'
  );
});

test('a card that stops receiving events goes stale, and its .live row stays legible', async ({
  page
}) => {
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/');
  // Mid-move when the stream dies: the motion chip must not pulse "moving" forever.
  socket.send(positionEvent({ stability: 0.3 }));

  const card = page.locator('.board-card');
  await expect(card).toHaveCount(1);
  await expect(card).not.toHaveClass(/stale/);
  await expect(card.locator('.motion-chip')).toHaveText('moving');

  await expect(card).toHaveClass(/stale/, { timeout: PAST_STALE_MS });
  await expect(card.locator('.stale-chip')).toBeVisible();

  const filterOf = (selector: string) =>
    card.locator(selector).first().evaluate((el) => getComputedStyle(el).filter);

  // Stale = the reading stops asserting itself...
  await expect.poll(() => filterOf('.board-wrap')).not.toBe('none');
  await expect.poll(() => filterOf('.meta > div:not(.live)')).not.toBe('none');
  // ...except the row that says HOW stale, which is the one thing you want to read.
  expect(await filterOf('.meta > .live')).toBe('none');
  await expect(card.locator('.meta > .live')).toContainText('ago');

  expect(
    await card.locator('.motion-chip').evaluate((el) => getComputedStyle(el).animationName)
  ).toBe('none');
});

test('the calibrate button states why it is blocked, in its title', async ({ page }) => {
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/');

  const calibrate = page.locator('button.calib');

  socket.send(positionEvent({ seq: 1 }));
  await expect(calibrate).toBeEnabled();
  await expect(calibrate).toHaveAttribute('title', /rebuild the piece bank/);

  // Calibrating off a mid-motion frame poisons the bank; so does a non-ok reading.
  socket.send(positionEvent({ seq: 2, stability: 0.3 }));
  await expect(calibrate).toBeDisabled();
  await expect(calibrate).toHaveAttribute('title', /the board is moving/);

  socket.send(positionEvent({ seq: 3, status: 'needs_setup' }));
  await expect(calibrate).toBeDisabled();
  await expect(calibrate).toHaveAttribute('title', /"needs_setup"/);

  // Recovery is immediate — an arriving event un-blocks it without waiting a tick.
  socket.send(positionEvent({ seq: 4 }));
  await expect(calibrate).toBeEnabled();

  // ...and a stream that dies takes the button with it, saying so.
  await expect(calibrate).toBeDisabled({ timeout: PAST_STALE_MS });
  await expect(calibrate).toHaveAttribute('title', /the stream is stale/);
});

test('the frame preview fetches only while it is open', async ({ page }) => {
  const frameRequests: string[] = [];
  await page.route('**/api/boards/**/frame**', (route) => {
    frameRequests.push(route.request().url());
    // 404 until the node has ingested a frame: not an error, just a note in the pane.
    return route.fulfill({ status: 404, json: { error: 'no frame yet' } });
  });

  const socket = await FakeViewerSocket.install(page);
  await page.goto('/');
  socket.send(positionEvent({ seq: 7 }));
  await expect(page.locator('.board-card')).toHaveCount(1);

  // Several events go by with the pane collapsed; none of them costs the node a frame.
  socket.send(positionEvent({ seq: 8 }));
  socket.send(positionEvent({ seq: 9 }));
  await expect(page.locator('.board-card .meta code').first()).toHaveText('9');
  expect(frameRequests).toEqual([]);

  const tab = page.locator('.preview-tab');
  await expect(tab).toHaveAttribute('aria-expanded', 'false');
  await tab.click();
  await expect(tab).toHaveAttribute('aria-expanded', 'true');

  await expect.poll(() => frameRequests.length).toBeGreaterThan(0);
  expect(frameRequests[0]).toContain('seq=9');
  // A node with no frame leaves the pane explaining itself rather than broken.
  await expect(page.locator('.frame-note')).toBeVisible();

  // Closing it stops the traffic again, however many events arrive after.
  await tab.click();
  const taken = frameRequests.length;
  socket.send(positionEvent({ seq: 10 }));
  await expect(page.locator('.board-card .meta code').first()).toHaveText('10');
  expect(frameRequests).toHaveLength(taken);
});
