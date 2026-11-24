/**
 * New reads must not reset what the user opened or scrolled on the live page: the game
 * panel's PGN expander and its move list's scroll position keep their state, and the
 * DOM nodes themselves survive (a keyed update, not a remount).
 */

import { expect, test } from '@playwright/test';
import { CLASSES, FakeViewerSocket, positionEvent, START_PLACEMENT } from './helpers/node';

const T0 = Date.parse('2026-11-24T19:41:05.000Z');

/** A read certain of `placement` on every square. */
function squareDistOf(placement: string): number[][][] {
  const p = 0.999;
  const rest = (1 - p) / (CLASSES.length - 1);
  return placement.split('/').map((rank) =>
    [...rank]
      .flatMap((ch) => (/[1-8]/.test(ch) ? Array<string>(Number(ch)).fill('empty') : [ch]))
      .map((name) => CLASSES.map((c) => (c === name ? p : rest)))
  );
}

function read(placement: string, seq: number): Record<string, unknown> {
  const utc = new Date(T0 + seq * 1000).toISOString();
  return {
    ...positionEvent({ board_id: 'alpha', seq, placement, square_dist: squareDistOf(placement) }),
    utc,
    changed_utc: utc
  };
}

/** Nf3 Nf6 Ng1 Ng8: four legal plies that return to the start, repeatable. */
const KNIGHT_CYCLE = [
  'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R',
  'rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R',
  'rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR',
  START_PLACEMENT
];

test('new reads keep the PGN expander open and the move list scrolled', async ({ page }) => {
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/');

  let seq = 1;
  const send = (placement: string) => {
    // Two reads per position, as the node sends while a board settles.
    socket.send(read(placement, seq++));
    socket.send(read(placement, seq++));
  };
  send(START_PLACEMENT);
  for (let round = 0; round < 10; round++) KNIGHT_CYCLE.forEach(send);

  const panel = page.locator('.game-panel');
  const moves = panel.locator('.moves');
  await expect.poll(() => moves.evaluate((el) => el.scrollHeight - el.clientHeight)).toBeGreaterThan(60);

  await panel.locator('details summary').click();
  await expect(panel.locator('details')).toHaveAttribute('open', '');
  await moves.evaluate((el) => (el.scrollTop = 40));
  expect(await moves.evaluate((el) => el.scrollTop)).toBe(40);
  await panel.locator('details').evaluate((el) => el.setAttribute('data-kept', ''));

  // More reads arrive, including a new move.
  send(KNIGHT_CYCLE[0]!);
  send(KNIGHT_CYCLE[0]!);
  await expect(panel.locator('.count')).toHaveText(/^41 plies/);

  await expect(panel.locator('details')).toHaveAttribute('data-kept', '');
  await expect(panel.locator('details')).toHaveAttribute('open', '');
  expect(await moves.evaluate((el) => el.scrollTop)).toBe(40);
});
