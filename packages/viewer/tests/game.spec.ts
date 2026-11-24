/**
 * The live page's game panel (`#lib/components/game/GamePanel.svelte`): a scripted run
 * of reads through the fake `/ws/viewer` socket becomes a PGN, which the page shows,
 * copies and links to Lichess.
 *
 * The decoding itself (noise, missed plies, takebacks, promotions) is Vitest's
 * (`src/lib/game/*.test.ts`); this pins what only the running page shows: the panel
 * beside each board, its move list, the clipboard and the link.
 */

import { expect, test } from '@playwright/test';
import { CLASSES, FakeViewerSocket, positionEvent, START_PLACEMENT } from './helpers/node';

const T0 = Date.parse('2026-11-24T19:41:05.000Z');

/** A read certain of `placement` on every square. */
function squareDistOf(placement: string): number[][][] {
  const p = 0.999;
  const rest = (1 - p) / (CLASSES.length - 1);
  return placement.split('/').map((rank) =>
    [...rank].flatMap((ch) => (/[1-8]/.test(ch) ? Array<string>(Number(ch)).fill('empty') : [ch])).map((name) =>
      CLASSES.map((c) => (c === name ? p : rest))
    )
  );
}

/** One `ok` event: the read shows `placement`, which the node has settled on. */
function read(placement: string, seconds: number, seq: number): Record<string, unknown> {
  const utc = new Date(T0 + seconds * 1000).toISOString();
  // `EventOverrides` has no times, so they go on top of the finished event.
  return { ...positionEvent({ board_id: 'alpha', seq, placement, square_dist: squareDistOf(placement) }), utc, changed_utc: utc };
}

const AFTER_E4 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR';
const AFTER_E5 = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR';
const AFTER_NF3 = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R';

test('the reads become a PGN the page shows, copies and opens in Lichess', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const socket = await FakeViewerSocket.install(page);
  await page.goto('/');

  const script = [START_PLACEMENT, AFTER_E4, AFTER_E4, AFTER_E5, AFTER_E5, AFTER_NF3, AFTER_NF3, AFTER_NF3];
  script.forEach((placement, i) => socket.send(read(placement, 2 * i, i + 1)));

  const panel = page.locator('.game-panel');
  await expect(panel.locator('.moves li')).toHaveText([/^1\.\s*e4\s*e5$/, /^2\.\s*Nf3$/]);
  await expect(panel.locator('.count')).toHaveText('3 plies');

  const pgn = panel.locator('.pgn');
  await expect(pgn).toContainText('[UTCDate "2026.11.24"]');
  await expect(pgn).toContainText('[UTCTime "19:41:05"]');
  await expect(pgn).toContainText('1. e4 { [%emt 0:00:02] } e5 { [%emt 0:00:04] } 2. Nf3 { [%emt 0:00:04] } *');

  await expect(panel.locator('a.lichess')).toHaveAttribute('href', 'https://lichess.org/analysis/pgn/1._e4_e5_2._Nf3_*');
  await expect(panel.locator('a.lichess')).toHaveAttribute('target', '_blank');

  await panel.locator('button.copy').click();
  await expect(panel.locator('button.copy')).toHaveText('Copied');
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toBe(await pgn.textContent());

  await panel.locator('button.new-game').click();
  await expect(panel.locator('.empty')).toBeVisible();
  await expect(panel.locator('a.lichess')).toHaveAttribute('href', 'https://lichess.org/analysis/pgn/*');
});
