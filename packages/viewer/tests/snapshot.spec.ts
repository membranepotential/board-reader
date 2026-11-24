/**
 * `/snapshot` — the pipeline inspector.
 *
 * The page is a diagnostic surface for the reading algorithm, so the properties worth
 * pinning are about what it *shows*: that all four stages exist and start idle (nothing
 * hidden behind an opt-in), that a photo is read as it is, whole, the moment it is
 * chosen, that inspecting a square shows the ranked distribution rather than only the
 * argmax, and that a confirmed board reaches `/labels`. All of them need a real file
 * input and a real chessground — none of them is reachable from a unit test.
 *
 * The node is stubbed, so the reading asserted on is the reading this file states.
 */

import { expect, test, type Page } from '@playwright/test';
import { estimateReply, squareDist, stubIdleNode, stubJson, START_PLACEMENT } from './helpers/node';
import { checkerboardPng } from './helpers/png';

const STAGE_TITLES = ['Capture', 'Read', 'Impossible-board repair', 'Position output'];

/**
 * a8 is a black rook in the start position, and every square carries this same ranking
 * (see `squareDist`) — so the inspector's assertions hold wherever the click lands.
 */
const RANKING = ['r', 'q', 'k', 'empty'] as const;

const READING = estimateReply({ confidence: 0.88, square_dist: squareDist(RANKING) });

/** Load a photo the way a person does — through the file input, with real bytes. */
async function choosePhoto(page: Page): Promise<void> {
  await page.locator('#file').setInputFiles({
    name: 'board.png',
    mimeType: 'image/png',
    buffer: checkerboardPng(900, 700)
  });
  await expect(page.locator('.photo-frame img')).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await stubIdleNode(page);
});

test('the four pipeline stages all render, in order, and start idle', async ({ page }) => {
  await page.goto('/snapshot');

  const steps = page.locator('.step');
  await expect(steps).toHaveCount(STAGE_TITLES.length);
  await expect(steps.locator('summary > .name')).toHaveText(STAGE_TITLES);
  await expect(steps.locator('summary > .idx')).toHaveText(['1', '2', '3', '4']);
  await expect(steps.locator('summary > .chip')).toHaveText(
    Array(STAGE_TITLES.length).fill('idle')
  );

  // Full detail always exists — no opt-in diagnostics checkbox — and every body starts
  // collapsed, so the page opens as a summary rather than a wall.
  await expect(page.locator('.step details[open]')).toHaveCount(0);
  await expect(page.locator('.step details')).toHaveCount(STAGE_TITLES.length);
});

test('a chosen photo is read at once, whole, with nothing but the board id', async ({
  page
}) => {
  await stubJson(page, '**/api/estimate**', READING);
  const posted = page.waitForRequest(
    (request) => request.url().includes('/api/estimate') && request.method() === 'POST'
  );

  await page.goto('/snapshot');
  await choosePhoto(page);

  // No corners, no debug images, no rotation: the recognizer finds the board itself and
  // answers in the chess frame.
  const request = await posted;
  expect(new URL(request.url()).search).toBe('?board_id=snapshot');
  expect(request.headers()['content-type']).toBe('image/jpeg');

  await expect(page.locator('.board-col .board-wrap piece')).toHaveCount(32);
  // By position, not by text: "read" appears in more than one stage's copy.
  const capture = page.locator('.step').nth(0);
  await expect(capture.locator('summary > .chip')).toHaveText('ok');
  const read = page.locator('.step').nth(1);
  await expect(read.locator('summary > .chip')).not.toHaveText('idle');
  await expect(read.locator('summary > .sum')).toContainText('position confidence 0.880');
});

test('inspecting a square shows its ranked classes, not just the argmax', async ({ page }) => {
  await stubJson(page, '**/api/estimate**', READING);

  await page.goto('/snapshot');
  await choosePhoto(page);

  const board = page.locator('.board-col .board-wrap');
  await expect(board.locator('piece')).toHaveCount(32);

  // Click a square — opening one never changes it, so nothing on the board may move.
  const box = await board.boundingBox();
  expect(box).not.toBeNull();
  const rect = box as { width: number; height: number };
  await board.click({ position: { x: rect.width * 0.0625, y: rect.height * 0.0625 } });

  const inspector = page.locator('.inspector');
  await expect(inspector).toBeVisible();
  await expect(inspector.locator('.sq')).toHaveText('a8');
  await expect(inspector.locator('.facts')).toContainText('model read black rook (r) at 62.0%');
  await expect(inspector.locator('.facts')).toContainText('shown now black rook — as read');
  await expect(board.locator('piece')).toHaveCount(32);

  // The whole point of the panel: the distribution, ranked, with its own numbers.
  const rows = inspector.locator('.rank-row');
  await expect(rows).toHaveCount(5);
  await expect(rows.locator('.rk-cls')).toHaveText(['r', 'q', 'k', '.', /./]);
  const percentages = await rows.locator('.rk-pct').allInnerTexts();
  const numbers = percentages.map((text) => Number(text.replace(/[<%\s]/g, '')));
  expect(numbers).toEqual([...numbers].sort((a, b) => b - a));
  expect(numbers[0]).toBeCloseTo(62, 1);
});

test('saving the labelling posts the board as shown for the retained photo', async ({ page }) => {
  // A retaining node: the read names its photo, and `/uploads` lists it.
  await page.route('**/api/uploads**', (route) =>
    route.fulfill({
      json: {
        retention: true,
        uploads: [
          {
            id: 'u1',
            ts: 1_754_560_800,
            kind: 'estimate',
            placement: START_PLACEMENT,
            labeled: false,
            label_placement: null,
            label_corrections: null
          }
        ]
      }
    })
  );
  await stubJson(page, '**/api/estimate**', { ...READING, upload: 'u1' });
  await stubJson(page, '**/api/labels/u1', {
    saved: 'u1.label.json',
    placement: START_PLACEMENT,
    frame: 'chess',
    corrections: 0
  });

  await page.goto('/snapshot');
  await choosePhoto(page);
  await expect(page.locator('.board-col .board-wrap piece')).toHaveCount(32);

  const posted = page.waitForRequest(
    (request) => request.url().includes('/api/labels/u1') && request.method() === 'POST'
  );
  await page.getByRole('button', { name: 'Save labelling' }).click();
  // The board as shown, in the chess frame the read came in — nothing about corners.
  expect((await posted).postDataJSON()).toEqual({ placement: START_PLACEMENT, corrections: 0 });
  await expect(page.getByRole('button', { name: 'Labelling saved' })).toBeDisabled();
});
