/**
 * `/snapshot` — the pipeline inspector.
 *
 * The page is a diagnostic surface for the reading algorithm, so the properties worth
 * pinning are about what it *shows*: that all eight stages exist and start idle (nothing
 * hidden behind an opt-in), that the geometry a stage reports really follows the corners
 * the operator placed, and that inspecting a square shows the ranked distribution rather
 * than only the argmax. All three need a laid-out canvas, a real file input and a real
 * chessground — none of them is reachable from a unit test.
 *
 * The node is stubbed, so the reading asserted on is the reading this file states.
 */

import { expect, test, type Page } from '@playwright/test';
import { squareDist, stubIdleNode, stubJson, START_PLACEMENT } from './helpers/node';
import { checkerboardPng } from './helpers/png';

const STAGE_TITLES = [
  'Capture',
  'Board detection',
  'Rectify & geometry',
  'Square sampling',
  'Classification',
  'Orientation',
  'Impossible-board repair',
  'Position output'
];

/** Fractions of the photo the four corners are tapped at, in a8, h8, h1, a1 order. */
const CORNERS = [
  [0.12, 0.12],
  [0.88, 0.14],
  [0.86, 0.88],
  [0.14, 0.86]
] as const;

/** The same board framed visibly narrower — a different quad, so a different geometry. */
const NARROWER = [
  [0.32, 0.12],
  [0.68, 0.14],
  [0.66, 0.88],
  [0.34, 0.86]
] as const;

/**
 * a8 is a black rook in the start position, and every square carries this same ranking
 * (see `squareDist`) — so the inspector's assertions hold wherever the click lands.
 */
const RANKING = ['r', 'q', 'k', 'empty'] as const;

const READING = {
  board_id: 'snapshot',
  seq: 1,
  utc: '2026-08-07T12:00:00.000Z',
  placement: START_PLACEMENT,
  confidence: 0.88,
  stability: 1,
  status: 'ok',
  orientation: 'white',
  clock: { running: 'none' },
  autorotated: 0,
  warnings: [],
  square_dist: squareDist(RANKING)
};

/** Load a photo the way a person does — through the file input, with real bytes. */
async function choosePhoto(page: Page): Promise<void> {
  await page.locator('#file').setInputFiles({
    name: 'board.png',
    mimeType: 'image/png',
    buffer: checkerboardPng(900, 700)
  });
  await expect(page.locator('.photo-frame canvas')).toBeVisible();
}

async function placeCorners(
  page: Page,
  corners: readonly (readonly [number, number])[] = CORNERS
): Promise<void> {
  const canvas = page.locator('.corner-stage canvas');
  const handles = page.locator('.corner-handle');
  const placed = await handles.count();
  for (const [index, [fx, fy]] of corners.entries()) {
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    const { width, height } = box as { width: number; height: number };
    // Each tap is its own gesture, and the handle it creates has to appear before the
    // next one is aimed: placing a corner re-flows the column around the canvas.
    await canvas.click({ position: { x: width * fx, y: height * fy } });
    await expect(handles).toHaveCount(placed + index + 1);
  }
}

/**
 * Fine-tune one already-placed corner. Deliberately a pointer gesture rather than a
 * synthesised event: the handle takes pointer capture on `pointerdown`, which is what
 * keeps a drag that wanders off the handle attached to it.
 */
async function dragHandle(page: Page, index: number, dx: number, dy: number): Promise<void> {
  const handle = page.locator('.corner-handle').nth(index);
  const box = await handle.boundingBox();
  expect(box).not.toBeNull();
  const { x, y, width, height } = box as { x: number; y: number; width: number; height: number };
  const [cx, cy] = [x + width / 2, y + height / 2];
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + dx, cy + dy, { steps: 8 });
  await page.mouse.up();
}

test.beforeEach(async ({ page }) => {
  await stubIdleNode(page);
  // Detection off: these specs are about what the page does with corners a HUMAN
  // placed, and an auto-detected quad would also auto-run the read out from under them.
  await stubJson(page, '**/api/detect**', { error: 'no board found in this photo' }, 422);
});

test('the eight pipeline stages all render, in order, and start idle', async ({ page }) => {
  await page.goto('/snapshot');

  const steps = page.locator('.step');
  await expect(steps).toHaveCount(STAGE_TITLES.length);
  await expect(steps.locator('summary > .name')).toHaveText(STAGE_TITLES);
  await expect(steps.locator('summary > .idx')).toHaveText(['1', '2', '3', '4', '5', '6', '7', '8']);
  await expect(steps.locator('summary > .chip')).toHaveText(
    Array(STAGE_TITLES.length).fill('idle')
  );

  // Full detail always exists — no opt-in diagnostics checkbox — and every body starts
  // collapsed, so the page opens as a summary rather than a wall.
  await expect(page.locator('.step details[open]')).toHaveCount(0);
  await expect(page.locator('.step details')).toHaveCount(STAGE_TITLES.length);
});

test('placing the corners updates the geometry the rectify stage reports', async ({ page }) => {
  await page.goto('/snapshot');
  await choosePhoto(page);

  // A photo alone is not corners: capture is done, geometry has nothing to measure yet.
  const capture = page.locator('.step', { hasText: 'Capture' }).first();
  await expect(capture.locator('summary > .chip')).toHaveText('ok');
  const rectify = page.locator('.step', { hasText: 'Rectify & geometry' }).first();
  await expect(rectify.locator('summary > .chip')).toHaveText('idle');
  await expect(rectify.locator('summary > .sum')).toHaveText('waiting for corners');

  await placeCorners(page);

  // Four corners = a homography = a measurable board, reported in the stage headline.
  const headline = rectify.locator('summary > .sum');
  await expect(rectify.locator('summary > .chip')).not.toHaveText('idle');
  await expect(headline).toContainText(/aspect \d/);

  // The readout FOLLOWS the corners rather than describing the photo. Fine-tuning one
  // handle is enough to move it — this is the gesture the whole page is built around.
  const framed = await headline.innerText();
  await dragHandle(page, 0, 90, 0);
  await expect(headline).not.toHaveText(framed);

  // The same measurement, spelled out in the stage's table in the photo's own pixels.
  await rectify.locator('summary').click();
  const rows = rectify.locator('table.kv tr');
  await expect(rows.filter({ hasText: 'projected aspect' })).toHaveCount(1);
  const widthRow = rows.filter({ hasText: 'mean edge width' }).locator('td');
  await expect(widthRow).toHaveText(/^\d+(\.\d+)? px \(a8-h8 /);
  const wide = Number((await widthRow.innerText()).split(' ')[0]);

  // ...and it follows a board framed again from scratch, too.
  await page.getByRole('button', { name: 'Reset corners' }).click();
  await expect(page.locator('.corner-handle')).toHaveCount(0);
  await expect(headline).toHaveText('waiting for corners');
  await placeCorners(page, NARROWER);

  await expect
    .poll(async () => Number((await widthRow.innerText()).split(' ')[0]))
    .toBeLessThan(wide * 0.75);
});

test('inspecting a square shows its ranked classes, not just the argmax', async ({ page }) => {
  await stubJson(page, '**/api/estimate**', READING);

  await page.goto('/snapshot');
  await choosePhoto(page);
  await placeCorners(page);

  await page.getByRole('button', { name: 'Read position' }).click();

  const board = page.locator('.board-col .board-wrap');
  await expect(board.locator('piece')).toHaveCount(32);
  const classify = page.locator('.step', { hasText: 'Classification' }).first();
  await expect(classify.locator('summary > .chip')).not.toHaveText('idle');

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
