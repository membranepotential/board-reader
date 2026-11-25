/**
 * Two phones and a control page against one node (`helpers/cameras.ts`): the node
 * assigns each phone its board id, keeps it across a reload and a node restart,
 * renames it from /control, and refuses a phone that asks for an id another camera
 * holds. Each phone is its own browser context, so its own `localStorage`, so its
 * own device id, exactly as two phones are.
 */

import { expect, test, type Browser, type Page } from '@playwright/test';
import { FakeCameraNode } from './helpers/cameras';
import { LANDSCAPE, drawnBoardCamera, liveViewfinder } from './helpers/camera';

test.use({ viewport: LANDSCAPE });

async function open(browser: Browser, baseURL: string | undefined, node: FakeCameraNode, path: string): Promise<Page> {
  const context = await browser.newContext({ ...(baseURL === undefined ? {} : { baseURL }), viewport: LANDSCAPE });
  await node.attach(context);
  const page = await context.newPage();
  if (path.startsWith('/camera')) await drawnBoardCamera(page);
  await page.goto(path);
  if (path.startsWith('/camera')) await liveViewfinder(page);
  return page;
}

const status = (phone: Page) => phone.getByTestId('camera-status');
const rows = (control: Page) => control.getByTestId('camera-row');

test('each phone gets its own board, and keeps it across a reload', async ({ browser, baseURL }) => {
  const node = new FakeCameraNode();
  const first = await open(browser, baseURL, node, '/camera');
  await expect(status(first)).toContainText('board board-1');
  const second = await open(browser, baseURL, node, '/camera');
  await expect(status(second)).toContainText('board board-2');

  await first.reload();
  await liveViewfinder(first);
  await expect(status(first)).toContainText('board board-1');

  const control = await open(browser, baseURL, node, '/control');
  await expect(rows(control)).toHaveCount(2);
  await expect(rows(control).nth(0)).toContainText('board-1');
  await expect(rows(control).nth(1)).toContainText('board-2');
  await expect(rows(control).nth(1)).toContainText('connected');
});

test('a rename on /control reaches the phone at once and survives a reconnect and a restart', async ({
  browser,
  baseURL
}) => {
  const node = new FakeCameraNode();
  await open(browser, baseURL, node, '/camera');
  const phone = await open(browser, baseURL, node, '/camera');
  await expect(status(phone)).toContainText('board board-2');
  const control = await open(browser, baseURL, node, '/control');

  // By position: while it is edited, the row's id is an input's value, not its text.
  const row = rows(control).nth(1);
  await expect(row).toContainText('board-2');
  await row.getByRole('button', { name: 'Rename' }).click();
  await row.getByRole('textbox').fill('left');
  await row.getByRole('button', { name: 'Save' }).click();

  await expect(status(phone)).toContainText('board left');
  await expect(rows(control).filter({ hasText: 'left' })).toHaveCount(1);
  // Picked on /control, the renamed camera's controls talk to its new board.
  await rows(control).filter({ hasText: 'left' }).getByRole('button', { name: 'left' }).click();
  await expect(control.locator('header')).toContainText('board left');
  await control.getByRole('button', { name: 'Start streaming' }).click();
  await expect.poll(() => node.headers.at(-1)?.['board_id'], { timeout: 8_000 }).toBe('left');

  await phone.reload();
  await liveViewfinder(phone);
  await expect(status(phone)).toContainText('board left');

  node.restart();
  await expect(status(phone)).toContainText('node connected');
  await expect(status(phone)).toContainText('board left');
  await expect(rows(control).filter({ hasText: 'left' })).toContainText('connected');
});

test('a phone asking for a board another camera holds is refused, and both pages say so', async ({
  browser,
  baseURL
}) => {
  const node = new FakeCameraNode();
  await open(browser, baseURL, node, '/camera');
  const second = await open(browser, baseURL, node, '/camera?board=board-1');

  await expect(status(second)).toContainText('board board-2');
  await expect(status(second)).toContainText('board-1 is taken by camera');
  const control = await open(browser, baseURL, node, '/control');
  await expect(rows(control).filter({ hasText: 'board-2' }).locator('[data-warning="refused-board"]')).toContainText(
    'asked for board-1'
  );

  // Renaming to an id another camera holds is refused on /control as well.
  const row = rows(control).nth(1);
  await row.getByRole('button', { name: 'Rename' }).click();
  await row.getByRole('textbox').fill('board-1');
  await row.getByRole('button', { name: 'Save' }).click();
  await expect(row.getByRole('alert')).toContainText('not renamed: board-1 is taken by camera');
  await expect(status(second)).toContainText('board board-2');
});
