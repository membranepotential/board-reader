/**
 * Two phones and a control page against one node (`helpers/cameras.ts`): the node
 * assigns each phone its board id, keeps it across a reload and a node restart,
 * names it from /control without changing the id, and forgets a disconnected one. Each phone is its own browser context, so its own `localStorage`, so its
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

test('a rename on /control names the camera, keeps its board id and survives a restart', async ({
  browser,
  baseURL
}) => {
  const node = new FakeCameraNode();
  await open(browser, baseURL, node, '/camera');
  const phone = await open(browser, baseURL, node, '/camera');
  await expect(status(phone)).toContainText('board board-2');
  const control = await open(browser, baseURL, node, '/control');

  // By position: while it is edited, the row's name is an input's value, not its text.
  const row = rows(control).nth(1);
  await expect(row).toContainText('board-2');
  await row.getByRole('button', { name: 'Rename' }).click();
  await row.getByRole('textbox').fill('kitchen table');
  await row.getByRole('button', { name: 'Save' }).click();

  await expect(status(phone)).toContainText('kitchen table (board board-2)');
  await expect(rows(control).filter({ hasText: 'kitchen table' })).toHaveCount(1);
  await rows(control).filter({ hasText: 'kitchen table' }).getByRole('button', { name: 'kitchen table' }).click();
  await expect(control.locator('header')).toContainText('camera kitchen table · board board-2');
  // The frames still name the board id, and now the phone that sends them.
  await control.getByRole('button', { name: 'Start streaming' }).click();
  await expect.poll(() => node.headers.at(-1)?.['board_id'], { timeout: 8_000 }).toBe('board-2');
  expect(node.headers.at(-1)?.['device_id']).toBe([...node.boards].find(([, b]) => b === 'board-2')?.[0]);

  node.restart();
  await expect(status(phone)).toContainText('node connected');
  await expect(status(phone)).toContainText('kitchen table (board board-2)');

  // Another camera's name or board id is refused, and said on /control.
  const first = rows(control).nth(0);
  await first.getByRole('button', { name: 'Rename' }).click();
  await first.getByRole('textbox').fill('board-2');
  await first.getByRole('button', { name: 'Save' }).click();
  await expect(control.getByRole('alert')).toContainText('the node refused: board-2 is taken by camera kitchen table');
});

test('a disconnected camera can be forgotten after a confirm; a connected one cannot', async ({
  browser,
  baseURL
}) => {
  const node = new FakeCameraNode();
  const first = await open(browser, baseURL, node, '/camera');
  const second = await open(browser, baseURL, node, '/camera');
  await expect(status(second)).toContainText('board board-2');
  const control = await open(browser, baseURL, node, '/control');
  await expect(rows(control)).toHaveCount(2);

  const forgetFirst = rows(control).nth(0).getByRole('button', { name: 'Forget' });
  await expect(forgetFirst).toBeDisabled();
  await expect(control.getByText('A connected camera cannot be forgotten')).toBeVisible();

  await first.context().close();
  await expect(rows(control).nth(0)).toContainText('not connected');
  let asked = '';
  control.once('dialog', (dialog) => {
    asked = dialog.message();
    void dialog.accept();
  });
  await forgetFirst.click();
  await expect(rows(control)).toHaveCount(1);
  expect(asked).toContain('Forget camera board-1?');

  // Its id is free again: the next new phone gets it.
  const third = await open(browser, baseURL, node, '/camera');
  await expect(status(third)).toContainText('board board-1');
});

test('a name being typed survives the list the node pushes every second', async ({ browser, baseURL }) => {
  const node = new FakeCameraNode();
  await open(browser, baseURL, node, '/camera');
  const control = await open(browser, baseURL, node, '/control');
  const row = rows(control).nth(0);
  await row.getByRole('button', { name: 'Rename' }).click();
  const field = row.getByRole('textbox');
  await field.fill('kitch');
  node.tick();
  await control.waitForTimeout(300);
  await expect(field).toHaveValue('kitch');
  await field.fill('board-1');
  node.tick();
  await control.waitForTimeout(300);
  await expect(field).toBeVisible();
});
