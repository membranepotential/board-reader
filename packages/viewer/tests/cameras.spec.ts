/**
 * Two phones and a control page against one node (`helpers/cameras.ts`): the node
 * assigns each phone its board id, keeps it across a reload and a node restart,
 * names it from /control without changing the id, and forgets a disconnected one.
 * /control shows one card per camera. Each phone is its own browser context, so its own `localStorage`, so its
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
const cards = (control: Page) => control.getByTestId('camera-card');

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
  await expect(cards(control)).toHaveCount(2);
  await expect(cards(control).nth(0).locator('h2')).toContainText('board-1');
  await expect(cards(control).nth(1).locator('h2')).toContainText('board-2');
  await expect(cards(control).nth(1).locator('h2')).toContainText('connected');
  // No camera table or picker any more: the cards are the list.
  await expect(control.locator('table')).toHaveCount(0);
  await expect(control.getByRole('button', { name: 'Rename' })).toHaveCount(0);
});

test("a rename in a camera's card names the camera, keeps its board id and survives a restart", async ({
  browser,
  baseURL
}) => {
  const node = new FakeCameraNode();
  await open(browser, baseURL, node, '/camera');
  const phone = await open(browser, baseURL, node, '/camera');
  await expect(status(phone)).toContainText('board board-2');
  const control = await open(browser, baseURL, node, '/control');

  const card = cards(control).nth(1);
  await expect(card.locator('h2')).toContainText('board-2');
  await card.getByLabel('name').fill('kitchen table');
  await card.getByRole('button', { name: 'Save name' }).click();

  await expect(status(phone)).toContainText('kitchen table (board board-2)');
  await expect(card.locator('h2')).toContainText('kitchen table');
  await expect(card.locator('h2')).toContainText('board-2');
  await expect(card.getByRole('button', { name: 'Save name' })).toBeDisabled();
  // The frames still name the board id, and now the phone that sends them.
  await card.getByRole('button', { name: 'Start streaming' }).click();
  await expect.poll(() => node.headers.at(-1)?.['board_id'], { timeout: 8_000 }).toBe('board-2');
  expect(node.headers.at(-1)?.['device_id']).toBe([...node.boards].find(([, b]) => b === 'board-2')?.[0]);

  node.restart();
  await expect(status(phone)).toContainText('node connected');
  await expect(status(phone)).toContainText('kitchen table (board board-2)');

  // Another camera's name or board id is refused, and said on /control.
  const first = cards(control).nth(0);
  await first.getByLabel('name').fill('board-2');
  await first.getByLabel('name').press('Enter');
  await expect(control.getByRole('alert')).toContainText('the node refused: board-2 is taken by camera kitchen table');
  // The refused name stays in the field to be corrected.
  await expect(first.getByLabel('name')).toHaveValue('board-2');
});

test('a name being typed survives the list the node pushes every second', async ({ browser, baseURL }) => {
  const node = new FakeCameraNode();
  await open(browser, baseURL, node, '/camera');
  const control = await open(browser, baseURL, node, '/control');
  const field = cards(control).nth(0).getByLabel('name');
  await expect(field).toHaveValue('');

  await field.fill('kitch');
  node.tick();
  await control.waitForTimeout(300);
  await expect(field).toHaveValue('kitch');
  await field.pressSequentially('en');
  node.tick();
  await control.waitForTimeout(300);
  await expect(field).toHaveValue('kitchen');

  // Once the node has the name, the field shows it, and the next push keeps it.
  await field.press('Enter');
  await expect(cards(control).nth(0).locator('h2')).toContainText('kitchen');
  node.tick();
  await control.waitForTimeout(300);
  await expect(field).toHaveValue('kitchen');
});

test('a disconnected camera can be forgotten from its card after a confirm; a connected one cannot', async ({
  browser,
  baseURL
}) => {
  const node = new FakeCameraNode();
  const first = await open(browser, baseURL, node, '/camera');
  const second = await open(browser, baseURL, node, '/camera');
  await expect(status(second)).toContainText('board board-2');
  const control = await open(browser, baseURL, node, '/control');
  await expect(cards(control)).toHaveCount(2);

  const forgetFirst = cards(control).nth(0).getByRole('button', { name: 'Forget' });
  await expect(forgetFirst).toHaveCount(0);

  await first.context().close();
  await expect(cards(control).nth(0).locator('h2')).toContainText('not connected');
  let asked = '';
  control.once('dialog', (dialog) => {
    asked = dialog.message();
    void dialog.accept();
  });
  await forgetFirst.click();
  await expect(cards(control)).toHaveCount(1);
  expect(asked).toContain('Forget camera board-1?');

  // Its id is free again: the next new phone gets it.
  const third = await open(browser, baseURL, node, '/camera');
  await expect(status(third)).toContainText('board board-1');
});
