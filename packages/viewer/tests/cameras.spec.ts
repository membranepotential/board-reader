/**
 * Two phones and a control page against one node (`helpers/cameras.ts`): each phone
 * streams under its device id as its board id, gets a default name (`board-N`), keeps
 * both across a reload and a node restart, is renamed from /control without changing
 * the id, and a disconnected one is forgotten.
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
/** The device id (= board id) of the camera the node calls `name`. */
const deviceCalled = (node: FakeCameraNode, name: string) => [...node.names].find(([, n]) => n === name)![0];

test('each phone gets its own board and default name, and keeps them across a reload', async ({ browser, baseURL }) => {
  const node = new FakeCameraNode();
  const first = await open(browser, baseURL, node, '/camera');
  await expect(status(first)).toContainText('board-1');
  const second = await open(browser, baseURL, node, '/camera');
  await expect(status(second)).toContainText('board-2');

  await first.reload();
  await liveViewfinder(first);
  await expect(status(first)).toContainText('board-1');

  const control = await open(browser, baseURL, node, '/control');
  await expect(cards(control)).toHaveCount(2);
  // The name leads; the camera's id shows as small text, whole in the stats.
  const id = deviceCalled(node, 'board-1');
  await expect(cards(control).nth(0).locator('h2')).toContainText(`board-1 ${id.slice(0, 8)}`);
  await expect(cards(control).nth(0).getByTestId('camera-id')).toHaveText(id);
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
  await expect(status(phone)).toContainText('board-2');
  const id = deviceCalled(node, 'board-2');
  const control = await open(browser, baseURL, node, '/control');

  const card = cards(control).nth(1);
  await expect(card.locator('h2')).toContainText('board-2');
  await card.getByLabel('name').fill('kitchen table');
  await card.getByRole('button', { name: 'Save name' }).click();

  await expect(status(phone)).toContainText('kitchen table');
  await expect(card.locator('h2')).toContainText(`kitchen table ${id.slice(0, 8)}`);
  await expect(card.getByRole('button', { name: 'Save name' })).toBeDisabled();
  // The frames still name the board id, which is the phone's device id.
  await card.getByRole('button', { name: 'Start streaming' }).click();
  await expect.poll(() => node.headers.at(-1)?.['board_id'], { timeout: 8_000 }).toBe(id);
  expect(node.headers.at(-1)?.['device_id']).toBe(id);

  node.restart();
  await expect(status(phone)).toContainText('node connected');
  await expect(status(phone)).toContainText('kitchen table');

  // Another camera's name is refused, and said on /control.
  const first = cards(control).nth(0);
  await first.getByLabel('name').fill('kitchen table');
  await first.getByLabel('name').press('Enter');
  await expect(control.getByRole('alert')).toContainText(
    'the node refused: kitchen table is taken by camera kitchen table'
  );
  // The refused name stays in the field to be corrected.
  await expect(first.getByLabel('name')).toHaveValue('kitchen table');
});

test('a name being typed survives the list the node pushes every second', async ({ browser, baseURL }) => {
  const node = new FakeCameraNode();
  await open(browser, baseURL, node, '/camera');
  const control = await open(browser, baseURL, node, '/control');
  const field = cards(control).nth(0).getByLabel('name');
  await expect(field).toHaveValue('board-1');

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
  await expect(status(second)).toContainText('board-2');
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

  // Its name is free again: the next new phone is called board-1.
  const third = await open(browser, baseURL, node, '/camera');
  await expect(status(third)).toContainText('board-1');
});
