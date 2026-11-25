/**
 * /control's list of the node's streams (`WS /streams`, DESIGN D22), against a fake
 * node: the Add stream modal lists each stream with its state, and Remove asks
 * first, then sends `remove_stream`. The node's side (the process, the forget) is the
 * Python suite's (`test_supervisor.py`, `test_streams_router.py`).
 */

import { expect, test } from '@playwright/test';

const RUNNING = {
  id: 'stream-1',
  url: 'https://www.youtube.com/watch?v=chessgame',
  start_s: 3600,
  state: 'running',
  device_id: 'stream-device-one',
  board_id: 'board-2',
  error: null
};
const CRASHED = {
  ...RUNNING,
  id: 'stream-2',
  url: 'https://video.test/broken',
  start_s: null,
  state: 'crashed',
  device_id: 'stream-device-two',
  board_id: null,
  error: 'exited with code 1: ERROR: Video unavailable'
};

test('the Add stream modal lists the streams, and Remove asks, then removes', async ({ page }) => {
  const asked: unknown[] = [];
  await page.routeWebSocket(/\/ws\/cameras$/, (ws) => {
    ws.send(JSON.stringify({ type: 'cameras', cameras: [] }));
  });
  await page.routeWebSocket(/\/ws\/streams$/, (ws) => {
    let streams = [RUNNING, CRASHED];
    ws.send(JSON.stringify({ type: 'streams', streams }));
    ws.onMessage((message) => {
      const parsed = JSON.parse(message as string) as { type: string; stream_id: string };
      asked.push(parsed);
      if (parsed.stream_id === 'stream-2') {
        ws.send(JSON.stringify({ type: 'refused', reason: 'no stream stream-2' }));
        return;
      }
      streams = streams.filter((stream) => stream.id !== parsed.stream_id);
      ws.send(JSON.stringify({ type: 'streams', streams }));
    });
  });
  await page.goto('/control');
  await page.getByRole('button', { name: 'Add stream' }).click();
  const list = page.getByRole('dialog', { name: 'Add a video or live stream' }).getByRole('region', {
    name: "the node's streams"
  });
  await expect(list).toContainText('from 1:00:00');
  await expect(list).toContainText('running as board-2');
  await expect(list).toContainText('crashed, starting again: exited with code 1: ERROR: Video unavailable');

  // Dismissed: nothing goes out.
  page.once('dialog', (dialog) => void dialog.dismiss());
  await list.getByRole('button', { name: `Remove ${RUNNING.url}` }).click();
  await expect(list.getByRole('listitem')).toHaveCount(2);
  expect(asked).toEqual([]);

  page.once('dialog', (dialog) => {
    expect(dialog.message()).toContain('forgets its camera board-2');
    void dialog.accept();
  });
  await list.getByRole('button', { name: `Remove ${RUNNING.url}` }).click();
  await expect(list.getByRole('listitem')).toHaveCount(1);
  expect(asked).toEqual([{ type: 'remove_stream', stream_id: 'stream-1' }]);

  page.once('dialog', (dialog) => void dialog.accept());
  await list.getByRole('button', { name: `Remove ${CRASHED.url}` }).click();
  await expect(list.getByRole('alert')).toContainText('the node refused: no stream stream-2');
});
