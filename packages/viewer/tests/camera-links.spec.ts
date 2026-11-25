/**
 * `GET /api/camera-links`, the addresses behind /control's camera QR code, answered
 * by the built viewer server rather than relayed to the node.
 *
 * Which links come back depends on the machine and on `--tls-port`, which this
 * suite's server does not pass, so the spec checks the contract: the route is the
 * viewer's own (not a 502 from the relay to the closed node port), the answer parses,
 * and every link opens `/camera` over https with no board in it. The choice of links
 * is Vitest's (`src/lib/components/control/camera-links.test.ts`).
 */

import { expect, test } from '@playwright/test';
import { cameraLinksSchema } from '../src/lib/components/control/camera-links.ts';

test('the camera links are the viewer’s own answer, https and board-free', async ({ request }) => {
  const response = await request.get('/api/camera-links');
  expect(response.status()).toBe(200);
  const { links } = cameraLinksSchema.parse(await response.json());
  for (const { url } of links) {
    const parsed = new URL(url);
    expect(parsed.protocol).toBe('https:');
    expect(parsed.pathname).toBe('/camera');
    expect(parsed.search).toBe('');
  }
});
