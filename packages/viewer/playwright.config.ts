import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end tests for the three pages, driven in a real Chromium.
 *
 * These exist because of what the *other* two layers cannot reach. Vitest covers the
 * library — the tracker maths, the FEN sanitizer, the liveness arithmetic, the pipeline
 * model — as pure functions, and the viewer server (`server/*.test.ts`): the relay, the
 * dispatch, the certificate. Between them sits everything that is only true once a browser has laid the page out
 * and run it: a chessground board with no pieces on it, a rail button that is 31 px tall
 * on a phone, a `@media (orientation: portrait)` rule that never applies, a preview that
 * fetches while collapsed. Each of those has actually shipped here at least once, and
 * none of them is visible to a unit test.
 *
 * The app under test is the **built** app served by the real viewer server
 * (`vite build`, then `node server/main.ts`), not the dev server, so what is driven is
 * what ships.
 *
 * There is no inference node behind it. Every `/api/…` request and every `/ws/…` socket
 * is intercepted per test (`tests/helpers/node.ts`), so these specs need neither a GPU
 * nor a model, and a spec that asserts on a reading states the exact reading it
 * asserts on. The server is pointed at a closed port, so a request a spec forgot to
 * intercept fails fast with the relay's 502 rather than reaching a dev node that
 * happens to be running.
 */

const PORT = 4173;
const ORIGIN = `http://localhost:${PORT}`;

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: Boolean(process.env['CI']),
  retries: process.env['CI'] ? 1 : 0,
  reporter: process.env['CI'] ? [['github'], ['list']] : [['list']],
  // The staleness spec deliberately waits out a 5 s threshold; everything else is fast.
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: ORIGIN,
    trace: 'retain-on-failure'
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // The camera page needs a camera. `--use-fake-device-for-media-stream` supplies
        // the frames; `--use-fake-ui-for-media-stream` auto-answers the permission
        // prompt, and without it the `getUserMedia` promise simply never settles in
        // headless Chromium — the page hangs in `requesting` with nothing to assert on.
        launchOptions: {
          args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream']
        },
        permissions: ['camera']
      }
    }
  ],
  webServer: {
    // The pages are opened at `localhost`, a secure context, which is what /camera
    // needs to be offered a camera at all.
    command: `pnpm exec vite build && node server/main.ts --host 127.0.0.1 --port ${PORT}`,
    env: {
      BR_INFERENCE_HTTP: 'http://127.0.0.1:1',
      BR_INFERENCE_WS: 'ws://127.0.0.1:1/viewer',
      BR_CLIENT_LOG_DIR: ''
    },
    url: ORIGIN,
    reuseExistingServer: !process.env['CI'],
    timeout: 180_000,
    stdout: 'ignore',
    stderr: 'pipe'
  }
});
