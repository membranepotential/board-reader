import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end tests for the three pages, driven in a real Chromium.
 *
 * These exist because of what the *other* two layers cannot reach. Vitest covers the
 * library — the tracker maths, the FEN sanitizer, the liveness arithmetic, the pipeline
 * model — as pure functions. The Python suite covers the server: which file each route
 * serves, that the relay matches before the file server, that no absolute origin ships.
 * Between them sits everything that is only true once a browser has laid the page out
 * and run it: a chessground board with no pieces on it, a rail button that is 31 px tall
 * on a phone, a `@media (orientation: portrait)` rule that never applies, a preview that
 * fetches while collapsed. Each of those has actually shipped here at least once, and
 * none of them is visible to a unit test.
 *
 * The app under test is the **built** app (`vite build`, then `vite preview`), not the
 * dev server: the bundle is what the Python viewer serves, so it is what gets driven.
 *
 * There is no inference node behind it. Every `/api/…` request and the `/ws/viewer`
 * socket are intercepted per test (`tests/helpers/node.ts`), so these specs need
 * neither a GPU nor a model, and a spec that asserts on a reading states the exact
 * reading it asserts on.
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
    // `localhost`, not 127.0.0.1: vite preview binds the hostname, and `localhost` is a
    // secure context, which is what /camera needs to be offered a camera at all.
    command: `pnpm exec vite build && pnpm exec vite preview --port ${PORT} --strictPort`,
    url: ORIGIN,
    reuseExistingServer: !process.env['CI'],
    timeout: 180_000,
    stdout: 'ignore',
    stderr: 'pipe'
  }
});
