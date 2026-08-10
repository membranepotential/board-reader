import { sveltekit } from '@sveltejs/kit/vite';
// From vitest, not vite: this config carries a `test` key, which only vitest's
// re-exported `defineConfig` types.
import { defineConfig } from 'vitest/config';

// Where `pnpm dev` sends /api and /ws during development: the real Python viewer,
// which relays them on to the inference node. Pointing at the viewer rather than the
// node directly means the dev server exercises the same relay production uses.
const VIEWER = process.env.BR_VIEWER_ORIGIN ?? 'http://127.0.0.1:8080';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': { target: VIEWER, changeOrigin: true },
      '/ws': { target: VIEWER, changeOrigin: true, ws: true }
    }
  },
  test: {
    // The library is plain maths and wire framing; it is tested in node, with the
    // handful of DOM-touching helpers isolated behind their own environment comment.
    environment: 'node',
    include: ['src/**/*.test.ts']
  }
});
