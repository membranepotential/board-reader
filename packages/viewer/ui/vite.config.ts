import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
// From vitest, not vite: this config carries a `test` key, which only vitest's
// re-exported `defineConfig` types.
import { defineConfig } from 'vitest/config';

// Where `pnpm dev` sends /api and /ws during development: the real Python viewer,
// which relays them on to the inference node. Pointing at the viewer rather than the
// node directly means the dev server exercises the same relay production uses.
const VIEWER = process.env.BR_VIEWER_ORIGIN ?? 'http://127.0.0.1:8080';

export default defineConfig({
  plugins: [
    // SvelteKit 3 takes its configuration here; svelte.config.js is gone.
    sveltekit({
      preprocess: vitePreprocess(),
      // Kit 3 no longer generates `$lib`; its replacement is the `#lib` subpath
      // import. Every module under src/ still imports `$lib/...`, so the alias is
      // declared here until that rename lands as its own change.
      alias: { $lib: 'src/lib' },
      /**
       * The UI is prerendered to plain files and served by the Python viewer
       * (`board_reader_viewer.server`), which also relays the inference node's HTTP API
       * and sockets same-origin (`proxy.py`).
       *
       * `fallback` is deliberately unset — every route is prerendered (see
       * `src/routes/+layout.ts`), so a missing file is a build error here rather than a
       * blank page in someone's hand at the board.
       */
      adapter: adapter({
        // Straight into the Python package, which ships it as package data.
        pages: '../src/board_reader_viewer/web',
        assets: '../src/board_reader_viewer/web',
        precompress: false,
        strict: true
      })
    })
  ],
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
