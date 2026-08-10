import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/**
 * The UI is prerendered to plain files and served by the Python viewer
 * (`board_reader_viewer.server`), which also relays the inference node's HTTP API and
 * sockets same-origin (`proxy.py`). There is no Node runtime in the deployment: the
 * viewer stays a byte-forwarder that imports no `board_reader_*` package, and the
 * browser still sees exactly one origin and one certificate.
 *
 * `fallback` is deliberately unset — every route is prerendered (see
 * `src/routes/+layout.ts`), so a missing file is a build error here rather than a
 * blank page in someone's hand at the board.
 *
 * @type {import('@sveltejs/kit').Config}
 */
export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      // Straight into the Python package, which ships it as package data. Nothing
      // else writes here, so the directory is generated in full every build.
      pages: '../src/board_reader_viewer/web',
      assets: '../src/board_reader_viewer/web',
      precompress: false,
      strict: true
    })
  }
};
