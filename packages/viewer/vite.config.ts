import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
// From vitest, not vite: this config carries a `test` key, which only vitest's
// re-exported `defineConfig` types.
import { defineConfig } from 'vitest/config';

import { wsOrigin } from './server/ws.ts';

export default defineConfig({
  plugins: [
    sveltekit({
      preprocess: vitePreprocess(),
      /**
       * The viewer is a SvelteKit server. The pages are prerendered
       * (`src/routes/+layout.ts`); the HTTP endpoints are routes: `/api/[...path]`
       * relays to the inference node and `/api/client-log` keeps the phone's notes.
       * adapter-node's `handler` runs inside a custom server (`server/main.ts`) that
       * adds what SvelteKit cannot do: the `/ws/*` WebSocket relay and the TLS port
       * `/camera` needs.
       *
       * A static build stays possible for the on-phone app idea (docs/on-phone-app.md):
       * add `@sveltejs/adapter-static` and use it here with
       * `{ pages: 'build', assets: 'build', strict: false }`. `strict: false` lets it
       * skip the two `/api` routes, which cannot be prerendered; such an app reaches
       * its recognizer some other way than this relay.
       */
      adapter: adapter({ out: 'build', precompress: true }),
      // The viewer has no cookies and no session, so SvelteKit's cross-site form check
      // has nothing to protect, and it misfires here: with no reverse proxy in front,
      // adapter-node takes every request's origin to be https, so the phone's
      // text/plain posts to /api/client-log on the plain port would be refused.
      csrf: { trustedOrigins: ['*'] }
    })
  ],
  server: {
    // `pnpm dev` (hot reload) against a running inference node. `/api/*` needs
    // nothing here: it is a route. The sockets are relayed by the custom server in
    // production (server/ws.ts), which Vite does not run, so this proxy stands in for
    // it: `/ws/<path>` becomes the node's `/<path>`.
    proxy: {
      '/ws/': {
        target: wsOrigin(process.env),
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.slice('/ws'.length)
      }
    }
  },
  test: {
    // The library is plain maths and wire framing, and the server is Node; both are
    // tested in node, with the handful of DOM-touching helpers isolated behind their
    // own environment comment.
    environment: 'node',
    include: ['src/**/*.test.ts', 'server/**/*.test.ts']
  }
});
