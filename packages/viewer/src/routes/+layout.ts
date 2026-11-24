// Every page is prerendered to a file (the `/api` routes opt out: they are the
// relay, see vite.config.ts). `ssr` stays on so prerendering produces real markup,
// but nothing may touch a browser API at module scope.
export const prerender = true;

// 'never' (the default, stated because the file layout depends on it): prerendering
// emits `snapshot.html` and `camera.html` beside `index.html`, which adapter-node
// serves at /snapshot and /camera — no directory indirection, no redirect.
export const trailingSlash = 'never';

