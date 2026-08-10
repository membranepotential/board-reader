// Every route is prerendered to a file the Python viewer serves; there is no Node
// runtime behind this app (see svelte.config.js). `ssr` stays on so prerendering
// produces real markup, but nothing may touch a browser API at module scope.
export const prerender = true;

// 'never' (the default, stated because the file layout depends on it): prerendering
// emits `snapshot.html` and `camera.html` beside `index.html`, which is exactly what
// server.py serves at /snapshot and /camera — no directory indirection, no redirect.
export const trailingSlash = 'never';

