/**
 * `GET /api/camera-links` — the URLs a phone can open `/camera` at, best first, for
 * the QR code on /control (`CameraQr.svelte`).
 *
 * Answers `{"links": [{"url": "https://192.168.1.10:8443/camera", "via": "lan"}, …]}`
 * (`CameraLinks` in `#lib/components/control/camera-links.ts`, which also chooses
 * them). The machine's addresses are read per request, so a laptop that changed
 * networks gets its new address. This is the viewer's own route, more specific than
 * `/api/[...path]`, so it wins over the relay to the node.
 */

import { BR_PUBLIC_URL, BR_VIEWER_TLS_NAMES, BR_VIEWER_TLS_PORT } from '$app/env/private';
import os from 'node:os';

import { cameraLinks, type CameraLinks } from '#lib/components/control/camera-links.ts';

import type { RequestHandler } from './$types';

export const prerender = false;

export const GET: RequestHandler = () => {
  const tls =
    BR_VIEWER_TLS_PORT === undefined ? undefined : { port: BR_VIEWER_TLS_PORT, names: BR_VIEWER_TLS_NAMES };
  const links = cameraLinks({
    publicOrigin: BR_PUBLIC_URL,
    tls,
    interfaces: os.networkInterfaces(),
    hostname: os.hostname()
  });
  return Response.json({ links } satisfies CameraLinks);
};
