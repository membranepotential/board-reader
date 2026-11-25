/**
 * The addresses a phone can open `/camera` at, for the QR code on /control.
 *
 * The phone needs a secure context for its camera, so every link is https (or the
 * public URL the operator configured, which a tunnel serves over TLS). The control
 * page usually runs on this machine at `localhost`, which no phone reaches, so the
 * links name the machine as the LAN sees it. A link also has to be a name the
 * viewer's self-signed certificate covers, because a phone rejects a certificate that
 * does not name the address it opened before it offers to trust it.
 *
 * `cameraLinks` is pure (the route `/api/camera-links` feeds it the machine's state),
 * and the schema below is the one definition of the route's answer.
 */

import type { NetworkInterfaceInfo } from 'node:os';
import { z } from 'zod';

/** Where a link comes from, in the order they are offered. */
export const LINK_SOURCES = ['public', 'lan', 'hostname', 'loopback'] as const;
export type LinkSource = (typeof LINK_SOURCES)[number];

export const cameraLinksSchema = z.object({
  links: z.array(z.object({ url: z.url(), via: z.enum(LINK_SOURCES) }))
});

/** The answer of `GET /api/camera-links`: best link first, possibly none. */
export type CameraLinks = z.infer<typeof cameraLinksSchema>;
export type CameraLink = CameraLinks['links'][number];

/** `os.networkInterfaces()`'s shape, read-only. */
export type Interfaces = Readonly<Record<string, readonly NetworkInterfaceInfo[] | undefined>>;

/** The viewer's TLS port and the names its certificate covers (`server/tls.ts`). */
export interface TlsListener {
  readonly port: number;
  /** The certificate's names, routed LAN address before the others. */
  readonly names: readonly string[];
}

export interface LinkInputs {
  /** `BR_PUBLIC_URL`: an origin that serves this viewer over TLS, such as a tunnel. */
  readonly publicOrigin: string | undefined;
  /** `undefined` when the viewer serves no TLS port. */
  readonly tls: TlsListener | undefined;
  readonly interfaces: Interfaces;
  readonly hostname: string;
}

const CAMERA_PATH = '/camera';

/** Docker and libvirt bridges and veth pairs: addresses a phone cannot route to. */
const VIRTUAL_INTERFACE = /^(docker\d*|br-|veth|virbr)/;

const isLoopback = (address: string): boolean => address === 'localhost' || address.startsWith('127.');
const isLinkLocal = (address: string): boolean => address.startsWith('169.254.');

/** The IPv4 addresses on real, non-internal interfaces. */
function lanAddresses(interfaces: Interfaces): Set<string> {
  const found = Object.entries(interfaces)
    .filter(([name]) => !VIRTUAL_INTERFACE.test(name))
    .flatMap(([, entries]) => entries ?? [])
    .filter((entry) => entry.family === 'IPv4' && !entry.internal && !isLinkLocal(entry.address))
    .map((entry) => entry.address);
  return new Set(found);
}

const link = (origin: string, via: LinkSource): CameraLink => ({
  url: new URL(CAMERA_PATH, origin).href,
  via
});

/**
 * Every link a phone may open, best first: the public URL, then the LAN addresses
 * (the routed one first), then the hostname, then `localhost`, which only this
 * machine reaches. Without a TLS port only the public URL is a link at all.
 */
export function cameraLinks({ publicOrigin, tls, interfaces, hostname }: LinkInputs): CameraLink[] {
  const links = publicOrigin === undefined ? [] : [link(publicOrigin, 'public')];
  if (tls === undefined) return links;

  const https = (host: string): string => `https://${host}:${tls.port}`;
  const lan = lanAddresses(interfaces);
  const covered = (name: string): boolean => tls.names.includes(name);
  return [
    ...links,
    ...tls.names.filter((name) => lan.has(name)).map((ip) => link(https(ip), 'lan')),
    ...(covered(hostname) && !isLoopback(hostname) ? [link(https(hostname), 'hostname')] : []),
    ...(covered('localhost') ? [link(https('localhost'), 'loopback')] : [])
  ];
}
