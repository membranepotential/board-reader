import type { NetworkInterfaceInfo } from 'node:os';
import { describe, expect, it } from 'vitest';

import { cameraLinks, cameraLinksSchema, type Interfaces, type LinkInputs } from './camera-links.ts';

const ipv4 = (address: string, internal = false): NetworkInterfaceInfo => ({
  address,
  netmask: '255.255.255.0',
  family: 'IPv4',
  mac: '00:00:00:00:00:00',
  internal,
  cidr: `${address}/24`
});

/** A laptop on Wi-Fi with Docker running and an unconfigured second interface. */
const interfaces: Interfaces = {
  lo: [ipv4('127.0.0.1', true)],
  wlp82s0: [ipv4('192.168.1.10'), { ...ipv4('fe80::1'), family: 'IPv6', scopeid: 3 }],
  enp0s31f6: [ipv4('169.254.12.7')],
  docker0: [ipv4('172.17.0.1')],
  'br-3f2a': [ipv4('172.18.0.1')],
  eth1: [ipv4('10.0.0.5')]
};

/** What server/tls.ts would certify on that laptop: routed address after the fixed names. */
const tls = {
  port: 8443,
  names: ['localhost', '127.0.0.1', 'euclid', '10.0.0.5', '192.168.1.10', '172.17.0.1', '169.254.12.7']
};

const inputs = (over: Partial<LinkInputs> = {}): LinkInputs => ({
  publicOrigin: undefined,
  tls,
  interfaces,
  hostname: 'euclid',
  ...over
});

describe('cameraLinks', () => {
  it('offers the LAN addresses, then the hostname, then localhost, all on the TLS port', () => {
    expect(cameraLinks(inputs())).toEqual([
      { url: 'https://10.0.0.5:8443/camera', via: 'lan' },
      { url: 'https://192.168.1.10:8443/camera', via: 'lan' },
      { url: 'https://euclid:8443/camera', via: 'hostname' },
      { url: 'https://localhost:8443/camera', via: 'loopback' }
    ]);
  });

  it('puts the public URL first and keeps only its origin', () => {
    const links = cameraLinks(inputs({ publicOrigin: 'https://quiet-fox.trycloudflare.com' }));
    expect(links[0]).toEqual({ url: 'https://quiet-fox.trycloudflare.com/camera', via: 'public' });
    expect(links).toHaveLength(5);
  });

  it('skips an interface address the certificate does not name', () => {
    const tailnet = { ...interfaces, tailscale0: [ipv4('100.114.181.35')] };
    const urls = cameraLinks(inputs({ interfaces: tailnet })).map((l) => l.url);
    expect(urls).not.toContain('https://100.114.181.35:8443/camera');
  });

  it('offers no hostname that only names this machine', () => {
    expect(cameraLinks(inputs({ hostname: 'localhost' })).map((l) => l.via)).not.toContain('hostname');
  });

  it('offers only the public URL without a TLS port', () => {
    expect(cameraLinks(inputs({ tls: undefined }))).toEqual([]);
    expect(cameraLinks(inputs({ tls: undefined, publicOrigin: 'https://x.example' }))).toEqual([
      { url: 'https://x.example/camera', via: 'public' }
    ]);
  });

  it('produces what the route schema accepts', () => {
    expect(cameraLinksSchema.safeParse({ links: cameraLinks(inputs()) }).success).toBe(true);
  });
});
