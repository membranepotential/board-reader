import { X509Certificate } from 'node:crypto';
import fs from 'node:fs';
import https from 'node:https';
import type { AddressInfo } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import tls from 'node:tls';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { cacheDir, ensureCertificate, subjectNames } from './tls.ts';

let dir: string;
beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'viewer-tls-'));
});
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

const NAMES = ['localhost', '127.0.0.1', 'box', '192.168.1.7'] as const;

describe('a minted certificate', () => {
  it('names every address, is no CA, and carries serverAuth (iOS refuses one without)', async () => {
    const cert = await ensureCertificate(dir, NAMES);
    const parsed = new X509Certificate(cert.cert);
    expect(cert.generated).toBe(true);
    expect(parsed.subjectAltName).toBe('DNS:localhost, IP Address:127.0.0.1, DNS:box, IP Address:192.168.1.7');
    expect(parsed.ca).toBe(false);
    expect(parsed.keyUsage).toEqual(['1.3.6.1.5.5.7.3.1']);
    expect(parsed.subject).toContain('CN=localhost');
    expect(parsed.verify(parsed.publicKey)).toBe(true); // self-signed, and the signature holds
  });

  it('is valid from an hour ago for a year', async () => {
    const now = new Date('2026-11-24T12:00:00Z');
    const parsed = new X509Certificate((await ensureCertificate(dir, NAMES, now)).cert);
    expect(parsed.validFromDate).toEqual(new Date('2026-11-24T11:00:00Z'));
    expect(parsed.validToDate).toEqual(new Date('2027-11-24T12:00:00Z'));
  });

  it('keeps its private key readable by the owner only', async () => {
    const cert = await ensureCertificate(dir, NAMES);
    expect(fs.statSync(cert.keyfile).mode & 0o777).toBe(0o600);
  });

  it('serves a TLS handshake that a client trusting it accepts for an IP and a name', async () => {
    const cert = await ensureCertificate(dir, NAMES);
    const server = https.createServer({ cert: cert.cert, key: cert.key }, (_req, res) => res.end('ok'));
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as AddressInfo).port;
    try {
      for (const servername of ['localhost', '127.0.0.1']) {
        const authorized = await new Promise<boolean>((resolve, reject) => {
          const socket = tls.connect(
            { host: '127.0.0.1', port, ca: cert.cert, ...(servername === 'localhost' ? { servername } : {}) },
            () => {
              resolve(socket.authorized);
              socket.end();
            }
          );
          socket.once('error', reject);
        });
        expect(authorized).toBe(true);
      }
    } finally {
      server.close();
    }
  });
});

describe('the cache', () => {
  it('reuses a pair that still covers this host', async () => {
    const first = await ensureCertificate(dir, NAMES);
    const second = await ensureCertificate(dir, NAMES.slice(0, 2) as unknown as typeof NAMES);
    expect(second.generated).toBe(false);
    expect(second.cert).toBe(first.cert);
  });

  it('replaces a pair that no longer names a current address (a new network)', async () => {
    const first = await ensureCertificate(dir, NAMES);
    const moved = await ensureCertificate(dir, [...NAMES, '10.0.0.4']);
    expect(moved.generated).toBe(true);
    expect(moved.cert).not.toBe(first.cert);
    expect(new X509Certificate(moved.cert).subjectAltName).toContain('IP Address:10.0.0.4');
  });

  it('replaces a pair close to expiry', async () => {
    await ensureCertificate(dir, NAMES, new Date('2026-01-01T00:00:00Z'));
    expect((await ensureCertificate(dir, NAMES, new Date('2026-12-20T00:00:00Z'))).generated).toBe(true);
  });

  it('replaces a corrupt or mismatched pair rather than failing to start', async () => {
    await ensureCertificate(dir, NAMES);
    fs.writeFileSync(path.join(dir, 'cert.pem'), 'not a certificate');
    expect((await ensureCertificate(dir, NAMES)).generated).toBe(true);

    const other = fs.mkdtempSync(path.join(os.tmpdir(), 'viewer-tls-other-'));
    fs.copyFileSync((await ensureCertificate(other, NAMES)).keyfile, path.join(dir, 'key.pem'));
    fs.rmSync(other, { recursive: true });
    expect((await ensureCertificate(dir, NAMES)).generated).toBe(true);
  });

  it('lives under XDG_CACHE_HOME, where the Python viewer kept it', async () => {
    expect(cacheDir({ XDG_CACHE_HOME: '/x' })).toBe('/x/board-reader/viewer-tls');
    expect(cacheDir({})).toBe(path.join(os.homedir(), '.cache/board-reader/viewer-tls'));
  });

  it('covers localhost, loopback, the hostname and every LAN address, once each', async () => {
    expect(subjectNames(['192.168.1.7', '192.168.1.7'])).toEqual([
      'localhost',
      '127.0.0.1',
      os.hostname(),
      '192.168.1.7'
    ]);
  });
});
