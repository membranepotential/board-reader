/**
 * The self-signed certificate for the viewer's TLS port, so a phone can use its camera.
 *
 * `getUserMedia` needs a secure context: over plain http a phone on the LAN is not even
 * offered the camera. There is no public hostname here, so the way in is a certificate
 * the operator accepts once. It covers every address the machine answers on and is
 * cached, so that acceptance is a one-time cost per machine.
 *
 * The cached pair is reused until it is nearly expired or no longer names the
 * machine's current addresses, because moving between networks changes the LAN IP, and
 * a phone rejects a certificate that does not name the address typed before it ever
 * offers the choice to trust it.
 */

import { createPrivateKey, X509Certificate } from 'node:crypto';
import dgram from 'node:dgram';
import { lookup } from 'node:dns/promises';
import fs from 'node:fs';
import { isIPv4 } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { generate } from 'selfsigned';

const VALID_DAYS = 365;
// Renew this long before expiry, so a long-running box never wakes up to a
// certificate the phone refuses.
const RENEW_BEFORE_DAYS = 15;
const DAY_MS = 86_400_000;
const SERVER_AUTH = '1.3.6.1.5.5.7.3.1';

/** A cert/key pair on disk, what it covers, and whether this start minted it. */
export interface Certificate {
  readonly certfile: string;
  readonly keyfile: string;
  readonly cert: string;
  readonly key: string;
  readonly names: readonly string[];
  readonly generated: boolean;
}

/** Where the pair lives: `$XDG_CACHE_HOME/board-reader/viewer-tls`. */
export function cacheDir(env: NodeJS.ProcessEnv = process.env): string {
  const base = env['XDG_CACHE_HOME'] || path.join(os.homedir(), '.cache');
  return path.join(base, 'board-reader', 'viewer-tls');
}

/** The address the kernel routes out of, found without sending anything. */
function routedAddress(): Promise<string | undefined> {
  return new Promise((resolve) => {
    const probe = dgram.createSocket('udp4');
    const done = (address: string | undefined): void => {
      probe.close();
      resolve(address);
    };
    probe.once('error', () => done(undefined)); // no route out: loopback-only box
    // A UDP connect only sets the socket's peer; no packet leaves.
    probe.connect(80, '1.1.1.1', () => done(probe.address().address));
  });
}

/**
 * Every non-loopback IPv4 address this host answers on, routed address first.
 *
 * Two sources, because neither is complete: the routed address is the LAN address
 * that matters, and the hostname's addresses catch a second interface.
 */
export async function localIPv4Addresses(): Promise<string[]> {
  const routed = await routedAddress();
  const named = await lookup(os.hostname(), { all: true, family: 4 }).catch(() => []);
  const found = [routed, ...named.map((entry) => entry.address)];
  return [...new Set(found.filter((a): a is string => a !== undefined && !a.startsWith('127.')))];
}

/** The names the certificate must cover, in the order a banner offers them. */
export function subjectNames(lan: readonly string[]): [string, ...string[]] {
  return [...new Set(['localhost', '127.0.0.1', os.hostname(), ...lan].filter(Boolean))] as [
    string,
    ...string[]
  ];
}

/** The SAN entries of a parsed certificate (`DNS:x, IP Address:y` → `[x, y]`). */
function sanEntries(cert: X509Certificate): Set<string> {
  return new Set(
    (cert.subjectAltName ?? '').split(', ').map((entry) => entry.slice(entry.indexOf(':') + 1))
  );
}

/** Whether a cached pair may be served for `names` at `now`. */
function reusable(cert: string, key: string, names: readonly string[], now: Date): boolean {
  let parsed: X509Certificate;
  try {
    parsed = new X509Certificate(cert);
    if (!parsed.checkPrivateKey(createPrivateKey(key))) return false;
  } catch {
    return false; // corrupt or half-written: mint a new one
  }
  const fresh = parsed.validToDate.getTime() - now.getTime() > RENEW_BEFORE_DAYS * DAY_MS;
  // `keyUsage` is Node's name for the extended key usages. A pair without serverAuth
  // is retired: iOS rejects it outright.
  const serverAuth = parsed.keyUsage?.includes(SERVER_AUTH) ?? false;
  const present = sanEntries(parsed);
  return fresh && serverAuth && names.every((name) => present.has(name));
}

function readIfPresent(file: string): string | undefined {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch {
    return undefined;
  }
}

/** Write the private key so it is never readable by anyone else, not even briefly. */
function writePrivate(file: string, content: string): void {
  const fd = fs.openSync(file, 'w', 0o600);
  try {
    fs.fchmodSync(fd, 0o600); // the file may predate us with a looser mode
    fs.writeSync(fd, content);
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * A usable self-signed pair for `names`, minted only when the cache cannot serve.
 *
 * Reuse requires both files present and matching, comfortably short of expiry, the
 * serverAuth usage, and SANs covering every name.
 */
export async function ensureCertificate(
  directory: string,
  names: readonly [string, ...string[]],
  now: Date = new Date()
): Promise<Certificate> {
  const certfile = path.join(directory, 'cert.pem');
  const keyfile = path.join(directory, 'key.pem');
  const cachedCert = readIfPresent(certfile);
  const cachedKey = readIfPresent(keyfile);
  if (cachedCert !== undefined && cachedKey !== undefined && reusable(cachedCert, cachedKey, names, now)) {
    return { certfile, keyfile, cert: cachedCert, key: cachedKey, names, generated: false };
  }

  const { cert, private: key } = await generate(
    [
      { name: 'commonName', value: names[0] },
      { name: 'organizationName', value: 'board-reader' }
    ],
    {
      // P-256: supported by anything with a camera API, and generated in microseconds.
      keyType: 'ec',
      curve: 'P-256',
      algorithm: 'sha256',
      // Backdated an hour: a phone whose clock runs slightly behind must not meet a
      // certificate that is not valid yet.
      notBeforeDate: new Date(now.getTime() - 3_600_000),
      notAfterDate: new Date(now.getTime() + VALID_DAYS * DAY_MS),
      extensions: [
        { name: 'basicConstraints', cA: false, critical: true },
        { name: 'keyUsage', digitalSignature: true, keyEncipherment: true, critical: true },
        // iOS rejects a TLS server certificate without serverAuth, with no override.
        { name: 'extKeyUsage', serverAuth: true },
        {
          name: 'subjectAltName',
          altNames: names.map((name) => (isIPv4(name) ? { type: 7, ip: name } : { type: 2, value: name }))
        }
      ]
    }
  );
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(certfile, cert);
  writePrivate(keyfile, key);
  return { certfile, keyfile, cert, key, names, generated: true };
}
