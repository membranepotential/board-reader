/** The startup banner: where to point the phone and the laptop, and why one may refuse. */

/** The TLS port as the banner describes it. */
export interface TlsBind {
  readonly port: number;
  readonly certfile: string;
  readonly names: readonly string[];
  readonly generated: boolean;
}

export interface BannerInput {
  readonly host: string;
  readonly port: number;
  readonly tls: TlsBind | undefined;
  /** This machine's LAN IPv4 addresses, routed one first. */
  readonly lan: readonly string[];
  /** The node's HTTP address, as configured. */
  readonly node: string;
}

const WILDCARD = new Set(['0.0.0.0', '::', '']);

/**
 * The hosts a browser can reach the viewer at. A wildcard bind answers on every local
 * address, so all are offered; a specific bind answers on exactly that one, and
 * suggesting another would be a lie.
 */
function reachable(host: string, lan: readonly string[]): string[] {
  return WILDCARD.has(host) ? [...lan, 'localhost'] : [host];
}

export function banner({ host, port, tls, lan, node }: BannerInput): string[] {
  const hosts = reachable(host, lan);
  const remote = hosts[0] as string; // the address a phone uses
  const local = WILDCARD.has(host) ? 'localhost' : host; // this machine's own browser
  const lines = [`board-reader viewer — the node at ${node} is relayed under /api and /ws`];

  if (tls !== undefined) {
    const [first, ...rest] = hosts.map((h) => `https://${h}:${tls.port}/camera`);
    lines.push(`  camera    ${first}   (self-signed: accept the warning once)`);
    lines.push(...rest.map((url) => `            ${url}`));
  } else {
    lines.push(
      '  camera    blocked over plain http (getUserMedia needs a secure context):',
      `            pass --tls-port, or forward the port: adb reverse tcp:${port} tcp:${port}`
    );
  }
  lines.push(
    `  control   http://${local}:${port}/control`,
    `  snapshot  http://${remote}:${port}/snapshot`,
    `  live      http://${remote}:${port}/`
  );
  if (tls !== undefined) {
    lines.push(
      `  tls       ${tls.generated ? 'generated' : 'reusing'} ${tls.certfile}`,
      `            valid for ${tls.names.join(', ')}`
    );
  }
  if (host.startsWith('127.') || host === 'localhost') {
    lines.push(`  (bound to ${host} — pass --host 0.0.0.0 to reach it from a phone)`);
  }
  return lines;
}
