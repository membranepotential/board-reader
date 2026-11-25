import { defineEnvVars } from '@sveltejs/kit/env';

/** The viewer's own configuration, read at runtime (`$app/env/private`). */
export const variables = defineEnvVars({
  BR_INFERENCE_HTTP: {
    description: "The inference node's HTTP address; `/api/<path>` relays to its `/<path>`.",
    // Only the origin: a path on the configured URL would otherwise prefix every call.
    schema: (value) => new URL(value || 'http://localhost:8000').origin
  },
  BR_CLIENT_LOG_DIR: {
    description:
      "Where the phone's trial notes (`POST /api/client-log`) are appended; unset or empty drops them. " +
      'server/main.ts defaults it to `client-logs` for a real server.',
    schema: (value) => value || undefined
  },
  BR_PUBLIC_URL: {
    description:
      'An origin that serves this viewer over TLS from outside the LAN (a Cloudflare quick tunnel); ' +
      "/control offers its `/camera` first in the phone's QR code. Unset: only this machine's own addresses.",
    // Only the origin: the pages use root-relative URLs, so a path prefix could not work.
    schema: (value) => (value ? new URL(value).origin : undefined)
  },
  BR_VIEWER_TLS_PORT: {
    description: 'Set by server/main.ts: the https port it serves, unset without `--tls-port`.',
    schema: (value) => (value ? Number(value) : undefined)
  },
  BR_VIEWER_TLS_NAMES: {
    description: "Set by server/main.ts: the names its TLS certificate covers, comma-separated.",
    schema: (value) => (value ? value.split(',') : [])
  }
});
