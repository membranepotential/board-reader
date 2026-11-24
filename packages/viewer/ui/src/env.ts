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
  }
});
