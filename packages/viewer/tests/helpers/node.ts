/**
 * A stand-in for the inference node, made of Playwright route interception.
 *
 * The pages only ever talk to their own origin (`#lib/api`), so intercepting
 * `**​/api/**` and the `/ws/viewer` socket replaces the whole node — no server, no
 * model, no GPU. Every spec then states the exact reading it asserts on, which is the
 * only way an assertion about "a stale card" or "a ranked distribution" can mean
 * anything.
 */

import type { Page, WebSocketRoute } from '@playwright/test';

export const START_PLACEMENT = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';

/** `board_reader_core.protocol.CLASSES`, in its canonical order. */
export const CLASSES = [
  'empty',
  'P',
  'N',
  'B',
  'R',
  'Q',
  'K',
  'p',
  'n',
  'b',
  'r',
  'q',
  'k'
] as const;

export type ClassName = (typeof CLASSES)[number];

export interface EventOverrides {
  readonly board_id?: string;
  readonly seq?: number;
  readonly placement?: string;
  readonly status?: 'ok' | 'recalibrating' | 'needs_setup' | 'stale';
  readonly confidence?: number;
  readonly stability?: number;
  readonly square_dist?: readonly (readonly (readonly number[])[])[];
}

/**
 * One `/ws/viewer` position event, as the node spells it (DESIGN §5): every key, or
 * the page's validator (`#lib/protocol/generated`) refuses it whole.
 */
export function positionEvent(overrides: EventOverrides = {}): Record<string, unknown> {
  return {
    board_id: 'phone',
    seq: 1,
    utc: '2026-08-07T12:00:00.000Z',
    orientation: 'a1_bottom_left',
    classes: [...CLASSES],
    placement: START_PLACEMENT,
    square_dist: squareDist(['empty']),
    confidence: 0.93,
    stability: 1,
    status: 'ok',
    clock: { running: 'none' },
    ...overrides
  };
}

/**
 * One `POST /estimate` reply: the position event plus what only a one-shot read
 * carries — `upload` (only when the node retains photos) and `warnings` (only ever
 * `low_light`). Nothing else: no corners, no debug images, no rotation.
 */
export function estimateReply(
  overrides: EventOverrides & { readonly upload?: string; readonly warnings?: readonly string[] } = {}
): Record<string, unknown> {
  const { upload, warnings, ...event } = overrides;
  return {
    ...positionEvent({ board_id: 'snapshot', ...event }),
    warnings: warnings ?? [],
    ...(upload === undefined ? {} : { upload })
  };
}

/**
 * An 8×8×13 distribution in which every square carries the same ranking.
 *
 * Uniform-per-square on purpose: a spec about the inspector is asserting that the
 * ranking is rendered in order, not which square happens to hold what, so the
 * assertion should not depend on where the click landed.
 */
export function squareDist(ranking: readonly ClassName[]): number[][][] {
  const weights = [0.62, 0.2, 0.1, 0.05];
  const cell = CLASSES.map(() => 0.03 / (CLASSES.length - weights.length));
  ranking.slice(0, weights.length).forEach((cls, i) => {
    cell[CLASSES.indexOf(cls)] = weights[i] as number;
  });
  return Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => [...cell]));
}

/**
 * The live `/ws/viewer` socket, mocked end to end.
 *
 * `page.routeWebSocket` with no `connectToServer()` makes the route *be* the server, so
 * `send()` here is an event arriving at the page. Sends before the page has connected
 * are queued rather than dropped — the alternative is a race in every spec.
 */
export class FakeViewerSocket {
  #route: WebSocketRoute | null = null;
  #queued: string[] = [];

  static async install(page: Page): Promise<FakeViewerSocket> {
    const socket = new FakeViewerSocket();
    await page.routeWebSocket('**/ws/viewer', (route) => socket.#attach(route));
    return socket;
  }

  #attach(route: WebSocketRoute): void {
    this.#route = route;
    // A reload closes this socket; without this, a send() racing the reloaded
    // page's reconnect goes to the DEAD route instead of the queue and the event
    // silently never renders. Cleared only if a newer route has not taken over.
    route.onClose(() => {
      if (this.#route === route) this.#route = null;
    });
    for (const message of this.#queued) route.send(message);
    this.#queued = [];
  }

  get connected(): boolean {
    return this.#route !== null;
  }

  send(event: Record<string, unknown>): void {
    const message = JSON.stringify(event);
    if (this.#route === null) this.#queued.push(message);
    else this.#route.send(message);
  }
}

/**
 * The `/ws/ingest` socket, mocked, counting what the phone actually puts on the wire.
 *
 * The page's own "sent" counter is behind the settings sheet, and a spec about
 * forwarding should not have to open a sheet to read one number. Each forwarded
 * frame is two messages — a JSON header, then the JPEG — so counting the TEXT ones
 * counts frames. Nothing is ever sent back: an ack is the node's business, and the
 * client treats an unacked socket as a healthy one that is simply ahead.
 */
export class FakeIngestSocket {
  #frames = 0;

  static async install(page: Page): Promise<FakeIngestSocket> {
    const socket = new FakeIngestSocket();
    await page.routeWebSocket('**/ws/ingest', (route) => {
      route.onMessage((message) => {
        if (typeof message === 'string') socket.#frames++;
      });
    });
    return socket;
  }

  get frames(): number {
    return this.#frames;
  }
}

/**
 * The endpoints every page asks for on load, answered so nothing is left pending.
 *
 * `retention: false` is a real answer from a node that keeps no photos, and it is the
 * quietest one: the snapshot page's history list then has nothing to fetch thumbnails
 * for.
 */
export async function stubIdleNode(page: Page): Promise<void> {
  await page.route('**/api/uploads**', (route) =>
    route.fulfill({ json: { retention: false, uploads: [] } })
  );
}

/** Answer one node endpoint with a JSON body. */
export function stubJson(page: Page, pattern: string, body: unknown, status = 200) {
  return page.route(pattern, (route) => route.fulfill({ status, json: body }));
}
