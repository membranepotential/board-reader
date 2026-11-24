/**
 * The inference node's API, as this page may address it: **same-origin, always**.
 *
 * The viewer serves this UI and relays the node's whole HTTP API under `/api/…`
 * plus `/ws/viewer` and `/ws/ingest` (`board_reader_viewer.proxy`). That single
 * origin is load-bearing, not a tidiness preference:
 *
 *  - `getUserMedia` needs a secure context, and a page served over https cannot
 *    open an insecure `ws://` socket (mixed content) nor accept a self-signed
 *    certificate for a *second* origin's `wss://` — there is no interstitial for
 *    WebSockets. One origin means the phone accepts exactly one certificate once.
 *  - the live page and `/snapshot` used to be handed an absolute
 *    `http://<lan-ip>:8000`, which made them LAN-only: blocked mixed content over
 *    TLS, and a host no browser behind a tunnel could reach.
 *
 * So every URL this module produces is **relative**. Nothing here constructs an
 * origin, reads one from configuration, or accepts one as an argument — that is
 * the invariant, and it is why the pages work unchanged on a LAN, over https and
 * behind a tunnel.
 */

import {
  parseDetectResponse,
  parseEstimateResponse,
  parseLabelResponse,
  parseUploadsResponse,
  type DetectResponse,
  type EstimateResponse,
  type LabelResponse,
  type ParseResult,
  type UploadsResponse
} from './protocol';
import { err, ok, type Result } from './result';

/** The viewer's catch-all relay prefix; `/x` on the node is `/api/x` here. */
export const API_PREFIX = '/api';
export const VIEWER_SOCKET = '/ws/viewer';
export const INGEST_SOCKET = '/ws/ingest';

export interface ApiFailure {
  /** HTTP status, or 0 when the request never got an answer at all. */
  readonly status: number;
  /** The node's own wording where it gave one — operator-facing, shown verbatim. */
  readonly message: string;
}

export type ApiResult<T> = Result<T, ApiFailure>;

type QueryValue = string | number | boolean | undefined | null;

/** `?a=1&b=2`, or `''` when nothing is set. */
function query(params: Readonly<Record<string, QueryValue>>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    parts.push(`${key}=${encodeURIComponent(String(value))}`);
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

/**
 * `path` is the node's own path (`/estimate`, `/boards/x/frame`) — the relay
 * prefix is added here so no caller has to remember it, and no caller can spell
 * an absolute one.
 */
export function apiPath(path: string, params: Readonly<Record<string, QueryValue>> = {}): string {
  if (!path.startsWith('/')) throw new TypeError(`api path must start with "/": ${path}`);
  return `${API_PREFIX}${path}${query(params)}`;
}

/**
 * Relative WS URL for one of the viewer's sockets — `wss:` exactly when the page
 * itself is on `https:`, derived from the page rather than configured.
 *
 * `pageHref` exists so this is testable (and so a prerender never touches
 * `location`); in the browser it defaults to the page's own address.
 */
export function wsUrl(path: string, pageHref?: string): string {
  const href = pageHref ?? (typeof location === 'undefined' ? undefined : location.href);
  if (href === undefined) {
    throw new TypeError('wsUrl needs a page address: there is no document here');
  }
  const url = new URL(path, href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
}

// ---- requests ----------------------------------------------------------------

const JPEG_HEADERS = { 'Content-Type': 'image/jpeg' } as const;

/**
 * How long `/detect` may take before the page stops waiting for it.
 *
 * A request that never answers is worse than a refusal: the camera page hands the
 * camera and the quad to exactly one activity at a time, so a `fetch` that hangs
 * for ever leaves every idle-gated control — drag, detect, the automatic framing
 * and the drift check — disabled with nothing on screen to say why. Detection
 * costs ~0.1 s on a hit and 2-4 s on a miss (worst recorded 1.7 s), so 15 s is
 * far outside the working distribution and only ever fires on something
 * genuinely stuck.
 */
export const DETECT_TIMEOUT_MS = 15_000;

/** A `fetch` that gave up on its own clock, rather than one that failed. */
function timedOut(cause: unknown): boolean {
  return (
    typeof cause === 'object' &&
    cause !== null &&
    'name' in cause &&
    (cause as { readonly name?: unknown }).name === 'TimeoutError'
  );
}

/** The node states its refusals as `{"error": "..."}`; anything else, keep the body. */
async function failureOf(response: Response): Promise<ApiFailure> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  let body = '';
  try {
    body = await response.text();
  } catch {
    return { status: response.status, message: fallback };
  }
  try {
    const parsed: unknown = JSON.parse(body);
    if (typeof parsed === 'object' && parsed !== null && 'error' in parsed) {
      const reason = (parsed as { error: unknown }).error;
      if (typeof reason === 'string' && reason !== '') {
        return { status: response.status, message: reason };
      }
    }
  } catch {
    /* not JSON — the raw body is the most informative thing left */
  }
  return { status: response.status, message: body.trim() || fallback };
}

async function requestJson<T>(
  url: string,
  init: RequestInit,
  parse: (raw: unknown) => ParseResult<T>
): Promise<ApiResult<T>> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (cause) {
    // A timeout is not an unreachable viewer — it is an answer that never came,
    // and saying so is the difference between "check the network" and "the node
    // is wedged".
    if (timedOut(cause)) return err({ status: 0, message: 'the node did not answer in time' });
    // The relay answers 502 with the node's address when the NODE is down, so a
    // thrown fetch means this page could not reach its own viewer.
    return err({ status: 0, message: `${String(cause)} (is the viewer still reachable?)` });
  }
  if (!response.ok) return err(await failureOf(response));
  let body: unknown;
  try {
    body = await response.json();
  } catch (cause) {
    return err({ status: response.status, message: `unreadable reply: ${String(cause)}` });
  }
  const parsed = parse(body);
  return parsed.ok ? ok(parsed.value) : err({ status: response.status, message: parsed.error });
}

// ---- endpoints ---------------------------------------------------------------

/**
 * Read one photo. The body is the whole photo: the recognizer finds the board and
 * its orientation itself, and answers in the chess frame (rank 8 on top).
 */
export function estimate(photo: Blob, boardId: string): Promise<ApiResult<EstimateResponse>> {
  const url = apiPath('/estimate', { board_id: boardId });
  return requestJson(url, { method: 'POST', body: photo, headers: JPEG_HEADERS }, parseEstimateResponse);
}

/**
 * Locate the board → a proposed quad in GEOMETRIC TL,TR,BR,BL order.
 *
 * Which physical corner is a8 is not knowable from geometry (DESIGN decision #15),
 * so a detection replaces the quad and never the labelling.
 */
export function detect(photo: Blob): Promise<ApiResult<DetectResponse>> {
  return requestJson(
    apiPath('/detect'),
    {
      method: 'POST',
      body: photo,
      headers: JPEG_HEADERS,
      // Bounded here rather than at each caller: all three of them (look again,
      // the automatic first framing, the periodic drift check) claim the page's
      // single activity slot for the duration, so all three need the guarantee.
      signal: AbortSignal.timeout(DETECT_TIMEOUT_MS)
    },
    parseDetectResponse
  );
}

export function listUploads(limit = 100): Promise<ApiResult<UploadsResponse>> {
  return requestJson(apiPath('/uploads', { limit }), { method: 'GET' }, parseUploadsResponse);
}

/** `?max=<px>` downscales the long side for a thumbnail; omitted, the stored bytes. */
export function uploadPhotoUrl(uploadId: string, maxSide?: number): string {
  return apiPath(`/uploads/${encodeURIComponent(uploadId)}/photo`, { max: maxSide });
}

/**
 * Record the operator's confirmed position for a retained photo — eval data only;
 * nothing reads a label back into the recognizer.
 */
export function saveLabel(
  uploadId: string,
  label: { readonly placement: string; readonly corrections: number }
): Promise<ApiResult<LabelResponse>> {
  return requestJson(
    apiPath(`/labels/${encodeURIComponent(uploadId)}`),
    {
      method: 'POST',
      body: JSON.stringify(label),
      headers: { 'Content-Type': 'application/json' }
    },
    parseLabelResponse
  );
}

/**
 * The most recent JPEG ingested for a board, as received.
 *
 * `bust` only defeats the cache — pass the last seen `seq` so a new frame is a new
 * URL and a repeated one is not refetched.
 */
export function boardFrameUrl(boardId: string, bust: string | number): string {
  return apiPath(`/boards/${encodeURIComponent(boardId)}/frame`, { seq: bust });
}
