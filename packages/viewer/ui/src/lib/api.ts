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
  cornersParam,
  parseCalibrateResponse,
  parseCornersParam,
  parseDetectResponse,
  parseEstimateResponse,
  parseLabelResponse,
  parseUploadsResponse,
  type CalibrateResponse,
  type CornerQuad,
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
  /**
   * `/calibrate`'s named cause, when it found one: the same photo passes once
   * turned this many degrees — i.e. the board is rotated in the frame. Lets the
   * page say the one actionable sentence instead of the node's full paragraph.
   */
  readonly rotatedDeg?: number;
}

export type ApiResult<T> = Result<T, ApiFailure>;

type QueryValue = string | number | boolean | undefined | null;

/**
 * `?a=1&b=2`, or `''` when nothing is set.
 *
 * Commas are left literal: they are a legal query sub-delimiter, and `?corners=`
 * is specified in the node's own `x0,y0,…` spelling (`server._parse_corners`), so
 * the string on the wire reads as the operator would type it.
 */
function query(params: Readonly<Record<string, QueryValue>>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    parts.push(`${key}=${encodeURIComponent(String(value)).replace(/%2C/g, ',')}`);
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
        const deg = (parsed as { rotated_deg?: unknown }).rotated_deg;
        return typeof deg === 'number' && Number.isFinite(deg)
          ? { status: response.status, message: reason, rotatedDeg: deg }
          : { status: response.status, message: reason };
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

export interface EstimateOptions {
  /** a8,h8,h1,a1 in the POSTED image's pixels. Omitted, the body is a rectified crop. */
  readonly corners?: CornerQuad;
  readonly boardId?: string;
  /** `?debug=1` — the boxes/mosaic pair. The snapshot page asks unconditionally. */
  readonly debug?: boolean;
  /**
   * Let the node turn the reading to the orientation its own reading implies.
   * Off once the operator has asserted a labelling by hand: theirs wins.
   */
  readonly autorotate?: boolean;
}

export function estimate(photo: Blob, options: EstimateOptions = {}): Promise<ApiResult<EstimateResponse>> {
  const url = apiPath('/estimate', {
    board_id: options.boardId,
    corners: options.corners ? cornersParam(options.corners) : undefined,
    debug: options.debug ? 1 : undefined,
    autorotate: options.autorotate ? 1 : undefined
  });
  return requestJson(url, { method: 'POST', body: photo, headers: JPEG_HEADERS }, parseEstimateResponse);
}

/**
 * Few-shot per-set calibration from one labelled photo (DESIGN §4.6).
 *
 * The body is the ORIGINAL photo, full resolution — never a downscaled stream
 * frame: calibration reads one exemplar per square and every pixel of a piece is
 * evidence. `placement` defaults to the start position on the node, which is the
 * one position containing all 13 classes.
 */
export function calibrate(
  photo: Blob,
  options: { readonly corners: CornerQuad; readonly placement?: string }
): Promise<ApiResult<CalibrateResponse>> {
  const url = apiPath('/calibrate', {
    corners: cornersParam(options.corners),
    placement: options.placement
  });
  return requestJson(url, { method: 'POST', body: photo, headers: JPEG_HEADERS }, parseCalibrateResponse);
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
    { method: 'POST', body: photo, headers: JPEG_HEADERS },
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
 * nothing reads a label back into the calibration bank.
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

export interface BoardFrame {
  readonly jpeg: Blob;
  /**
   * The playing surface within THIS image, when the node stated one on `X-Corners`
   * — see `cornersOf` for why that matters.
   */
  readonly corners: CornerQuad | null;
  readonly seq: number | null;
  readonly captureTs: number | null;
}

/**
 * Where the board sits inside a retained frame.
 *
 * A rectified edge stream retains the board itself, so the image's own four
 * corners are the board's; a phone stream retains an UN-rectified crop and the
 * node states the quad within it on `X-Corners` (which `proxy.py` forwards for
 * exactly this reason). Preferring the header keeps ONE calibrate path for both
 * kinds of stream, and a node that does not send it behaves as before.
 */
export function cornersOf(
  stated: string | null,
  image: { readonly width: number; readonly height: number }
): CornerQuad {
  return (
    parseCornersParam(stated) ?? [
      [0, 0],
      [image.width, 0],
      [image.width, image.height],
      [0, image.height]
    ]
  );
}

/** An absent header is absent, not zero — `Number(null)` is 0 and would read as seq 0. */
function numericHeader(response: Response, name: string): number | null {
  const raw = response.headers.get(name);
  if (raw === null) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export async function fetchBoardFrame(
  boardId: string,
  bust: string | number
): Promise<ApiResult<BoardFrame>> {
  let response: Response;
  try {
    response = await fetch(boardFrameUrl(boardId, bust), { cache: 'no-store' });
  } catch (cause) {
    return err({ status: 0, message: `${String(cause)} (is the viewer still reachable?)` });
  }
  // 404 until the node has ingested a frame, and an older node may not serve this
  // at all — either way the caller keeps working without a preview.
  if (!response.ok) return err(await failureOf(response));
  return ok({
    jpeg: await response.blob(),
    corners: parseCornersParam(response.headers.get('X-Corners')),
    seq: numericHeader(response, 'X-Seq'),
    captureTs: numericHeader(response, 'X-Capture-Ts')
  });
}
