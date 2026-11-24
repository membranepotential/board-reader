/**
 * The control page's client for the camera control channel and the node's health.
 *
 * Routes (inference node, relayed same-origin under `/api` by the viewer):
 *
 *  - `GET/PUT /boards/{id}/camera` — the control view: config, the phone's last
 *    report and when it came, preview and streamed-frame metadata, and the node's
 *    own clock (`board_reader_inference.camera_control.CameraRegistry.view`).
 *  - `GET /boards/{id}/camera/preview` — the phone's latest full-frame JPEG.
 *  - `GET /boards/{id}/frame` — the latest frame `/ingest` received (`boardFrameUrl`).
 *  - `GET /healthz` — the recognizer and the ingest counters.
 *
 * Everything is parsed here once, so the page only ever holds well-formed values.
 */

import { CONTROL_TIMEOUT_MS, apiPath, type ApiFailure, type ApiResult } from '$lib/api';
import {
  isRecord,
  parseCameraConfig,
  type CameraConfig,
  type CameraDevice,
  type CameraReport,
  type FrameSize,
  type IngestState,
  type ParseResult
} from '$lib/protocol';
import { err, ok } from '$lib/result';

// ---- shapes ------------------------------------------------------------------

/** The phone's latest full-frame preview, as the node describes it. */
export interface PreviewMeta {
  readonly seq: number;
  /** Node clock, seconds. */
  readonly at: number;
  readonly width: number;
  readonly height: number;
}

/** The last frame `/ingest` received for the board. */
export interface StreamedFrameMeta {
  readonly seq: number;
  /** Node clock, seconds. */
  readonly received_at: number;
  readonly width: number;
  readonly height: number;
}

/** `GET /boards/{id}/camera`. Ages are measured against `now`, the node's clock. */
export interface CameraView {
  readonly board_id: string;
  readonly config: CameraConfig;
  readonly report: CameraReport | null;
  readonly seen_at: number | null;
  readonly preview: PreviewMeta | null;
  readonly frame: StreamedFrameMeta | null;
  readonly now: number;
}

/** The slice of `/healthz` the control page shows. */
export interface NodeHealth {
  readonly recognizer: string;
  /** Why the stub serves instead of the model, when it does. */
  readonly fallback: string | null;
  readonly received: number;
  readonly refused: number;
}

// ---- parsing -----------------------------------------------------------------

const INGEST_STATES = ['idle', 'connecting', 'streaming', 'reconnecting'] as const satisfies
  readonly IngestState[];

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function nullableNum(value: unknown): number | null | undefined {
  return value === null ? null : (num(value) ?? undefined);
}

function frameSize(value: unknown): FrameSize | null {
  if (!isRecord(value)) return null;
  const width = num(value['width']);
  const height = num(value['height']);
  return width !== null && height !== null && width > 0 && height > 0 ? { width, height } : null;
}

export function parseCameraReport(raw: unknown): ParseResult<CameraReport> {
  if (!isRecord(raw)) return err('the phone report is not an object');
  const devicesRaw = raw['devices'];
  if (!Array.isArray(devicesRaw)) return err('the phone report has no device list');
  const devices: CameraDevice[] = [];
  for (const entry of devicesRaw) {
    if (!isRecord(entry) || typeof entry['id'] !== 'string' || typeof entry['label'] !== 'string') {
      return err('the phone report has a malformed device');
    }
    devices.push({ id: entry['id'], label: entry['label'] });
  }
  const device = raw['device'];
  if (device !== null && typeof device !== 'string') return err('the phone report has no usable lens');
  const frame = raw['frame'] === null ? null : frameSize(raw['frame']);
  if (raw['frame'] !== null && frame === null) return err('the phone report has no usable frame size');
  const ingest = INGEST_STATES.find((state) => state === raw['ingest']);
  if (ingest === undefined) return err('the phone report has no known stream state');
  return ok({ devices, device, frame, ingest });
}

function parsePreview(value: unknown): PreviewMeta | null | undefined {
  if (value === null) return null;
  if (!isRecord(value)) return undefined;
  const seq = num(value['seq']);
  const at = num(value['at']);
  const size = frameSize(value);
  return seq === null || at === null || size === null ? undefined : { seq, at, ...size };
}

function parseStreamed(value: unknown): StreamedFrameMeta | null | undefined {
  if (value === null) return null;
  if (!isRecord(value)) return undefined;
  const seq = num(value['seq']);
  const received = num(value['received_at']);
  const size = frameSize(value);
  return seq === null || received === null || size === null
    ? undefined
    : { seq, received_at: received, ...size };
}

export function parseCameraView(raw: unknown): ParseResult<CameraView> {
  if (!isRecord(raw)) return err('the camera view is not an object');
  const boardId = raw['board_id'];
  if (typeof boardId !== 'string') return err('the camera view names no board');
  const config = parseCameraConfig(raw['config']);
  if (!config.ok) return config;
  let report: CameraReport | null = null;
  if (raw['report'] !== null) {
    const parsed = parseCameraReport(raw['report']);
    if (!parsed.ok) return parsed;
    report = parsed.value;
  }
  const seenAt = nullableNum(raw['seen_at']);
  const preview = parsePreview(raw['preview']);
  const frame = parseStreamed(raw['frame']);
  const now = num(raw['now']);
  if (seenAt === undefined || preview === undefined || frame === undefined || now === null) {
    return err('the camera view is malformed');
  }
  return ok({ board_id: boardId, config: config.value, report, seen_at: seenAt, preview, frame, now });
}

export function parseHealth(raw: unknown): ParseResult<NodeHealth> {
  if (!isRecord(raw)) return err('the health reply is not an object');
  const recognizer = raw['recognizer'];
  const ingest = raw['ingest'];
  if (!isRecord(recognizer) || typeof recognizer['name'] !== 'string' || !isRecord(ingest)) {
    return err('the health reply names no recognizer');
  }
  const fallback = recognizer['fallback'];
  return ok({
    recognizer: recognizer['name'],
    fallback: typeof fallback === 'string' ? fallback : null,
    received: num(ingest['received']) ?? 0,
    refused: num(ingest['refused']) ?? 0
  });
}

// ---- requests ----------------------------------------------------------------

/** The node's `{"error": "..."}` wording when it gave one, else the status line. */
async function failureOf(response: Response): Promise<ApiFailure> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  try {
    const body: unknown = await response.json();
    if (isRecord(body) && typeof body['error'] === 'string' && body['error'] !== '') {
      return { status: response.status, message: body['error'] };
    }
  } catch {
    /* not JSON: the status line is all there is */
  }
  return { status: response.status, message: fallback };
}

async function requestJson<T>(
  url: string,
  init: RequestInit,
  parse: (raw: unknown) => ParseResult<T>
): Promise<ApiResult<T>> {
  let response: Response;
  try {
    response = await fetch(url, { ...init, signal: AbortSignal.timeout(CONTROL_TIMEOUT_MS) });
  } catch (cause) {
    const timedOut = cause instanceof DOMException && cause.name === 'TimeoutError';
    return err({
      status: 0,
      message: timedOut ? 'the node did not answer in time' : `the viewer is unreachable: ${String(cause)}`
    });
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

function cameraPath(boardId: string, rest = ''): string {
  return `/boards/${encodeURIComponent(boardId)}/camera${rest}`;
}

export function getCamera(boardId: string): Promise<ApiResult<CameraView>> {
  return requestJson(apiPath(cameraPath(boardId)), { method: 'GET' }, parseCameraView);
}

/** Replace the whole config; the reply is the view with it applied. */
export function putCamera(boardId: string, config: CameraConfig): Promise<ApiResult<CameraView>> {
  return requestJson(
    apiPath(cameraPath(boardId)),
    { method: 'PUT', body: JSON.stringify(config), headers: { 'Content-Type': 'application/json' } },
    parseCameraView
  );
}

export function getHealth(): Promise<ApiResult<NodeHealth>> {
  return requestJson(apiPath('/healthz'), { method: 'GET' }, parseHealth);
}

/** The phone's full-frame preview; `seq` makes each new preview a new URL. */
export function previewUrl(boardId: string, seq: number): string {
  return apiPath(cameraPath(boardId, '/preview'), { seq });
}
