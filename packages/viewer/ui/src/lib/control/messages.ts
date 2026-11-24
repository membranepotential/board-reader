/**
 * The camera control channel's messages (`board_reader_core.control`), parsed once.
 *
 * Two sockets per board, relayed same-origin by the viewer under `/ws`:
 *
 *  - the phone's, `cameraSocket(id)`: it sends `report` (text) and, only while the
 *    node asks, preview JPEGs (binary); it hears `config`, `previews` and `refused`.
 *  - a control page's, `controlSocket(id)`: it sends `configure`; it hears `state`,
 *    `preview`, `frame`, `health` and `refused`.
 *
 * Every text message is a JSON object whose `type` names it. Anything that does not
 * parse is answered with a reason, never half-applied.
 */

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

/** The board's camera as the node holds it. */
export interface CameraState {
  readonly config: CameraConfig;
  /** The phone's last report; kept after it disconnects, so its lens list survives. */
  readonly report: CameraReport | null;
  readonly phone_connected: boolean;
}

/** A JPEG the node pushed, ready for an `<img src>`. */
export interface Picture extends FrameSize {
  readonly src: string;
}

/** The latest frame `/ingest` received for the board. */
export interface ReceivedFrame extends Picture {
  readonly seq: number;
}

/** Which edge of the image White's back rank lies along. */
export type WhiteSide = 'bottom' | 'left' | 'top' | 'right';

/**
 * Which way the board lies in the camera's image, as the recognizer reads it.
 *
 * `degrees` is the direction of e1->e8: 0 White at the bottom, 90 White on the left,
 * 180 White at the top, -90 White on the right. `camera_moved` says the angle drifted
 * well away from where it settled after the last crop, lens or resolution change.
 */
export interface BoardDirection {
  readonly degrees: number;
  readonly white_side: WhiteSide;
  readonly camera_moved: boolean;
}

/** The node's own facts, about once a second; ages and rates on the node's clock. */
export interface NodeHealth {
  readonly recognizer: string;
  /** Why the stub serves instead of the model, when it does. */
  readonly fallback: string | null;
  readonly received: number;
  readonly refused: number;
  /** Seconds since the board's last frame arrived; `null` before the first. */
  readonly frame_age_s: number | null;
  /** Frames per second since the previous tick; `null` when not measurable. */
  readonly fps: number | null;
  /** `null` before the first read that has one. */
  readonly direction: BoardDirection | null;
}

/** What the node sends the phone. */
export type PhoneDown =
  | { readonly type: 'config'; readonly config: CameraConfig }
  | { readonly type: 'previews'; readonly on: boolean }
  | { readonly type: 'refused'; readonly reason: string };

/** What the node sends a control page. */
export type ControlDown =
  | { readonly type: 'state'; readonly state: CameraState }
  | { readonly type: 'preview'; readonly picture: Picture }
  | { readonly type: 'frame'; readonly frame: ReceivedFrame }
  | { readonly type: 'health'; readonly health: NodeHealth }
  | { readonly type: 'refused'; readonly reason: string };

// ---- parsing -----------------------------------------------------------------

const INGEST_STATES = ['idle', 'connecting', 'streaming', 'reconnecting'] as const satisfies
  readonly IngestState[];
const WHITE_SIDES = ['bottom', 'left', 'top', 'right'] as const satisfies readonly WhiteSide[];

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

function parseJson(text: string): Record<string, unknown> | null {
  try {
    const raw: unknown = JSON.parse(text);
    return isRecord(raw) ? raw : null;
  } catch {
    return null;
  }
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

function parsePicture(raw: Record<string, unknown>): Picture | null {
  const size = frameSize(raw);
  const jpeg = raw['jpeg'];
  if (size === null || typeof jpeg !== 'string' || jpeg === '') return null;
  return { ...size, src: `data:image/jpeg;base64,${jpeg}` };
}

function parseState(raw: Record<string, unknown>): ParseResult<CameraState> {
  const config = parseCameraConfig(raw['config']);
  if (!config.ok) return config;
  let report: CameraReport | null = null;
  if (raw['report'] !== null) {
    const parsed = parseCameraReport(raw['report']);
    if (!parsed.ok) return parsed;
    report = parsed.value;
  }
  const phone = raw['phone_connected'];
  if (typeof phone !== 'boolean') return err('the camera state does not say whether the phone is connected');
  return ok({ config: config.value, report, phone_connected: phone });
}

/** A direction, `null` for none, `undefined` for one that does not parse. */
function parseDirection(raw: unknown): BoardDirection | null | undefined {
  if (raw === null) return null;
  if (!isRecord(raw)) return undefined;
  const degrees = num(raw['degrees']);
  const side = WHITE_SIDES.find((s) => s === raw['white_side']);
  const moved = raw['camera_moved'];
  if (degrees === null || side === undefined || typeof moved !== 'boolean') return undefined;
  return { degrees, white_side: side, camera_moved: moved };
}

function parseHealth(raw: Record<string, unknown>): ParseResult<NodeHealth> {
  const recognizer = raw['recognizer'];
  const fallback = raw['fallback'];
  const received = num(raw['received']);
  const refused = num(raw['refused']);
  const age = nullableNum(raw['frame_age_s']);
  const fps = nullableNum(raw['fps']);
  const direction = parseDirection(raw['direction']);
  if (
    typeof recognizer !== 'string' ||
    (fallback !== null && typeof fallback !== 'string') ||
    received === null ||
    refused === null ||
    age === undefined ||
    fps === undefined ||
    direction === undefined
  ) {
    return err('the health message is malformed');
  }
  return ok({ recognizer, fallback, received, refused, frame_age_s: age, fps, direction });
}

function parseRefused(raw: Record<string, unknown>): ParseResult<{ type: 'refused'; reason: string }> {
  const reason = raw['reason'];
  return typeof reason === 'string' ? ok({ type: 'refused', reason }) : err('a refusal with no reason');
}

export function parsePhoneDown(text: string): ParseResult<PhoneDown> {
  const raw = parseJson(text);
  if (raw === null) return err('not a JSON object');
  switch (raw['type']) {
    case 'config': {
      const config = parseCameraConfig(raw['config']);
      return config.ok ? ok({ type: 'config', config: config.value }) : config;
    }
    case 'previews':
      return typeof raw['on'] === 'boolean'
        ? ok({ type: 'previews', on: raw['on'] })
        : err('a previews message that says neither on nor off');
    case 'refused':
      return parseRefused(raw);
    default:
      return err(`not a message for the phone: ${String(raw['type'])}`);
  }
}

export function parseControlDown(text: string): ParseResult<ControlDown> {
  const raw = parseJson(text);
  if (raw === null) return err('not a JSON object');
  switch (raw['type']) {
    case 'state': {
      const state = parseState(raw);
      return state.ok ? ok({ type: 'state', state: state.value }) : state;
    }
    case 'preview': {
      const picture = parsePicture(raw);
      return picture === null ? err('a preview with no picture') : ok({ type: 'preview', picture });
    }
    case 'frame': {
      const picture = parsePicture(raw);
      const seq = num(raw['seq']);
      return picture === null || seq === null
        ? err('a frame with no picture')
        : ok({ type: 'frame', frame: { ...picture, seq } });
    }
    case 'health': {
      const health = parseHealth(raw);
      return health.ok ? ok({ type: 'health', health: health.value }) : health;
    }
    case 'refused':
      return parseRefused(raw);
    default:
      return err(`not a message for a control page: ${String(raw['type'])}`);
  }
}

// ---- what goes up --------------------------------------------------------------

export function reportMessage(report: CameraReport): string {
  return JSON.stringify({ type: 'report', report });
}

export function configureMessage(config: CameraConfig): string {
  return JSON.stringify({ type: 'configure', config });
}
