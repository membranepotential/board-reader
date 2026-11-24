/**
 * The wire contract, in TypeScript — the browser end of `board_reader_core`.
 *
 * Every message shared with the Python side — the L3 position events and motion
 * updates on `/ws/viewer` (DESIGN §5), the L1→L2 frame headers on `/ws/ingest`
 * (DESIGN §4.5) and the camera control messages — is **generated** from the
 * pydantic models in `board_reader_core` into `protocol/generated.ts`: a zod
 * validator and a type per message, identical to the Python by construction.
 * Import those types and validators from there; `parseWire` is the one way to read
 * one off a wire.
 *
 * This module holds what is not a message format: the class vocabulary and the
 * board frame every `square_dist` is expressed in, the thresholds the pages read
 * the per-event scalars with, and the inference node's HTTP replies (`/estimate`,
 * `/uploads`, `/labels`) and its `/ingest` acks, which the node writes as plain
 * dicts and core does not model.
 *
 * **Everything crossing those wires is untrusted input**, so it is parsed once, at
 * the boundary, and what comes out the other side is a value whose type is already
 * true. A message is accepted whole or refused with a reason, exactly as the node
 * accepts or refuses it: the validators are strict where pydantic is strict.
 */

import type { z } from 'zod';

import { CameraConfig, PositionEvent, type FrameSize, type PriorRepair } from './generated';
import { err, ok, type Result } from '../result';

// ---- the 13-class vocabulary -------------------------------------------------

/** Index order is canonical and travels in every event (`protocol.CLASSES`). */
export const CLASSES: PositionEvent['classes'] = [
  'empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'
];

export type ClassName = (typeof CLASSES)[number];
/** Everything except `empty` — i.e. what a FEN placement field can spell. */
export type PieceSymbol = Exclude<ClassName, 'empty'>;

export const NUM_CLASSES = CLASSES.length;
export const EMPTY_CLASS_INDEX = 0;

/** Is `ch` one of the 12 piece letters? The narrowing every FEN reader needs. */
export function isPieceSymbol(ch: string): ch is PieceSymbol {
  return ch !== 'empty' && (CLASSES as readonly string[]).includes(ch);
}

// ---- the board frame ---------------------------------------------------------

export const GRID = 8;

/** Exactly eight of something. Rows run rank 8 → rank 1, columns file a → h. */
export type Grid8<T> = readonly [T, T, T, T, T, T, T, T];
/** An 8×8 board of `T` in the grid frame every `square_dist` is expressed in. */
export type Board8<T> = Grid8<Grid8<T>>;
/** The canonical 8×8×13 output (decision #14): placement and confidence derive from it. */
export type SquareDist = PositionEvent['square_dist'];
/** One square's distribution over `CLASSES`, in that index order. */
export type ClassScores = SquareDist[number][number];

export type FileLetter = 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h';
export type RankDigit = '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8';
/** Algebraic square name — also a chessground key. */
export type SquareKey = `${FileLetter}${RankDigit}`;

export const FILE_LETTERS: Grid8<FileLetter> = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
/** Indexed by grid ROW, so row 0 (rank 8) reads '8'. */
export const ROW_RANKS: Grid8<RankDigit> = ['8', '7', '6', '5', '4', '3', '2', '1'];

/**
 * Is `square` an algebraic square name? The wire's square fields are validated as
 * such (`^[a-h][1-8]$`), but a pattern cannot narrow a `string`; this can.
 */
export function isSquareKey(square: string): square is SquareKey {
  return /^[a-h][1-8]$/.test(square);
}

export const START_PLACEMENT = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
export const EMPTY_PLACEMENT = '8/8/8/8/8/8/8/8';

/**
 * The one place an array becomes a `Grid8`. Fixing the length in the type is what
 * lets every consumer index a rank without a `| undefined` in its way, so the
 * assertion is worth having — but only behind a real check, and only here.
 *
 * A wrong length is a bug in this library, never wire data: everything parsed from
 * the wire is length-checked and *refused*, not asserted.
 */
export function toGrid8<T>(items: readonly T[]): Grid8<T> {
  if (items.length !== GRID) {
    throw new RangeError(`expected ${GRID} items, got ${items.length}`);
  }
  return items as unknown as Grid8<T>;
}

// ---- parsing -------------------------------------------------------------------

export type ParseResult<T> = Result<T, string>;

/**
 * `raw` (decoded JSON, or the JSON text a socket delivers) read as `schema`, or the
 * reason it is not one: each problem prefixed with where it is, in one line, as
 * `board_reader_core.wiremodel.reason` words it.
 */
export function parseWire<S extends z.ZodType>(schema: S, raw: unknown): ParseResult<z.output<S>> {
  let data = raw;
  if (typeof raw === 'string') {
    try {
      data = JSON.parse(raw) as unknown;
    } catch {
      return err('not JSON');
    }
  }
  const parsed = schema.safeParse(data);
  if (parsed.success) return ok(parsed.data);
  return err(
    parsed.error.issues
      .map((issue) => (issue.path.length ? `${issue.path.join('.')}: ${issue.message}` : issue.message))
      .join('; ')
  );
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

// ---- the camera control channel (`board_reader_core.control`) ---------------
// The phone is only a camera (DESIGN D11, "dumb camera"): the control UI sets what
// it films, the node holds that per board, and pushes it to the phone.

export type Resolution = CameraConfig['resolution'];
/** The capture resolutions the control UI offers, in the node's order. */
export const RESOLUTION_NAMES: readonly Resolution[] = CameraConfig.shape.resolution.options;
/** What each resolution asks the camera for, as `ideal` constraints (`control.RESOLUTIONS`). */
export const RESOLUTIONS: Readonly<Record<Resolution, FrameSize>> = {
  '720p': { width: 1280, height: 720 },
  '1080p': { width: 1920, height: 1080 },
  '2160p': { width: 3840, height: 2160 }
};

export const DEFAULT_CAMERA_CONFIG: CameraConfig = {
  crop: null,
  device: null,
  resolution: '1080p',
  streaming: false
};

// ---- the /ingest ack -------------------------------------------------------------

/** The node's per-frame reply on `/ingest`: an ack, or a refusal of that one frame. */
export type IngestReply =
  | { readonly kind: 'ack'; readonly seq: number; readonly board_id: string }
  | { readonly kind: 'error'; readonly error: string };

export function parseIngestReply(raw: string): IngestReply | null {
  let data: unknown;
  try {
    data = JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
  if (!isRecord(data)) return null;
  if (typeof data['ack'] === 'number' && typeof data['board_id'] === 'string') {
    return { kind: 'ack', seq: data['ack'], board_id: data['board_id'] };
  }
  if (typeof data['error'] === 'string') return { kind: 'error', error: data['error'] };
  return null;
}

// ---- how the UI reads an event ---------------------------------------------------

/**
 * Warnings ride as node-authored strings and are shown verbatim. Enumerating them
 * in a union here would add a second place to edit for every warning the node
 * learns to raise — the same argument the viewer's catch-all `/api` route
 * (`routes/api/[...path]`) makes for relaying every path. The one the UI actually
 * *branches* on is named below.
 */
export const WARN_LOW_LIGHT = 'low_light';

/**
 * The threshold the node raises that warning at (`server.LOW_LIGHT_MEAN`), mirrored
 * so a page can show the same verdict on the photo in hand *before* posting it —
 * which is the only moment the operator can still do something about it.
 */
export const LOW_LIGHT_MEAN = 55.0;

/**
 * How the UI reads the two per-event scalars. Both are interpretations of the
 * protocol rather than of any one page, and every page that shows a "moving" chip
 * or rings a doubtful square must agree with the others.
 *
 * `stability` under `MOVING_BELOW` reads as "moving" (DESIGN D3 — a hint, never a
 * gate); a square whose max class probability is under `LOW_CONFIDENCE` stops
 * asserting a piece and carries a ring.
 */
export const MOVING_BELOW = 0.7;
export const LOW_CONFIDENCE = 0.5;

// ---- HTTP replies (`board_reader_inference.server`) --------------------------------

/** `POST /estimate`: the position event, plus what only a one-shot read carries. */
export interface EstimateResponse {
  readonly event: PositionEvent;
  /** The event's own `prior`, lifted out: the board the page establishes as its baseline. */
  readonly prior: PriorRepair | null;
  /** Names the retained photo, so this read's position can be confirmed via `/labels`. */
  readonly upload: string | null;
  readonly warnings: readonly string[];
}

/** One retained photo as `GET /uploads` lists it. */
export interface UploadRecord {
  readonly id: string;
  /** Epoch seconds, as the node records them. */
  readonly ts: number | null;
  readonly kind: string;
  readonly placement: string | null;
  readonly labeled: boolean;
  readonly label_placement: string | null;
  readonly label_corrections: number | null;
}

export interface UploadsResponse {
  /** False when `BR_SAVE_UPLOADS` is unset: not an error, an explainer for the page. */
  readonly retention: boolean;
  readonly uploads: readonly UploadRecord[];
}

export interface LabelResponse {
  readonly saved: string;
  readonly placement: string;
  readonly corrections: number;
}

/**
 * The reply is the event's own JSON with `upload` and `warnings` added beside it,
 * so those two are taken off and the rest must be a position event.
 */
export function parseEstimateResponse(raw: unknown): ParseResult<EstimateResponse> {
  if (!isRecord(raw)) return err('not a JSON object');
  const { upload, warnings, ...rest } = raw;
  const event = parseWire(PositionEvent, rest);
  if (!event.ok) return event;
  return ok({
    event: event.value,
    prior: event.value.prior ?? null,
    upload: asString(upload),
    warnings: Array.isArray(warnings) ? warnings.filter((w): w is string => typeof w === 'string') : []
  });
}

/**
 * Liberal on the timestamp: an ISO string, epoch seconds or epoch milliseconds all
 * parse. A row whose timestamp is unreadable still belongs in the list — it just
 * sorts last (snapshot.html `parseTs`).
 */
export function parseUploadTimestamp(ts: unknown): number | null {
  const n = asFiniteNumber(ts);
  if (n !== null) return n > 1e12 ? n / 1000 : n;
  const s = asString(ts);
  if (s === null) return null;
  const parsed = Date.parse(s);
  return Number.isNaN(parsed) ? null : parsed / 1000;
}

export function parseUploadsResponse(raw: unknown): ParseResult<UploadsResponse> {
  if (!isRecord(raw)) return err('not a JSON object');
  const list = Array.isArray(raw['uploads']) ? raw['uploads'] : [];
  const uploads: UploadRecord[] = [];
  for (const row of list) {
    if (!isRecord(row)) continue;
    const id = asString(row['id']);
    if (id === null) continue;
    uploads.push({
      id,
      ts: parseUploadTimestamp(row['ts']),
      kind: asString(row['kind']) ?? 'estimate',
      placement: asString(row['placement']),
      labeled: row['labeled'] === true,
      label_placement: asString(row['label_placement']),
      label_corrections: asFiniteNumber(row['label_corrections'])
    });
  }
  // `retention: false` is a real answer (retention is off); anything else the node
  // might send is treated as retention being on, which is what the pages assumed.
  return ok({ retention: raw['retention'] !== false, uploads });
}

export function parseLabelResponse(raw: unknown): ParseResult<LabelResponse> {
  if (!isRecord(raw)) return err('not a JSON object');
  const placement = asString(raw['placement']);
  if (placement === null) return err('missing placement');
  return ok({
    saved: asString(raw['saved']) ?? '',
    placement,
    corrections: asFiniteNumber(raw['corrections']) ?? 0
  });
}
