/**
 * The wire contract, in TypeScript — the browser end of `board_reader_core`.
 *
 * Two wires meet in this UI and both are described here:
 *
 *  - **L3 position events** (`board_reader_core.protocol`, DESIGN §5) arrive on
 *    `/ws/viewer` and are what the live page renders.
 *  - **L1→L2 frames** (`board_reader_core.wire`, DESIGN §4.5) leave on
 *    `/ws/ingest`: a JSON header, then its JPEG.
 *
 * Plus the inference node's HTTP replies (`board_reader_inference.server`), which
 * are the same contract seen through one-shot requests.
 *
 * **Everything crossing those wires is untrusted input**, so this module is also
 * the only place that parses it. The three hand-written pages each defended
 * against malformed data at render time and each defended slightly differently;
 * here it is done once, at the boundary, and what comes out the other side is a
 * value whose type is already true. Field names are kept exactly as they are on
 * the wire (snake_case) so a reader can hold this file and `protocol.py` side by
 * side and see one contract, not two.
 *
 * Parsing is **strict about identity and lenient about decoration**: an event
 * without `board_id`/`seq`/`utc`/`placement` is not an event and is refused, while
 * an unrecognised `status` or a missing `square_dist` falls back to the protocol
 * default. A viewer that dropped a whole reading over a chip it could not label
 * would show a stale board, which is the worse failure.
 */

import { err, ok, type Result } from './result';

// ---- the 13-class vocabulary -------------------------------------------------

/** Index order is canonical and travels in every event (`protocol.CLASSES`). */
export const CLASSES = [
  'empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'
] as const;

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
/** One square's distribution over `CLASSES`, in that index order. */
export type ClassScores = readonly [
  number, number, number, number, number, number, number,
  number, number, number, number, number, number
];
/** The canonical 8×8×13 output (decision #14): placement and confidence derive from it. */
export type SquareDist = Board8<ClassScores>;

export type FileLetter = 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h';
export type RankDigit = '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8';
/** Algebraic square name — also a chessground key. */
export type SquareKey = `${FileLetter}${RankDigit}`;

export const FILE_LETTERS: Grid8<FileLetter> = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
/** Indexed by grid ROW, so row 0 (rank 8) reads '8'. */
export const ROW_RANKS: Grid8<RankDigit> = ['8', '7', '6', '5', '4', '3', '2', '1'];

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

// ---- event vocabulary --------------------------------------------------------

/**
 * `status` deliberately has no `illegal` value — legality is not ours to judge
 * (DESIGN §5 / D1). An illegal-looking placement is reported as `ok`.
 */
export const POSITION_STATUSES = ['ok', 'recalibrating', 'needs_setup', 'stale'] as const;
export type PositionStatus = (typeof POSITION_STATUSES)[number];

/** Which side's clock is running (DESIGN D7). `none` = neither/unknown. */
export const CLOCK_STATES = ['white', 'black', 'none'] as const;
export type ClockRunning = (typeof CLOCK_STATES)[number];

/** The only frame the estimator names today (`protocol.DEFAULT_ORIENTATION`). */
export const DEFAULT_ORIENTATION = 'a1_bottom_left';
export type BoardOrientation = typeof DEFAULT_ORIENTATION;

/**
 * One L3 position event (DESIGN §5).
 *
 * `confidence` and `stability` are `number | null` rather than defaulted numbers
 * because the pages read a missing `stability` as "motion unknown" and must keep
 * being able to: a defaulted 1.0 would claim the board is at rest on a stream that
 * never said so. `square_dist` is likewise explicitly absent rather than empty.
 */
export interface PositionEvent {
  readonly board_id: string;
  readonly seq: number;
  readonly utc: string;
  /** FEN field 1, verbatim from the wire — sanitise it with `fenToGrid`. */
  readonly placement: string;
  readonly confidence: number | null;
  readonly stability: number | null;
  readonly status: PositionStatus;
  readonly orientation: string;
  readonly clock: ClockRunning;
  readonly square_dist: SquareDist | null;
  /**
   * The node's hard-constraint repair of this read (`PositionEvent.prior` on the
   * wire): the nearest *possible* board, beside the raw one. Boards are drawn
   * from `prior.placement` when present — `placement` above can spell two kings.
   */
  readonly prior: PriorDecode | null;
  /**
   * The node's ingest-stabilizer debounce (`PositionEvent.settled` on the wire):
   * a square must hold the same class for several consecutive reads before this
   * board adopts it, so a display that must not flicker draws
   * `settled.placement` — ahead of `prior.placement`, which the raw read still
   * takes when this is absent. Only ever present on a streamed reading; a
   * one-shot `/estimate` has nothing to settle over.
   */
  readonly settled: SettledBoard | null;
}

// ---- geometry on the wire ----------------------------------------------------

export type Point = readonly [x: number, y: number];
/** Four corners in canonical a8,h8,h1,a1 order — the order every endpoint takes. */
export type CornerQuad = readonly [Point, Point, Point, Point];
/** The same quad flattened, which is how `wire.Frame.header` carries it. */
export type CornerCoords = readonly [
  number, number, number, number, number, number, number, number
];

export function flattenQuad(quad: CornerQuad): CornerCoords {
  return [
    quad[0][0], quad[0][1], quad[1][0], quad[1][1],
    quad[2][0], quad[2][1], quad[3][0], quad[3][1]
  ];
}

/**
 * `x0,y0,…,x3,y3` — the spelling `?corners=` takes and `X-Corners` answers in.
 * One function so a page can never post a quad in an order the node reads
 * differently (`server._parse_corners`).
 */
export function cornersParam(quad: CornerQuad, digits = 0): string {
  const round = (v: number) => {
    const factor = 10 ** digits;
    return String(Math.round(v * factor) / factor);
  };
  return flattenQuad(quad).map(round).join(',');
}

/**
 * Inverse of `cornersParam`, tolerant by design: an unparseable string is *no*
 * quad, not a broken one. `scale` rebases the corners when the image they were
 * measured on has since been downscaled (snapshot.html `parseCorners`).
 */
export function parseCornersParam(spec: string | null | undefined, scale = 1): CornerQuad | null {
  const nums = String(spec ?? '').split(',').map(Number);
  if (nums.length !== 8 || !nums.every(Number.isFinite)) return null;
  const at = (i: number) => (nums[i] as number) * scale;
  return [
    [at(0), at(1)], [at(2), at(3)], [at(4), at(5)], [at(6), at(7)]
  ];
}

// ---- L1 → L2 ingest framing --------------------------------------------------

export const FRAME_TYPE = 'frame';

/**
 * The JSON header of one ingest frame (`wire.Frame.header`).
 *
 * `corners` present means the JPEG is the ORIGINAL camera image and the board is
 * that quad *in the sent image's own pixels*; absent means the JPEG is already a
 * rectified crop. Corners ride on every frame by design — there is no
 * `set_corners` control message, so a reconnect or a mid-stream recalibration
 * needs no handshake.
 */
export interface IngestFrameHeader {
  readonly type: typeof FRAME_TYPE;
  readonly board_id: string;
  readonly seq: number;
  /** Epoch SECONDS, stamped at capture on the edge (DESIGN D5) — not milliseconds. */
  readonly capture_ts: number;
  readonly stability: number;
  /** Edge focus score (tenengrad) — comparable only between neighbouring frames. */
  readonly sharpness?: number;
  readonly burst_id: string | null;
  readonly corners?: CornerCoords;
  readonly clock?: ClockRunning;
}

/** The node's per-frame reply on `/ingest`: an ack, or a refusal of that one frame. */
export type IngestReply =
  | { readonly kind: 'ack'; readonly seq: number; readonly board_id: string }
  | { readonly kind: 'error'; readonly error: string };

export function parseIngestReply(raw: string): IngestReply | null {
  const data: unknown = safeJson(raw);
  if (!isRecord(data)) return null;
  if (typeof data['ack'] === 'number' && typeof data['board_id'] === 'string') {
    return { kind: 'ack', seq: data['ack'], board_id: data['board_id'] };
  }
  if (typeof data['error'] === 'string') return { kind: 'error', error: data['error'] };
  return null;
}

// ---- HTTP replies ------------------------------------------------------------

/** `POST /calibrate` — the accepted case. A refusal comes back as `{error}` + 422. */
export interface CalibrateResponse {
  readonly classes_calibrated: number;
  readonly backbone: string;
  /** Where the bank was persisted, or null when persistence is off (`BR_SESSION_BANK`). */
  readonly saved: string | null;
  readonly exemplars: number;
  /** Quarter-turns the node had to apply to the labels to read the board. */
  readonly label_rotation: number;
}

/** `POST /detect` — geometric TL,TR,BR,BL. Which corner is a8 is NOT knowable (decision #15). */
export interface DetectResponse {
  readonly corners: CornerQuad;
}

/** Base64 JPEGs from `?debug=1`: the sampled boxes, and the 8×8 crop mosaic. */
export interface DebugImages {
  readonly boxes: string;
  readonly mosaic: string;
}

/** One relabelling the prior decoder applied (`prior.Flip`). */
export interface PriorFlip {
  readonly square: string;
  readonly from: string;
  readonly to: string;
  /** Margin given up, in nats. May be negative on a square's second flip. */
  readonly cost: number;
  readonly constraint: string;
}

/**
 * The nearest possible board, with its audit trail (`prior.PriorDecode`).
 * `residual` is SPARSE — only the constraints still violated appear in it.
 */
export interface PriorDecode {
  readonly placement: string;
  readonly flips: readonly PriorFlip[];
  readonly residual: Readonly<Record<string, number>>;
}

/**
 * A temporally debounced board, served beside the raw read (`prior.SettledBoard`
 * on the wire, sibling of `PriorDecode`). `pending` names the squares currently
 * reading differently from `placement` — board order, may be empty.
 */
export interface SettledBoard {
  readonly placement: string;
  readonly pending: readonly string[];
}

export interface AutoCalibration {
  readonly matches: number;
  readonly rotation: number;
  readonly exemplars: number;
  readonly saved: string | null;
}

export interface BankRefresh {
  readonly added: number;
  readonly total: number;
  readonly agree: number;
  readonly rotation: number;
  readonly saved: string | null;
}

/**
 * Warnings ride as node-authored strings and are shown verbatim. Enumerating them
 * in a union here would add a second place to edit for every warning the node
 * learns to raise — the same argument `proxy.py` makes for its catch-all `/api`
 * route. The two the UI actually *branches* on are named below.
 */
export const WARN_LOW_LIGHT = 'low_light';
export const WARN_VERY_LOW_ANGLE = 'very_low_angle';

/**
 * The thresholds the node raises those two warnings at (`server.LOW_LIGHT_MEAN`,
 * `boarddetect.LOW_ANGLE_ASPECT`), mirrored so a page can show the same verdict on
 * the photo in hand *before* posting it — which is the only moment the operator
 * can still do something about it.
 */
export const LOW_LIGHT_MEAN = 55.0;
export const LOW_ANGLE_ASPECT = 0.62;

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

/**
 * `POST /estimate`. Two genuinely different answers, so they are two variants:
 * with no piece-type calibration for this set the node answers occupancy only
 * (DESIGN §4.6), and that reply has no placement, no distribution and nothing a
 * human could confirm. The wire discriminates by the PRESENCE of `mode`
 * (`"occupancy"`); `kind` is this module's explicit stand-in so a consumer cannot
 * read a field the other branch never sends.
 */
export type EstimateResponse =
  | {
      readonly kind: 'occupancy';
      readonly board_id: string;
      readonly occupied: Board8<boolean>;
      readonly hint: string;
      readonly warnings: readonly string[];
      readonly debug: DebugImages | null;
    }
  | {
      readonly kind: 'position';
      readonly event: PositionEvent;
      readonly prior: PriorDecode | null;
      readonly autocalibrated: AutoCalibration | null;
      readonly bank_refreshed: BankRefresh | null;
      /** Degrees (0/90/180/270) the node turned the reading by under `?autorotate=1`. */
      readonly autorotated: number;
      /** Names the retained photo, so this read's position can be confirmed via `/labels`. */
      readonly upload: string | null;
      readonly warnings: readonly string[];
      readonly debug: DebugImages | null;
    };

/** One retained photo as `GET /uploads` lists it. */
export interface UploadRecord {
  readonly id: string;
  /** Epoch seconds, as the node records them. */
  readonly ts: number | null;
  readonly kind: string;
  /** `x0,y0,…` as posted, in the stored photo's own pixels — feed to `parseCornersParam`. */
  readonly corners: string | null;
  readonly placement: string | null;
  readonly autorotated: number | null;
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

// ---- parsing helpers ---------------------------------------------------------

export type ParseResult<T> = Result<T, string>;

function safeJson(raw: string): unknown {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return undefined;
  }
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

function asMember<T extends string>(
  value: unknown,
  members: readonly T[],
  fallback: T
): T {
  return typeof value === 'string' && (members as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
}

/** 8×8×13 finite numbers, or null. Anything else is "no distribution", never a partial one. */
export function parseSquareDist(value: unknown): SquareDist | null {
  if (!Array.isArray(value) || value.length !== GRID) return null;
  const rows: Grid8<ClassScores>[] = [];
  for (const rawRow of value) {
    if (!Array.isArray(rawRow) || rawRow.length !== GRID) return null;
    const cells: ClassScores[] = [];
    for (const rawCell of rawRow) {
      if (!Array.isArray(rawCell) || rawCell.length !== NUM_CLASSES) return null;
      if (!rawCell.every((p) => typeof p === 'number' && Number.isFinite(p))) return null;
      // Length and finiteness are checked on the two lines above; this is the
      // only place a wire array becomes a fixed-width `ClassScores`.
      cells.push(rawCell as unknown as ClassScores);
    }
    rows.push(toGrid8(cells));
  }
  return toGrid8(rows);
}

function parseBooleanBoard(value: unknown): Board8<boolean> | null {
  if (!Array.isArray(value) || value.length !== GRID) return null;
  const rows: Grid8<boolean>[] = [];
  for (const rawRow of value) {
    if (!Array.isArray(rawRow) || rawRow.length !== GRID) return null;
    rows.push(toGrid8(rawRow.map((cell) => Boolean(cell))));
  }
  return toGrid8(rows);
}

function parseWarnings(value: unknown): readonly string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((w): w is string => typeof w === 'string');
}

function parseDebug(value: unknown): DebugImages | null {
  if (!isRecord(value)) return null;
  const boxes = asString(value['boxes']);
  const mosaic = asString(value['mosaic']);
  // Both or neither: the page shows them as a pair and a half-pair would render a
  // broken image rather than the honest "no debug images for this read".
  return boxes !== null && mosaic !== null ? { boxes, mosaic } : null;
}

// ---- the position event ------------------------------------------------------

/**
 * Parse one `/ws/viewer` frame. `board_id`, `seq`, `utc` and `placement` are the
 * event's identity: without them there is nothing to render or to key by, so a
 * frame missing any of them is refused with a reason rather than rendered as a
 * board full of holes.
 */
export function parsePositionEvent(raw: unknown): ParseResult<PositionEvent> {
  const data = typeof raw === 'string' ? safeJson(raw) : raw;
  if (!isRecord(data)) return err('not a JSON object');

  const board_id = asString(data['board_id']);
  if (board_id === null || board_id === '') return err('missing board_id');
  const seq = asFiniteNumber(data['seq']);
  if (seq === null || seq < 0 || !Number.isInteger(seq)) return err('missing or invalid seq');
  const utc = asString(data['utc']);
  if (utc === null) return err('missing utc');
  const placement = asString(data['placement']);
  if (placement === null) return err('missing placement');

  const clock = data['clock'];
  return ok({
    board_id,
    seq,
    utc,
    placement,
    confidence: asFiniteNumber(data['confidence']),
    stability: asFiniteNumber(data['stability']),
    status: asMember(data['status'], POSITION_STATUSES, 'ok'),
    orientation: asString(data['orientation']) ?? DEFAULT_ORIENTATION,
    clock: asMember(isRecord(clock) ? clock['running'] : undefined, CLOCK_STATES, 'none'),
    square_dist: parseSquareDist(data['square_dist']),
    prior: parsePrior(data['prior']),
    settled: parseSettled(data['settled'])
  });
}

// ---- HTTP replies ------------------------------------------------------------

export function parseCalibrateResponse(raw: unknown): ParseResult<CalibrateResponse> {
  if (!isRecord(raw)) return err('not a JSON object');
  const classes = asFiniteNumber(raw['classes_calibrated']);
  if (classes === null) return err('missing classes_calibrated');
  return ok({
    classes_calibrated: classes,
    backbone: asString(raw['backbone']) ?? 'unknown',
    saved: asString(raw['saved']),
    exemplars: asFiniteNumber(raw['exemplars']) ?? 0,
    label_rotation: asFiniteNumber(raw['label_rotation']) ?? 0
  });
}

export function parseDetectResponse(raw: unknown): ParseResult<DetectResponse> {
  if (!isRecord(raw)) return err('not a JSON object');
  const corners = raw['corners'];
  if (!Array.isArray(corners) || corners.length !== 4) return err('no usable quad');
  const points: Point[] = [];
  for (const p of corners) {
    if (!Array.isArray(p) || p.length !== 2) return err('no usable quad');
    const x = asFiniteNumber(p[0]);
    const y = asFiniteNumber(p[1]);
    if (x === null || y === null) return err('no usable quad');
    points.push([x, y]);
  }
  // Exactly four finite (x, y) pairs, checked in the loop above.
  return ok({ corners: points as unknown as CornerQuad });
}

function parsePrior(value: unknown): PriorDecode | null {
  if (!isRecord(value)) return null;
  const placement = asString(value['placement']);
  if (placement === null) return null;
  const rawFlips = Array.isArray(value['flips']) ? value['flips'] : [];
  const flips: PriorFlip[] = [];
  for (const f of rawFlips) {
    if (!isRecord(f)) continue;
    const square = asString(f['square']);
    const from = asString(f['from']);
    const to = asString(f['to']);
    if (square === null || from === null || to === null) continue;
    flips.push({
      square,
      from,
      to,
      cost: asFiniteNumber(f['cost']) ?? 0,
      constraint: asString(f['constraint']) ?? 'unknown'
    });
  }
  const residual: Record<string, number> = {};
  if (isRecord(value['residual'])) {
    for (const [key, count] of Object.entries(value['residual'])) {
      const n = asFiniteNumber(count);
      if (n !== null) residual[key] = n;
    }
  }
  return { placement, flips, residual };
}

function parseSettled(value: unknown): SettledBoard | null {
  if (!isRecord(value)) return null;
  const placement = asString(value['placement']);
  if (placement === null) return null;
  const rawPending = Array.isArray(value['pending']) ? value['pending'] : [];
  const pending = rawPending.filter((s): s is string => typeof s === 'string');
  return { placement, pending };
}

function parseAutoCalibration(value: unknown): AutoCalibration | null {
  if (!isRecord(value)) return null;
  return {
    matches: asFiniteNumber(value['matches']) ?? 0,
    rotation: asFiniteNumber(value['rotation']) ?? 0,
    exemplars: asFiniteNumber(value['exemplars']) ?? 0,
    saved: asString(value['saved'])
  };
}

function parseBankRefresh(value: unknown): BankRefresh | null {
  if (!isRecord(value)) return null;
  return {
    added: asFiniteNumber(value['added']) ?? 0,
    total: asFiniteNumber(value['total']) ?? 0,
    agree: asFiniteNumber(value['agree']) ?? 0,
    rotation: asFiniteNumber(value['rotation']) ?? 0,
    saved: asString(value['saved'])
  };
}

export function parseEstimateResponse(raw: unknown): ParseResult<EstimateResponse> {
  if (!isRecord(raw)) return err('not a JSON object');
  const warnings = parseWarnings(raw['warnings']);
  const debug = parseDebug(raw['debug']);

  if (raw['mode'] === 'occupancy') {
    const occupied = parseBooleanBoard(raw['occupied']);
    // Occupancy IS the answer in this branch; without the grid there is nothing
    // to show, so this is a refusal rather than an empty board.
    if (occupied === null) return err('occupancy reply carries no 8x8 occupied grid');
    return ok({
      kind: 'occupancy',
      board_id: asString(raw['board_id']) ?? 'snapshot',
      occupied,
      hint: asString(raw['hint']) ?? 'uncalibrated for this piece set — showing occupancy only',
      warnings,
      debug
    });
  }

  const event = parsePositionEvent(raw);
  if (!event.ok) return event;
  return ok({
    kind: 'position',
    event: event.value,
    prior: parsePrior(raw['prior']),
    autocalibrated: parseAutoCalibration(raw['autocalibrated']),
    bank_refreshed: parseBankRefresh(raw['bank_refreshed']),
    autorotated: asFiniteNumber(raw['autorotated']) ?? 0,
    upload: asString(raw['upload']),
    warnings,
    debug
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
      corners: asString(row['corners']),
      placement: asString(row['placement']),
      autorotated: asFiniteNumber(row['autorotated']),
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
