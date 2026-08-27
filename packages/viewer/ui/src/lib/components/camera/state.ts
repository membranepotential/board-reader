/**
 * The /camera page's state, as types rather than as sixty functions over twenty
 * loose booleans.
 *
 * The hand-written page this replaces carried `pts` (sometimes four points,
 * sometimes zero), `quadKnown`, `dragIdx`, `detectBusy`, `calibBusy`, `streaming`,
 * `trackState`, `frameW/frameH` and `cameraLiveAt` as independent variables, and
 * every one of the roughly sixty functions began by re-deriving which combination
 * it was in. Most of the combinations were illegal — a quad with no frame to
 * measure it against, a drag and a detection owning the corners at once — and
 * nothing said so.
 *
 * Three unions replace all of it:
 *
 *  - `CameraState` — the camera is idle, being asked for, live, or blocked. Only
 *    `live` carries a frame size AND the framing, because a quad is expressed in
 *    that frame's pixels and means nothing without it. `pts.length !== 4` and
 *    "corners measured against the old resolution" both stop being expressible.
 *  - `Framing` — four points plus a labelling plus where they CAME from. `origin`
 *    replaces `quadKnown`: automatic detection exists precisely to fill the gap
 *    where the quad is a placeholder, and the difference between "the operator
 *    framed this" and "we guessed" is the thing that gates it.
 *  - `Activity` — exactly one thing may own the camera and the quad at a time: a
 *    drag, a detection, a calibration, or nothing. The old three-boolean guard
 *    `detectBusy || calibBusy || dragIdx !== null` becomes `kind !== 'idle'`.
 *
 * Everything here is pure and DOM-free so it is tested directly (`state.test.ts`);
 * the page is left with wiring, and the numbers below are the ones the browser
 * measured.
 */

import { asRotation, type FrameSize, type LabelRotation } from '$lib/geometry/corners';
import type { TrackerState } from '$lib/geometry/tracker';
import { isRecord, type CornerQuad, type Point } from '$lib/protocol';

// ---- capture constants -------------------------------------------------------

/** Downscale before encoding: the node samples squares, not pores. */
export const SEND_LONG_SIDE = 960;
export const JPEG_QUALITY = 0.75;
/** What `/detect` is shown — the snapshot page's `MAX_SRC`. */
export const DETECT_LONG_SIDE = 1600;
/** Calibration reads one exemplar per square, so it gets the good pixels. */
export const CALIBRATE_QUALITY = 0.95;

/**
 * Calibration grabs a BURST and keeps the sharpest frame of it (`$lib/edge/sharpness`).
 *
 * A phone is handheld, and one instantaneous grab lands on whichever 1/30 s the
 * operator's thumb fell on — often a smeared one, and that one frame is the bank for
 * the rest of the session. Ten candidates over about a second and a half is long
 * enough for a hand to pass through at least one still moment, and short enough that
 * nobody wonders whether the button worked. The corners are not held still across the
 * burst and do not need to be: the tracker keeps following the board, so each
 * candidate is paired with the quad measured at its own tick.
 */
export const CALIBRATE_SAMPLES = 10;
export const CALIBRATE_SAMPLE_MS = 150;
/** Long side of the scoring image — sharpness is a ranking, and 256 px ranks fine. */
export const CALIBRATE_SCORE_LONG_SIDE = 256;

/**
 * Tracking working resolution. Measured rather than guessed: ~2 ms a frame here,
 * so a phone stays inside the 50 ms the capture loop has, while still giving
 * Lucas-Kanade a square's worth of texture to lock onto.
 */
export const TRACK_LONG_SIDE = 400;
/** Seed points per square: 1 -> the board's own 9x9 lattice of square corners. */
export const TRACK_SUBDIVISIONS = 1;
export const TRACK_ITERATIONS = 10;
/** A tracked quad is persisted at most this often — the loop runs 20 times a second. */
export const TRACK_SAVE_MS = 2000;

export const BANNER_MS = 9000;
/** Status repaint period; the capture loop runs far faster and does not wait on it. */
export const TICK_MS = 200;
export const RATE_WINDOW_MS = 3000;

// ---- storage keys ------------------------------------------------------------
// Spelled exactly as the hand-written page spelled them: a phone that has already
// framed its board must keep that crop across this rewrite.

export const BOARD_KEY = 'board-reader.camera.board';
export const TRACKING_KEY = 'board-reader.camera.tracking';
export const LENS_KEY = 'board-reader.camera.lens';
export const PANEL_KEY = 'board-reader.camera.position';
export function cornerKey(boardId: string): string {
  return `board-reader.camera.corners.${boardId}`;
}

// ---- the unions --------------------------------------------------------------

/** Which of the four storage slots a handle is. */
export type CornerIndex = 0 | 1 | 2 | 3;

/**
 * Where the current quad came from. `placeholder` is the one value that is NOT a
 * framing: it is the page's guess at a board filling the middle of the frame, so
 * the page is streamable the moment the camera opens, and automatic detection
 * exists to replace it.
 */
export type FramingOrigin = 'placeholder' | 'restored' | 'operator' | 'detected';

export interface Framing {
  /** VIDEO PIXELS of the enclosing `CameraState.frame` — never screen space. */
  readonly quad: CornerQuad;
  /** Which physical corner the operator calls a8 (DESIGN decision #15). */
  readonly rot: LabelRotation;
  readonly origin: FramingOrigin;
}

/** Has this board ever actually been framed, by a saved crop, a detection or a finger? */
export function isFramed(framing: Framing): boolean {
  return framing.origin !== 'placeholder';
}

/** Why the camera is not available. The page renders one explainer per variant. */
export type BlockedReason =
  | { readonly kind: 'insecure' }
  | { readonly kind: 'unsupported' }
  | { readonly kind: 'refused'; readonly name: string; readonly detail: string };

export type CameraState =
  | { readonly kind: 'idle' }
  | { readonly kind: 'requesting' }
  | {
      readonly kind: 'live';
      readonly frame: FrameSize;
      /** Inseparable from the frame: the quad is in that frame's pixels. */
      readonly framing: Framing;
      /** `performance.now()` when the frame size first arrived — autoexposure
       *  and autofocus start settling from here, which is what the automatic
       *  detector's settle delay is measured against. */
      readonly liveAt: number;
    }
  | { readonly kind: 'blocked'; readonly reason: BlockedReason };

/**
 * The one thing currently owning the camera and the quad. Not a set of booleans:
 * a detection during a drag, or two detections at once, are the bugs this shape
 * refuses to let anyone write.
 */
export type Activity =
  | { readonly kind: 'idle' }
  | { readonly kind: 'dragging'; readonly corner: CornerIndex }
  | { readonly kind: 'detecting'; readonly auto: boolean }
  | { readonly kind: 'calibrating' };

/** Tracking as the chip states it: the tracker's own states, plus switched off. */
export type TrackingState = TrackerState | 'off';

// ---- quads -------------------------------------------------------------------

/**
 * A board roughly filling the middle of the frame. Always a usable quad, so the
 * page is streamable the moment the camera opens and framing is a refinement
 * rather than a prerequisite.
 */
export function placeholderQuad({ width: w, height: h }: FrameSize): CornerQuad {
  return [
    [0.18 * w, 0.2 * h],
    [0.82 * w, 0.2 * h],
    [0.82 * w, 0.84 * h],
    [0.18 * w, 0.84 * h]
  ];
}

/** Frame pixels -> fractions of the frame. */
export function normaliseQuad(quad: CornerQuad, frame: FrameSize): CornerQuad {
  return scale(quad, 1 / frame.width, 1 / frame.height);
}

/** Fractions of the frame -> frame pixels. */
export function denormaliseQuad(quad: CornerQuad, frame: FrameSize): CornerQuad {
  return scale(quad, frame.width, frame.height);
}

/**
 * Re-express a quad against a frame that has changed shape under it.
 *
 * Rotating the phone can swap videoWidth/videoHeight under a LIVE track — iOS
 * Safari rotates it, Android Chrome usually does not — and the same fraction of
 * the frame is then a different physical point.
 */
export function rebaseQuad(quad: CornerQuad, from: FrameSize, to: FrameSize): CornerQuad {
  return scale(quad, to.width / from.width, to.height / from.height);
}

function scale(quad: CornerQuad, sx: number, sy: number): CornerQuad {
  const at = (p: Point): Point => [p[0] * sx, p[1] * sy];
  return [at(quad[0]), at(quad[1]), at(quad[2]), at(quad[3])];
}

/**
 * A run-time-length list of points as a four-tuple, or `null`.
 *
 * The tracker speaks `NormalisedQuad`, which is a list rather than a tuple because
 * its internals build it point by point. This is the one place that narrowing
 * happens, so nothing downstream has to re-check a length.
 */
export function toQuad(points: readonly (readonly [number, number])[]): CornerQuad | null {
  if (points.length !== 4) return null;
  const [a, b, c, d] = points;
  if (a === undefined || b === undefined || c === undefined || d === undefined) return null;
  return [a, b, c, d];
}

// ---- fitting the picture into the stage --------------------------------------

export interface Box {
  readonly w: number;
  readonly h: number;
}

export interface PictureLayout {
  /** The stage minus its own chrome — where the picture is fitted, in CSS px. */
  readonly inner: Box;
  readonly width: number;
  readonly height: number;
  readonly left: number;
  readonly top: number;
  /**
   * CSS px per VIDEO px. **One uniform number**, and that is the whole reason the
   * picture is a JS-sized wrapper instead of an `object-fit: contain` element:
   * `contain` letterboxes INSIDE the element, so the element's rect stops being
   * the picture and every touch and every handle has to route through a second
   * coordinate space. The failure mode is silent — screen space leaking into the
   * crop sends WRONG corners on the wire while the preview still looks perfect.
   */
  readonly scale: number;
}

/**
 * `k = min(box.w/src.w, box.h/src.h)` — the same fit the snapshot page uses for its
 * photo canvas, for the same reason: CSS cannot express "as big as fits, exactly
 * this aspect ratio" against a box whose height comes from the viewport.
 */
export function fitPicture(frame: FrameSize, inner: Box): PictureLayout {
  const k = Math.min(inner.w / frame.width, inner.h / frame.height);
  const width = Math.max(1, Math.floor(frame.width * k));
  const height = Math.max(1, Math.floor(frame.height * k));
  return {
    inner,
    width,
    height,
    left: Math.round((inner.w - width) / 2),
    top: Math.round((inner.h - height) / 2),
    scale: width / frame.width
  };
}

// ---- persistence -------------------------------------------------------------
// Stored as FRACTIONS of the frame, never pixels: the camera may well come back at
// a different resolution (a different phone, a different lens, a rotation), and a
// crop expressed in the old resolution's pixels would silently point at the wrong
// part of the board.

export function serialiseFraming(framing: Framing, frame: FrameSize): string {
  const pts = normaliseQuad(framing.quad, frame).map(([x, y]) => [x, y]);
  return JSON.stringify({ pts, rot: framing.rot });
}

/** A stored crop restored against the CURRENT frame, or `null` if it is not one. */
export function parseStoredFraming(raw: string | null | undefined, frame: FrameSize): Framing | null {
  if (raw === null || raw === undefined) return null;
  let saved: unknown;
  try {
    saved = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(saved)) return null;
  const stored = saved['pts'];
  if (!Array.isArray(stored) || stored.length !== 4) return null;
  const points: Point[] = [];
  for (const entry of stored as readonly unknown[]) {
    if (!Array.isArray(entry) || entry.length !== 2) return null;
    const [x, y] = entry as readonly unknown[];
    if (typeof x !== 'number' || typeof y !== 'number') return null;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    points.push([x, y]);
  }
  const quad = toQuad(points);
  if (quad === null) return null;
  const rot = saved['rot'];
  return {
    quad: denormaliseQuad(quad, frame),
    rot: typeof rot === 'number' && Number.isInteger(rot) ? asRotation(rot) : 0,
    // Somebody framed this board once, so automatic detection stands down.
    origin: 'restored'
  };
}

// ---- automatic detection -----------------------------------------------------

/**
 * A settle delay before the first attempt. `MotionMeter.update` returns 1.0 on the
 * very first frame by construction, so a bare stability gate passes on frame one —
 * before autoexposure and autofocus have settled, spending the attempt on a dark,
 * soft frame.
 */
export const AUTO_SETTLE_MS = 2500;
/**
 * A stability floor stricter than `MOVING_BELOW`: a detect frame should be
 * exemplar-grade, not merely "the camera is not moving".
 */
export const AUTO_STABILITY = 0.9;
/**
 * Loss must be SUSTAINED. `PlaneTracker.lose()` freezes the quad and keeps tracking
 * from it precisely so a hand sweeping over the board recovers by itself, so `lost`
 * fires on essentially every move of a real game; re-detecting on the raw flag would
 * re-detect mid-game, every move.
 */
export const AUTO_LOST_MS = 6000;
/**
 * Exponential back-off with a give-up. `/detect` returns in ~0.1 s on a hit but
 * 2-4 s on a MISS, because a miss exhausts every coarse candidate. A page pointed
 * at a wall must never ask forever, and never on a flat timer. Browser-measured
 * against a wall: four attempts, gaps of about 8.7 / 18.5 / 32.7 s, then silence.
 */
export const AUTO_BACKOFF_MS: readonly number[] = [4000, 8000, 16000, 32000];

export interface AutoDetect {
  /** Consecutive misses. A hit zeroes it; four is giving up. */
  readonly fails: number;
  /** `performance.now()` before which no attempt is made. */
  readonly nextAt: number;
  /** When the tracker first reported `lost` in the current run of losses. */
  readonly lostSince: number | null;
}

export const AUTO_DETECT_START: AutoDetect = { fails: 0, nextAt: 0, lostSince: null };

export function autoGaveUp(auto: AutoDetect): boolean {
  return auto.fails >= AUTO_BACKOFF_MS.length;
}

/** Every attempt costs the back-off, hit or miss; a hit has already zeroed the count. */
export function backoffFor(fails: number): number {
  return AUTO_BACKOFF_MS[Math.min(fails, AUTO_BACKOFF_MS.length - 1)] as number;
}

/** Fold this frame's tracker state into the sustained-loss clock. */
export function noteTracking(auto: AutoDetect, tracking: TrackingState, now: number): AutoDetect {
  if (tracking !== 'lost') return auto.lostSince === null ? auto : { ...auto, lostSince: null };
  return auto.lostSince === null ? { ...auto, lostSince: now } : auto;
}

export interface AutoDetectGate {
  readonly auto: AutoDetect;
  readonly now: number;
  readonly camera: CameraState;
  readonly activity: Activity;
  /** A hidden page is not looking at anything. */
  readonly hidden: boolean;
  readonly stability: number;
}

/**
 * The common case should need no framing step at all: point the phone at the board
 * and the page finds it. Four gates, each of them a measurement rather than an
 * intuition — see the constants above for what each one was measured against.
 */
export function shouldAutoDetect({
  auto,
  now,
  camera,
  activity,
  hidden,
  stability
}: AutoDetectGate): boolean {
  // Nothing else may own the camera or the quad: not a drag, not a calibration,
  // not another detection.
  if (activity.kind !== 'idle' || hidden) return false;
  if (camera.kind !== 'live' || autoGaveUp(auto)) return false;
  if (now - camera.liveAt < AUTO_SETTLE_MS || now < auto.nextAt) return false;
  if (stability < AUTO_STABILITY) return false;
  const sustainedLoss = auto.lostSince !== null && now - auto.lostSince >= AUTO_LOST_MS;
  // A RESTORED framing is last session's quad over this session's world: the
  // board, the tripod or the perspective may all have changed while the page was
  // closed (2026-08-27: a stale restored quad read the board a full rank off and
  // correctly starved the zero-tap calibrate). Verify it with one detection as
  // soon as the scene is still; success re-origins the framing to 'detected'.
  return !isFramed(camera.framing) || sustainedLoss || camera.framing.origin === 'restored';
}

/**
 * Largest SAME-INDEX corner displacement between two quads, as a fraction of the
 * frame diagonal. Pointwise on purpose: two quads over the same physical corners
 * in a different storage order are NOT a small shift — that is a relabelling, and
 * the a8-check warning must fire for it.
 */
export function quadShift(a: CornerQuad, b: CornerQuad, frame: FrameSize): number {
  const diagonal = Math.hypot(frame.width, frame.height);
  let worst = 0;
  for (let i = 0; i < 4; i++) {
    const p = a[i] as Point;
    const q = b[i] as Point;
    worst = Math.max(worst, Math.hypot(p[0] - q[0], p[1] - q[1]));
  }
  return worst / diagonal;
}

/** A verifying re-detection landing within this is "the same framing, confirmed". */
export const QUAD_VERIFY_SHIFT = 0.03;

// ---- calibration guard -------------------------------------------------------

/**
 * Why calibrate cannot run right now, in the words the operator is shown, or
 * `null` when it can.
 *
 * Deliberately no branch for stability. A stability gate shipped
 * here once and made calibration UNREACHABLE on the device the page is for: the meter
 * is tuned for a camera on a stand (DESIGN D3 mirrors it value-for-value with the
 * Python edge), and a handheld phone's micro-shake parks it under `MOVING_BELOW`
 * essentially always. D3 also says in so many words that stability is a hint and
 * never a gate — and the frame is answered on its merits anyway, twice over: the page
 * keeps the sharpest of a burst (`CALIBRATE_SAMPLES`), and the node refuses a frame
 * that does not show the position it claims.
 *
 * A detection DOES block, and that one is not politeness: `/detect` takes 2-4 s on a
 * miss, and its `finally` hands `activity` back to idle unconditionally. A
 * calibration started underneath it would have that `finally` fire in the middle of
 * its burst — un-pausing the forwarder into a half-rebuilt bank and re-enabling this
 * very control. `Activity` says only one thing may own the camera at a time; this is
 * the reading of it the operator sees.
 *
 * The hand-written page had a further branch — "set the four board corners first" —
 * which is now unrepresentable: a live camera carries a framing by construction, so
 * there is no state in which the corners are missing.
 */
export function calibrateBlockReason(input: {
  readonly camera: CameraState;
  readonly activity: Activity;
  /** The 2026-08-27 hard block: a portrait frame crops pieces along the wrong axis. */
  readonly portrait?: boolean;
}): string | null {
  if (input.activity.kind === 'calibrating') return 'calibrating…';
  if (input.activity.kind === 'detecting') return 'looking for the board…';
  // Calibration reads the live camera directly, not the stream — it needs the
  // camera running, which is a weaker and more honest requirement than needing
  // the socket up.
  if (input.camera.kind !== 'live') return 'the camera is not running';
  // Piece crops extend toward the top of the image, so a sideways board cannot
  // be read — blocked rather than warned, because the failed reads look like a
  // position problem and cost an afternoon (2026-08-27).
  if (input.portrait) return 'the frame is portrait — turn the phone sideways';
  return null;
}

// ---- automatic calibration ----------------------------------------------------

/**
 * "detect board -> calibrate should be fully automatic" (operator, 2026-08-27).
 * Detection already is; this is calibration's half: once positions are flowing
 * and the reading LOOKS like a start position sitting still, the page fires the
 * calibrate itself — full-res burst, sharpest frame, exactly the manual path
 * minus the tap and the confirm. The node's agreement gate stays the authority:
 * a false trigger is refused server-side and costs one quiet attempt.
 */
export interface AutoCalibrate {
  /** `performance.now()` before which no attempt is made (cooldown after one). */
  readonly nextAt: number;
  readonly attempts: number;
}

export const AUTO_CALIBRATE_START: AutoCalibrate = { nextAt: 0, attempts: 0 };
/** A refusal is not retried before this — the board state that caused it rarely
 * changes within seconds, and every attempt costs a full-res burst. */
export const AUTO_CALIBRATE_COOLDOWN_MS = 30_000;
/** Stricter than `MOVING_BELOW`: calibration wants a board at REST, not merely
 * "not obviously moving" — the burst picks the sharpest frame either way, but a
 * hand hovering near the board must not seed the bank. */
export const AUTO_CALIBRATE_MIN_STABILITY = 0.9;

/**
 * Occupancy says "two opposite full bands, middle empty" — the start position in
 * any of the four label rotations, robust to the piece-TYPE misreads an
 * uncalibrated reader makes (occupancy transfers across styles; types do not).
 */
export function looksLikeStartOccupancy(grid: readonly (readonly string[])[]): boolean {
  if (grid.length !== 8 || grid.some((row) => row.length !== 8)) return false;
  const rows = grid.map((row) => row.filter((c) => c !== 'empty').length);
  const cols = [0, 1, 2, 3, 4, 5, 6, 7].map(
    (c) => grid.filter((row) => row[c] !== 'empty').length
  );
  const banded = (counts: number[]): boolean =>
    counts[0] === 8 && counts[1] === 8 && counts[6] === 8 && counts[7] === 8 &&
    counts[2] === 0 && counts[3] === 0 && counts[4] === 0 && counts[5] === 0;
  return banded(rows) || banded(cols);
}

export interface AutoCalibrateGate {
  readonly auto: AutoCalibrate;
  readonly now: number;
  readonly camera: CameraState;
  readonly activity: Activity;
  readonly portrait: boolean;
  /** Positions only flow while streaming — and the reading is the trigger. */
  readonly streaming: boolean;
  readonly stability: number;
  readonly bursting: boolean;
  /** Automation covers the FIRST calibration only; extra angles stay deliberate. */
  readonly calibrations: number;
  /** The node's current reading for this board, or null before any arrived. */
  readonly grid: readonly (readonly string[])[] | null;
}

export function shouldAutoCalibrate(gate: AutoCalibrateGate): boolean {
  if (gate.calibrations > 0) return false;
  if (!gate.streaming || gate.portrait || gate.bursting) return false;
  if (gate.now < gate.auto.nextAt) return false;
  if (gate.camera.kind !== 'live' || !isFramed(gate.camera.framing)) return false;
  if (gate.activity.kind !== 'idle') return false;
  if (gate.stability < AUTO_CALIBRATE_MIN_STABILITY) return false;
  return gate.grid !== null && looksLikeStartOccupancy(gate.grid);
}

// ---- the hint line -----------------------------------------------------------

/** The one action a hint may offer. A tag, not a callback, so this stays pure. */
export type HintAction = 'look-again';

export interface Hint {
  /** Identity of the sentence, so a caller can tell a rewording from a new hint. */
  readonly key: string;
  readonly text: string;
  readonly action: HintAction | null;
}

export interface HintInput {
  readonly camera: CameraState;
  readonly activity: Activity;
  readonly auto: AutoDetect;
  readonly tracking: TrackingState;
  readonly streaming: boolean;
  readonly calibrateBlock: string | null;
}

/**
 * One sentence, chosen by priority: the single most useful thing to say right now.
 *
 * It exists partly because `title` is unreachable on a touch screen — a disabled
 * control's reason has to be readable somewhere the operator can actually see it.
 */
export function hintFor({
  camera,
  activity,
  auto,
  tracking,
  streaming,
  calibrateBlock
}: HintInput): Hint | null {
  // With no camera the whole stage is gone and the explainer says everything.
  if (camera.kind === 'blocked') return null;
  if (autoGaveUp(auto)) {
    return {
      key: 'giveup',
      text: 'No board found — drag the handles onto it, or',
      action: 'look-again'
    };
  }
  if (activity.kind === 'detecting') {
    return { key: 'detecting', text: 'Looking for the board…', action: null };
  }
  if (camera.kind !== 'live') {
    return { key: 'nocam', text: 'Starting the camera…', action: null };
  }
  if (!isFramed(camera.framing)) {
    return {
      key: 'finding',
      text: 'Looking for the board — hold the phone still and keep it in frame.',
      action: null
    };
  }
  if (tracking === 'lost') {
    return {
      key: 'lost',
      text: 'The board is lost — the quad is frozen where it last fitted.',
      action: null
    };
  }
  // Not while streaming: during a game the board is moving half the time, and the
  // sentence would flicker on every move for no gain.
  if (!streaming && calibrateBlock !== null) {
    return {
      key: `calib:${calibrateBlock}`,
      text: `Calibrate is unavailable — ${calibrateBlock}`,
      action: null
    };
  }
  return null;
}

// ---- lenses ------------------------------------------------------------------

/**
 * One rear camera, as the picker knows it.
 *
 * iOS >= 16.3 exposes EVERY rear lens through `enumerateDevices` — but only once a
 * `getUserMedia` call has populated the labels, and with no way at all to tell
 * programmatically which one is the ultra wide: `deviceId`s are regenerated per
 * session, labels are localised, and list order is not guaranteed across models.
 * So the operator chooses, and the choice is remembered by LABEL — the only
 * property of a camera that survives a reload. (Digital zoom is not an alternative:
 * iOS Safari does not implement the `zoom` track capability at all.)
 */
export interface Lens {
  readonly id: string;
  readonly label: string;
}

// ---- banners -----------------------------------------------------------------

export type BannerKind = 'ok' | 'warn' | 'err';

export interface BannerMessage {
  readonly kind: BannerKind;
  readonly text: string;
  /**
   * For the one message the operator must ACT on rather than watch go by: an
   * automatic re-detection may have landed the labels on a different physical
   * corner, and there is no way for the page to know. Cleared by "rotate labels",
   * which IS the acknowledgement.
   */
  readonly sticky: boolean;
}

// ---- chips -------------------------------------------------------------------

export type ChipTone = 'neutral' | 'ok' | 'warn' | 'bad';

export interface Chip {
  readonly text: string;
  readonly tone: ChipTone;
}

export function trackingChip(tracking: TrackingState): Chip {
  switch (tracking) {
    case 'off':
      return { text: 'tracking off', tone: 'neutral' };
    case 'seeded':
      return { text: 'tracking — locking on', tone: 'warn' };
    case 'locked':
      return { text: 'tracking — locked', tone: 'ok' };
    case 'lost':
      return { text: 'tracking — lost, quad frozen', tone: 'bad' };
    default:
      return { text: 'tracking idle', tone: 'neutral' };
  }
}

// ---- rates -------------------------------------------------------------------

/**
 * Send rate over the trailing window.
 *
 * `(n-1)/span`, not `n/window`: at the 1 fps keepalive the window holds four
 * stamps spanning 3 s, which is 1.0/s — `n/window` would claim 1.3/s.
 */
export function ratePerSecond(stamps: readonly number[], streaming: boolean): number {
  const n = stamps.length;
  const first = stamps[0];
  const last = stamps[n - 1];
  const span = n >= 2 && first !== undefined && last !== undefined ? (last - first) / 1000 : 0;
  if (span > 0) return (n - 1) / span;
  return streaming ? n / (RATE_WINDOW_MS / 1000) : 0;
}

/** Stamps still inside the rate window. */
export function withinRateWindow(stamps: readonly number[], now: number): number[] {
  return stamps.filter((t) => now - t <= RATE_WINDOW_MS);
}

// ---- the floating position panel ---------------------------------------------

export const PANEL_MIN = 140;
export const PANEL_MAX = 520;

export interface PanelState {
  /** `null` until the panel has been placed once — it parks itself top-left. */
  readonly left: number | null;
  readonly top: number | null;
  readonly width: number;
  readonly shown: boolean;
}

export interface PlacedPanel {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly shown: boolean;
}

/**
 * Hidden until asked for. The panel is a second, smaller picture of the board laid
 * over the first one, and the operator's complaint about it was simply that it
 * covered the thing they were aiming — so `position` opens it and nothing else does.
 * A phone that has already stored a panel state keeps whatever it chose: this is only
 * the value a first visit starts from.
 */
export const PANEL_START: PanelState = { left: null, top: null, width: 220, shown: false };

/**
 * Default park: the top-left corner. The top offset is not clearance for anything —
 * all of the page's chrome is along the bottom now — it is simply an inset that keeps
 * the panel from looking wedged into the corner, and leaves the a8 handle reachable
 * when the board fills the frame.
 */
const PANEL_PARK_LEFT = 12;
const PANEL_PARK_TOP = 46;

/**
 * Clamped back into the viewport every time, mirroring the snapshot page's loupe: a
 * panel parked in the corner of a landscape phone would otherwise be off-screen in
 * portrait, with no way to reach its own drag strip.
 *
 * It parks top-LEFT because that is the emptiest corner of the viewfinder: the
 * overlay buttons float down the right-hand edge in landscape, and everything the
 * page says — chips, banner, hint — sits along the bottom.
 *
 * One degree of freedom, on purpose: the WIDTH. The board is a square sized from
 * that single number, because chessground derives its container's height from its
 * element's box and a free two-axis drag renders non-square squares.
 */
export function placePanel(
  panel: PanelState,
  view: {
    readonly innerWidth: number;
    readonly innerHeight: number;
  },
  height: number
): PlacedPanel {
  const width = Math.round(Math.max(PANEL_MIN, Math.min(PANEL_MAX, panel.width)));
  const preferredLeft = panel.left ?? PANEL_PARK_LEFT;
  const preferredTop = panel.top ?? PANEL_PARK_TOP;
  return {
    width,
    shown: panel.shown,
    left: Math.max(4, Math.min(view.innerWidth - width - 4, preferredLeft)),
    top: Math.max(4, Math.min(view.innerHeight - (height || width) - 4, preferredTop))
  };
}

// ---- camera stall watchdog ----------------------------------------------------
// A phone's camera track dies quietly: iOS mutes it when the app is backgrounded
// or another app takes the camera, and sometimes never revives it on return. The
// preview then shows the last frame forever while the loop happily streams it.
// The watchdog reads `video.currentTime` every tick — a live capture advances it;
// a dead one does not — and escalates: first `play()` (fixes the paused-on-return
// case for free), then a full camera reopen, backed off so a genuinely absent
// camera is not hammered.

/** No clock advance for this long (while visible) → try `play()`. */
export const STALL_PLAY_MS = 2000;
/** Still no advance → reopen the camera. Also the floor of the reopen backoff. */
export const STALL_REOPEN_MS = 6000;
export const STALL_REOPEN_MAX_MS = 60000;

export interface FrameClock {
  /** Last observed `video.currentTime` (seconds — its own clock, not ours). */
  readonly time: number;
  /** When it last advanced, `performance.now()` ms. */
  readonly changedAt: number;
  /** Whether `play()` has already been tried for THIS stall. */
  readonly played: boolean;
  /** Consecutive reopens without the clock ever advancing in between. */
  readonly reopens: number;
  readonly nextReopenAt: number;
}

export type StallAction = 'none' | 'play' | 'reopen';

export function freshFrameClock(now: number): FrameClock {
  return { time: -1, changedAt: now, played: false, reopens: 0, nextReopenAt: 0 };
}

/**
 * Fold one tick's reading in. `hidden` resets the grace period rather than
 * counting: a backgrounded page throttles both the video clock and this very
 * loop, and recovery must not fire the instant the operator returns.
 */
export function noteFrameClock(
  clock: FrameClock,
  time: number,
  now: number,
  hidden: boolean
): { clock: FrameClock; action: StallAction } {
  if (hidden || time !== clock.time) {
    return {
      clock: { time, changedAt: now, played: false, reopens: 0, nextReopenAt: 0 },
      action: 'none'
    };
  }
  const stalledMs = now - clock.changedAt;
  if (stalledMs >= STALL_PLAY_MS && !clock.played) {
    return { clock: { ...clock, played: true }, action: 'play' };
  }
  if (stalledMs >= STALL_REOPEN_MS && now >= clock.nextReopenAt) {
    const backoff = Math.min(STALL_REOPEN_MS * 2 ** clock.reopens, STALL_REOPEN_MAX_MS);
    return {
      clock: { ...clock, reopens: clock.reopens + 1, nextReopenAt: now + backoff },
      action: 'reopen'
    };
  }
  return { clock, action: 'none' };
}

export function parseStoredPanel(raw: string | null | undefined): PanelState {
  if (raw === null || raw === undefined) return PANEL_START;
  let saved: unknown;
  try {
    saved = JSON.parse(raw);
  } catch {
    return PANEL_START;
  }
  if (!isRecord(saved)) return PANEL_START;
  const number = (value: unknown): number | null =>
    typeof value === 'number' && Number.isFinite(value) ? value : null;
  return {
    left: number(saved['left']),
    top: number(saved['top']),
    width: number(saved['width']) ?? PANEL_START.width,
    shown: typeof saved['shown'] === 'boolean' ? saved['shown'] : PANEL_START.shown
  };
}
