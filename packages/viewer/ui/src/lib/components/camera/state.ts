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
  return !isFramed(camera.framing) || sustainedLoss;
}

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
}): string | null {
  if (input.activity.kind === 'calibrating') return 'calibrating…';
  if (input.activity.kind === 'detecting') return 'looking for the board…';
  // Calibration reads the live camera directly, not the stream — it needs the
  // camera running, which is a weaker and more honest requirement than needing
  // the socket up.
  if (input.camera.kind !== 'live') return 'the camera is not running';
  return null;
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

/** Default park: the top-left corner, clear of the chips above it. */
const PANEL_PARK_LEFT = 12;
const PANEL_PARK_TOP = 46;

/**
 * Clamped back into the viewport every time, mirroring the snapshot page's loupe: a
 * panel parked in the corner of a landscape phone would otherwise be off-screen in
 * portrait, with no way to reach its own drag strip.
 *
 * It parks top-LEFT because the overlay buttons float down the right-hand edge in
 * landscape; the chips own the very top, so the default top clears them.
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
