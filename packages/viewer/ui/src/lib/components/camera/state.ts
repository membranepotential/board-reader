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

/**
 * Was this framing established in THIS session — by the detector or by a finger —
 * as opposed to inherited from the last one?
 *
 * A `restored` quad is last session's crop over this session's world, and it can
 * be a full rank out (2026-08-27); that is precisely why `shouldAutoDetect`
 * verifies one. Anything that would BAKE IN a framing rather than check it must
 * therefore wait until it has been re-origined.
 */
export function isLiveFraming(framing: Framing): boolean {
  return framing.origin === 'operator' || framing.origin === 'detected';
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
  /**
   * The periodic drift check (`shouldRedetect`). Its own variant rather than a
   * flag on `detecting` because it is SILENT: it owns the camera and the quad
   * like any detection, but nobody asked for it, so the hint line must not
   * announce "looking for the board" every ten seconds at a board that is
   * already found and streaming.
   */
  | { readonly kind: 'verifying' }
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

// ---- periodic re-detection: re-locking a drifting frame ----------------------
//
// The tracker DRIFTS. It is a Lucas-Kanade plane tracker, so every frame's
// homography is measured against the last one and the small errors accumulate:
// on a board standing still, on a shaky holder, the quad walks about one square
// in two minutes (`docs/reports/2026-08-27-framing-drift.md`, take 5 — 82 px on
// the a1 corner, 58 % of that session's wrong squares). It is not a bump and no
// operator sees it happen; the picture still looks right, because the overlay is
// drawn where the tracker thinks the board is.
//
// So a resting, streaming page asks `/detect` every REDETECT_INTERVAL_MS what
// the board REALLY is, and re-anchors the tracker when the answer says it has
// walked. Every constant below is measured in that report (§4); each cites what
// from. The counter-intuitive part is the baseline: the detector's quad and the
// operator's quad do not mean the same line — see `Redetect.residual`.

/**
 * How often a resting page checks itself. Catching take 5's drift at seq 136
 * needs a detection every <= 12 frames ~ 6 s at 2.1 fps (every 24 frames only
 * catches it at seq 160, after the damage); 10 s leaves margin and costs ~80 ms
 * of node CPU per check.
 */
export const REDETECT_INTERVAL_MS = 10_000;

/**
 * The first N successful re-detections of a session are spent MEASURING, not
 * judging: they establish `Redetect.residual`.
 */
export const REDETECT_BASELINE_N = 3;

/**
 * Adopt above this baselined shift, in SQUARE WIDTHS. The four control takes
 * peak at 0.31/0.40/0.46/0.21 square of tracker drift and mis-address nothing;
 * take 5 passes 0.51 at seq 124 and 0.75 by seq 160, where it is reading six
 * squares off their true grid position.
 */
export const ADOPT_SHIFT_SQUARES = 0.45;

/**
 * ...on this many CONSECUTIVE successful re-detections. At K = 1 the threshold
 * fires once on take 3; at K = 2 the sweep adopts zero times across all four
 * controls at every cadence from 6 to 24 frames, and still catches take 5 at
 * seq 136 — before the worst stretch, and before the operator noticed.
 */
export const ADOPT_CONSECUTIVE = 2;

/**
 * Above this it is not drift. Two squares of disagreement is a REPOSITION
 * (DESIGN §4.4.1(b)) — a moved phone, a moved board, or a detector that locked
 * onto something else entirely — and the page must hand that to the operator
 * rather than quietly re-crop around it.
 */
export const ADOPT_MAX_SHIFT_SQUARES = 2.0;

/**
 * `/detect` returns an UNLABELLED quad (DESIGN decision #15), so its corners
 * must be matched onto the tracked ones before anything is compared. Require the
 * runner-up correspondence to be this many times worse than the winner. The
 * measured minimum over 230 detections is 9.7x (best <= 59 px, second best
 * >= 533 px), so this never blocks a real correction — it refuses an ambiguous
 * one, which is the only way this mechanism could rotate the labels off a8.
 */
export const ADOPT_RELABEL_MARGIN = 4.0;

/**
 * Consecutive-miss back-off, then silence. `/detect` answers in ~0.1 s on a hit
 * but 2-4 s on a MISS, and a page pointed at a board the detector cannot see
 * would otherwise post a full-frame JPEG every 10 s for the whole session, on a
 * phone battery, forever. Same shape as `AUTO_BACKOFF_MS` for the same reason —
 * and equally silent, because nobody asked for this check and the framing it
 * could not confirm is still the best one anybody has.
 */
export const REDETECT_BACKOFF_MS: readonly number[] = [20_000, 40_000, 80_000];

/**
 * A per-corner displacement in frame pixels: the same four slots as a quad, but
 * a difference rather than a place.
 */
export type CornerOffsets = readonly [Point, Point, Point, Point];

export interface Redetect {
  /** `performance.now()` before which no check is made. */
  readonly nextAt: number;
  /** Residual samples so far. Emptied once `residual` is fixed. */
  readonly samples: readonly CornerOffsets[];
  /**
   * R0 — the per-corner offset between where the operator's quad puts a corner
   * and where the detector puts it, or `null` while it is still being measured.
   *
   * These are two different conventions for the same board and they differ by a
   * CONSTANT: a finger-dragged corner sits on the board's edge, the detector's
   * on the playing surface, 17-33 px apart on the hand-framed takes (4-6 px when
   * the framing itself came from `/detect`). That offset is not drift and never
   * shrinks, so an ABSOLUTE threshold on detector-vs-tracked distance
   * false-alarms on every hand-framed session — the report's §2 finding, and the
   * reason this whole mechanism is baselined per session rather than compared
   * against zero.
   */
  readonly residual: CornerOffsets | null;
  /** Consecutive readings over `ADOPT_SHIFT_SQUARES`. */
  readonly over: number;
  /** Consecutive readings past `ADOPT_MAX_SHIFT_SQUARES` — a reposition, not drift. */
  readonly suspects: number;
  /**
   * Whether the operator has already been told about the current reposition
   * episode. Cleared only by a reading that is genuinely back in place, so a quad
   * oscillating across the cap says it once rather than every twenty seconds.
   */
  readonly warned: boolean;
  /** Consecutive detector misses. Any answer at all zeroes it; four is giving up. */
  readonly fails: number;
}

export const REDETECT_START: Redetect = {
  nextAt: 0,
  samples: [],
  residual: null,
  over: 0,
  suspects: 0,
  warned: false,
  fails: 0
};

export function redetectGaveUp(redetect: Redetect): boolean {
  return redetect.fails >= REDETECT_BACKOFF_MS.length;
}

/** The gap after this many CONSECUTIVE misses: 20 s, then 40, then 80, then silence. */
export function redetectBackoffFor(fails: number): number {
  const i = Math.min(Math.max(fails - 1, 0), REDETECT_BACKOFF_MS.length - 1);
  return REDETECT_BACKOFF_MS[i] as number;
}

export interface RedetectGate {
  readonly redetect: Redetect;
  readonly now: number;
  readonly camera: CameraState;
  readonly activity: Activity;
  /** The check exists to protect what is on the wire, so it runs while streaming. */
  readonly streaming: boolean;
  readonly stability: number;
  /** Mid-burst the board is in motion by definition — a detection there is wasted. */
  readonly bursting: boolean;
  readonly portrait: boolean;
  readonly hidden: boolean;
  /** Tracking switched off freezes the quad deliberately; nothing may re-seed it. */
  readonly tracking: boolean;
}

/**
 * A RESTING frame, and nothing else in flight.
 *
 * The stability floor is `AUTO_STABILITY` for the same reason the first
 * detection uses it: this frame is about to be measured to a fraction of a
 * square, so it must be exemplar-grade and not merely "the camera is not
 * moving". Take 5 reads >= 0.9 on 69 % of its frames, so resting frames are not
 * scarce.
 *
 * There is deliberately no gate on the tracker being `locked`: a tracker that
 * has LOST the board is already claimed, sooner, by `shouldAutoDetect`'s
 * sustained-loss path (`AUTO_LOST_MS` = 6 s, inside this 10 s interval), and a
 * `locked` requirement here would silently disable the whole mechanism on a
 * device whose tracker never reports it.
 */
export function shouldRedetect({
  redetect,
  now,
  camera,
  activity,
  streaming,
  stability,
  bursting,
  portrait,
  hidden,
  tracking
}: RedetectGate): boolean {
  if (!streaming || portrait || bursting || hidden) return false;
  // With tracking off the quad is frozen on purpose — the operator is holding a
  // crop still — so there is no drift to correct and nothing may re-seed it.
  if (!tracking) return false;
  // One thing owns the camera and the quad at a time: not a drag, not a
  // calibration, not another detection, and not a second check.
  if (activity.kind !== 'idle') return false;
  if (camera.kind !== 'live') return false;
  // Only a framing THIS session established. A `restored` one may be a rank out
  // (which is what `shouldAutoDetect` verifies it for), and R0 is permanent: were
  // a stale quad allowed to baseline — as it would be if that verification had
  // burned its own budget against a slow node — the rank-off framing becomes this
  // session's definition of correct, and real drift then reads as zero for ever.
  // The check starts the moment auto-detect or the operator re-origins it.
  if (!isLiveFraming(camera.framing)) return false;
  if (redetectGaveUp(redetect)) return false;
  if (now < redetect.nextAt) return false;
  return stability >= AUTO_STABILITY;
}

/**
 * One square's width in the quad's own pixels: the mean side over 8.
 *
 * The report's unit throughout, and deliberately NOT `quadShift`'s fraction of
 * the frame diagonal — a crop lands on the wrong square when the grid slips by a
 * square, not by a diagonal. (`QUAD_VERIFY_SHIFT` = 0.03 of the diagonal works
 * out at 0.47 square on that data, right in the middle of the control
 * distribution, so it is far too tight to reuse as an adoption threshold. It
 * keeps its own job above, which is telling a verify that moved nothing from one
 * that moved something.)
 */
export function squareWidth(quad: CornerQuad): number {
  let perimeter = 0;
  for (let i = 0; i < 4; i++) {
    const p = quad[i] as Point;
    const q = quad[(i + 1) % 4] as Point;
    perimeter += Math.hypot(q[0] - p[0], q[1] - p[1]);
  }
  return perimeter / 32; // mean side (perimeter / 4), then eight squares to a side
}

/** Corner-for-corner `a - b`. Both quads must already be in the same slot order. */
export function cornerOffsets(a: CornerQuad, b: CornerQuad): CornerOffsets {
  const at = (i: 0 | 1 | 2 | 3): Point => [a[i][0] - b[i][0], a[i][1] - b[i][1]];
  return [at(0), at(1), at(2), at(3)];
}

/**
 * The per-corner, per-axis MEDIAN of the samples.
 *
 * The report says mean; the median is used because the baseline is three samples
 * and it is permanent — one detection that landed a square off (the detector
 * misses cluster on exactly the frames this runs on) would otherwise poison
 * every later reading for the rest of the session. On the report's data the two
 * agree: repeatability there is 0.00 px median, 0.72 px worst.
 */
export function medianOffsets(samples: readonly CornerOffsets[]): CornerOffsets | null {
  if (samples.length === 0) return null;
  const median = (values: readonly number[]): number => {
    const sorted = [...values].sort((x, y) => x - y);
    const mid = sorted.length >> 1;
    const upper = sorted[mid] as number;
    return sorted.length % 2 === 1 ? upper : (upper + (sorted[mid - 1] as number)) / 2;
  };
  const at = (i: 0 | 1 | 2 | 3): Point => [
    median(samples.map((sample) => sample[i][0])),
    median(samples.map((sample) => sample[i][1]))
  ];
  return [at(0), at(1), at(2), at(3)];
}

/**
 * How far the tracked quad has walked, in square widths, with the convention
 * offset taken out: `max over corners of |(detected - tracked) - R0|`.
 *
 * Max rather than mean because drift is not uniform — take 5's a1 reached 82 px
 * while its h1 moved 27 — and it is the WORST corner that decides whether a
 * square's crop lands on its neighbour.
 */
export function baselinedShift(
  detected: CornerQuad,
  tracked: CornerQuad,
  residual: CornerOffsets
): number {
  // Normalised by the TRACKED grid, which is the one actually cutting squares out
  // of the frame — the question is how far off ITS own squares the crop has
  // wandered. (The report normalises by the detector's grid; the two differ by the
  // drift itself, a few per cent. Using the tracked quad also keeps a spurious
  // undersized detection from dividing a moderate disagreement into a huge one.)
  const unit = squareWidth(tracked);
  if (!(unit > 0)) return 0;
  const offsets = cornerOffsets(detected, tracked);
  let worst = 0;
  for (let i = 0; i < 4; i++) {
    const o = offsets[i] as Point;
    const r = residual[i] as Point;
    worst = Math.max(worst, Math.hypot(o[0] - r[0], o[1] - r[1]));
  }
  return worst / unit;
}

/** The detector's quad expressed in the operator's convention: `detected - R0`. */
export function correctedQuad(detected: CornerQuad, residual: CornerOffsets): CornerQuad {
  const at = (i: 0 | 1 | 2 | 3): Point => [
    detected[i][0] - residual[i][0],
    detected[i][1] - residual[i][1]
  ];
  return [at(0), at(1), at(2), at(3)];
}

export interface QuadMatch {
  /** `detected`, re-ordered onto `tracked`'s four slots. */
  readonly aligned: CornerQuad;
  /** The winning order's worst-corner distance, in pixels. */
  readonly cost: number;
  /** How many times worse the runner-up order is; `Infinity` for an exact hit. */
  readonly margin: number;
}

/**
 * The eight orders of four corners that keep them a ring: four rotations, each
 * either winding. Anything else crosses the quad over itself and is not a
 * relabelling of the same board.
 */
type CornerOrder = readonly [CornerIndex, CornerIndex, CornerIndex, CornerIndex];

const CORNER_ORDERS: readonly CornerOrder[] = [
  [0, 1, 2, 3],
  [1, 2, 3, 0],
  [2, 3, 0, 1],
  [3, 0, 1, 2],
  [0, 3, 2, 1],
  [3, 2, 1, 0],
  [2, 1, 0, 3],
  [1, 0, 3, 2]
];

/**
 * `detected` matched onto `tracked` by nearest corner, with the margin over the
 * runner-up.
 *
 * A detection carries no labels (decision #15): its storage order is the
 * detector's, and slot 0 is not necessarily the operator's a8. Comparing the two
 * quads slot-for-slot without this would read a mere re-ordering as a two-board
 * shift. Every non-winning order moves at least one corner by most of a board's
 * side, which is what makes `margin` a decisive check rather than a formality.
 */
export function matchCorners(detected: CornerQuad, tracked: CornerQuad): QuadMatch {
  let bestOrder = CORNER_ORDERS[0] as CornerOrder;
  let best = Infinity;
  let second = Infinity;
  for (const order of CORNER_ORDERS) {
    let cost = 0;
    for (let i = 0; i < 4; i++) {
      const p = detected[order[i] as CornerIndex];
      const q = tracked[i] as Point;
      cost = Math.max(cost, Math.hypot(p[0] - q[0], p[1] - q[1]));
    }
    if (cost < best) {
      second = best;
      best = cost;
      bestOrder = order;
    } else if (cost < second) {
      second = cost;
    }
  }
  return {
    aligned: [
      detected[bestOrder[0]],
      detected[bestOrder[1]],
      detected[bestOrder[2]],
      detected[bestOrder[3]]
    ],
    cost: best,
    margin: best === 0 ? Infinity : second / best
  };
}

/**
 * What one successful re-detection means. Exactly one of these, so the page
 * cannot act on two readings of the same number.
 */
export type RedetectOutcome =
  /** Still measuring the convention offset; `samples` of `REDETECT_BASELINE_N` taken. */
  | { readonly kind: 'baselining'; readonly samples: number }
  /**
   * The detected corners cannot be matched onto the tracked ones with confidence,
   * so nothing is concluded — no baseline sample, and no adoption however far the
   * quads are apart. A big enough translation does this by itself: two boards'
   * worth of displacement makes every corner pairing nearly as good as every
   * other, which is exactly when adopting one would be a relabelling.
   */
  | { readonly kind: 'ambiguous'; readonly margin: number }
  /** Inside the threshold: the tracker is where it says it is. */
  | { readonly kind: 'steady'; readonly shift: number }
  /** Over the threshold, but not yet on `ADOPT_CONSECUTIVE` readings. */
  | { readonly kind: 'arming'; readonly shift: number }
  /** Walked. `quad` is the correction, in the operator's own corner convention. */
  | { readonly kind: 'adopt'; readonly shift: number; readonly quad: CornerQuad }
  /**
   * Too far to be drift — a reposition. `repeats` counts the episode; `warn` is
   * true only on the reading that should actually tell the operator, which is not
   * simply the first of a run: a quad oscillating across the cap produces
   * suspect / arming / suspect, and only a reading genuinely back in place ends
   * the episode.
   */
  | {
      readonly kind: 'suspect';
      readonly shift: number;
      readonly repeats: number;
      readonly warn: boolean;
    };

export interface RedetectResult {
  readonly redetect: Redetect;
  readonly outcome: RedetectOutcome;
}

/**
 * A detector MISS is neutral: it neither contributes to the baseline nor moves
 * the adoption counter, it only schedules the next check.
 *
 * Deliberately not a reset. The misses cluster on the drifted frames themselves
 * (10 % on take 5 against 4 % overall), so a miss that disarmed the counter
 * would be most likely to do so exactly when the mechanism was about to fire.
 */
export function noteRedetectMiss(redetect: Redetect, now: number): Redetect {
  const fails = redetect.fails + 1;
  return { ...redetect, fails, nextAt: now + redetectBackoffFor(fails) };
}

/**
 * Fold one successful re-detection in: baseline, then judge.
 *
 * `tracked` is the quad as it stood when the detected FRAME was taken, not when
 * the reply came back — the tracker keeps running through the request, and
 * pairing the detector's answer with a quad from a few hundred ms later would
 * measure the wait as well as the drift.
 */
export function noteRedetection(input: {
  readonly redetect: Redetect;
  readonly now: number;
  readonly detected: CornerQuad;
  readonly tracked: CornerQuad;
}): RedetectResult {
  const { redetect, now, detected, tracked } = input;
  // The detector answered, so the failure budget is whole again — whatever the
  // answer turns out to mean.
  const next: Redetect = { ...redetect, nextAt: now + REDETECT_INTERVAL_MS, fails: 0 };

  const match = matchCorners(detected, tracked);
  const matched = match.margin >= ADOPT_RELABEL_MARGIN;
  const residual = redetect.residual;

  if (residual === null) {
    // A correspondence nobody can trust must not become the session's convention.
    if (!matched) return { redetect: next, outcome: { kind: 'ambiguous', margin: match.margin } };
    const samples = [...redetect.samples, cornerOffsets(match.aligned, tracked)];
    const done = samples.length >= REDETECT_BASELINE_N;
    return {
      redetect: {
        ...next,
        samples: done ? [] : samples,
        residual: done ? medianOffsets(samples) : null
      },
      outcome: { kind: 'baselining', samples: samples.length }
    };
  }

  const shift = baselinedShift(match.aligned, tracked, residual);

  // The reposition cap is judged FIRST, and deliberately without the margin: the
  // two are not independent. A quad displaced by most of a board makes every
  // corner pairing nearly as good as every other, so a genuine reposition scores
  // a poor margin BY CONSTRUCTION — check the margin first and every one of them
  // comes back 'ambiguous' and the operator is never told (review, 0ebbbd2).
  if (shift > ADOPT_MAX_SHIFT_SQUARES) {
    const repeats = redetect.suspects + 1;
    return {
      redetect: { ...next, over: 0, suspects: repeats, warned: true },
      outcome: { kind: 'suspect', shift, repeats, warn: !redetect.warned }
    };
  }

  // Below the cap, corners that do not pair off decisively conclude NOTHING: a
  // bad match is the one way this mechanism could put a8 on another corner. The
  // counters are left exactly as they were, so such a reading can delay an
  // adoption but never cause or cancel one.
  if (!matched) return { redetect: next, outcome: { kind: 'ambiguous', margin: match.margin } };

  if (shift <= ADOPT_SHIFT_SQUARES) {
    // Back in place: this is the only thing that ends a reposition episode.
    return {
      redetect: { ...next, over: 0, suspects: 0, warned: false },
      outcome: { kind: 'steady', shift }
    };
  }
  const over = redetect.over + 1;
  if (over < ADOPT_CONSECUTIVE) {
    return { redetect: { ...next, over }, outcome: { kind: 'arming', shift } };
  }
  // Adopted: the counter starts again, but R0 is KEPT — the correction is
  // expressed in the operator's convention, so the convention has not changed.
  return {
    redetect: { ...next, over: 0, suspects: 0, warned: false },
    outcome: { kind: 'adopt', shift, quad: correctedQuad(match.aligned, residual) }
  };
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
  /** The 2026-08-27 hard block: a portrait frame crops pieces along the wrong axis. */
  readonly portrait?: boolean;
}): string | null {
  if (input.activity.kind === 'calibrating') return 'calibrating…';
  if (input.activity.kind === 'detecting') return 'looking for the board…';
  // Same reasoning: a drift check ends with an unconditional hand-back to idle,
  // which would land in the middle of a calibration begun underneath it.
  if (input.activity.kind === 'verifying') return 'checking the framing…';
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
