/**
 * The browser edge's motion meter and adaptive forwarder — the JS mirror of
 * `board_reader_edge.motion` and `board_reader_edge.forwarder` (DESIGN D3/D4).
 *
 * The camera page IS an edge node: it samples faster than it ever forwards, meters
 * motion on every sampled frame, and sends at a keepalive rate that jumps to a
 * burst the moment the board starts moving. The Python edge is the reference for
 * the semantics; the constants below are mirrored value for value, so a scene
 * scores the same on a phone as it does on a Pi.
 *
 * Split into a pure meter over a grayscale thumbnail and a DOM-side sampler that
 * produces one, because that keeps the whole decision path testable in node — and
 * because the aliasing bug that once broke the tracker is the same shape as the one
 * that would break this: see `sampleThumbnail`.
 */

import { toGray } from './sharpness';

/** Sampled far faster than anything is forwarded — the meter must see every frame. */
export const CAPTURE_FPS = 20;

/** `ForwarderConfig`, mirrored: a keepalive so the reader keeps seeing the board… */
export const KEEPALIVE_FPS = 1.0;
/** …and a burst so a whole move is streamed. */
export const BURST_FPS = 12.0;
/** Stability under this starts a burst. */
export const ENTER_BURST = 0.7;
/** Stability must stay above this, for `EXIT_HOLD_S`, to end one — hysteresis. */
export const EXIT_BURST = 0.85;
export const EXIT_HOLD_S = 0.7;
/** Float slack so a rate due "exactly now" is not missed by a rounding bit. */
const DUE_EPS = 1e-6;

export interface MotionMeterOptions {
  /** Thumbnail side. 32×32 is enough to see a hand and cheap enough for 20 fps. */
  readonly downscale: number;
  /** Steepness of the (diff / floor) → stability curve, in RATIO units. */
  readonly gain: number;
  /** Fast attack: a hand entering frame must drop stability at once. */
  readonly attack: number;
  /** Slower release, so rest is not declared between two strides of one move. */
  readonly release: number;
  /**
   * Ratios up to this score a full 1.0 — the resting scatter band. Motion has to
   * exceed `tolerance ×` the learned floor before stability drops at all.
   */
  readonly tolerance: number;
  /** Floor chase rate when a frame differences ABOVE it (slow — rides out moves). */
  readonly floorUp: number;
  /** Floor chase rate when a frame differences BELOW it (fast — finds real rest). */
  readonly floorDown: number;
  /** The floor never learns below this — an absolute sensor-noise lower bound. */
  readonly noiseFloor: number;
}

/**
 * Tuned on the three 2026-08-27 phone takes (replayed from their recordings): the
 * absolute scale of the old meter did NOT transfer to the deployment phone — rest
 * measured 6× the assumed noise floor, so stability lived under 0.35 all day and
 * everything thresholded on it (auto-detect, the burst forwarder) was dead. The
 * meter now learns each rig's own resting difference (an asymmetric-chase floor,
 * settling in the low quantiles of the resting mass) and scores the RATIO to it,
 * which put ≥0.9 on 64–70 % of frames (the hold fraction) and <0.7 on 19–26 %
 * (the move fraction) across all three takes with these values.
 */
/** Ratio under which a frame counts as resting scatter and teaches the floor at
 * full rate; above it, at a fifth (see the update loop). */
export const FLOOR_NEAR = 4.0;
export const FLOOR_FAR_DIVISOR = 5.0;

export const MOTION_DEFAULTS: MotionMeterOptions = {
  downscale: 32,
  gain: 1.5,
  attack: 0.7,
  release: 0.4,
  tolerance: 2.0,
  floorUp: 0.02,
  floorDown: 0.08,
  noiseFloor: 0.0015
};

/**
 * Mean absolute inter-frame difference of a tiny grayscale thumbnail, through a
 * saturating gain curve and an asymmetric smoother.
 *
 * **Invariant: the FIRST frame reports 1.0.** There is nothing to difference
 * against, and an unknown board reports at rest — which matches D3's "stability is
 * a hint, never a gate". It also means a bare stability gate passes on frame one,
 * before autoexposure and autofocus have settled: anything that wants an
 * clean frame (automatic detection) must add its own settle
 * delay rather than trusting this number alone.
 */
export class MotionMeter {
  readonly options: MotionMeterOptions;
  stability = 1;
  private prev: Float32Array | null = null;
  /** The learned resting mean-difference of THIS rig's sampling path. */
  private floor: number | null = null;

  constructor(options: Partial<MotionMeterOptions> = {}) {
    this.options = { ...MOTION_DEFAULTS, ...options };
  }

  /**
   * `thumbnail` is `downscale²` gray values in [0,1]. It is RETAINED as the next
   * frame's reference, so the caller must not hand the same buffer back refilled —
   * every frame would then be differenced against itself and read as perfectly
   * still. `sampleThumbnail` allocates per call for exactly this reason.
   */
  update(thumbnail: Float32Array): number {
    const prev = this.prev;
    this.prev = thumbnail;
    if (prev === null || prev.length !== thumbnail.length) {
      this.stability = 1;
      return 1;
    }
    let sum = 0;
    for (let i = 0; i < thumbnail.length; i++) sum += Math.abs(thumbnail[i]! - prev[i]!);
    const diff = sum / thumbnail.length;
    const { gain, tolerance, floorUp, floorDown, noiseFloor, attack, release } = this.options;
    // Chase the resting difference asymmetrically: drop fast toward genuine rest,
    // rise slowly through a move — the floor settles in the resting mass's low
    // quantiles whatever this rig's sensor, holder and sampling path add up to.
    // Seeded at the absolute bound, NOT the first observed diff: a camera that
    // opens onto motion must not learn motion as its floor. The cost is a short
    // conservative learn-up (a few seconds reading "moving" on a noisy rig) that
    // the settle delays above this meter absorb anyway. The upward chase is
    // additionally GATED: a frame far above the floor (a clear move, ratio past
    // FLOOR_NEAR) teaches it at a fifth of the rate, so multi-second continuous
    // motion — a piece dragged, the camera carried — is not adopted as rest.
    if (this.floor === null) this.floor = noiseFloor;
    const near = diff / this.floor <= FLOOR_NEAR;
    const step = diff < this.floor ? floorDown : near ? floorUp : floorUp / FLOOR_FAR_DIVISOR;
    this.floor = Math.max(noiseFloor, this.floor + step * (diff - this.floor));
    const instant = Math.exp(-gain * Math.max(diff / this.floor - tolerance, 0));
    const weight = instant < this.stability ? attack : release;
    this.stability = Math.min(1, Math.max(0, (1 - weight) * this.stability + weight * instant));
    return this.stability;
  }

  reset(): void {
    this.prev = null;
    this.stability = 1;
    this.floor = null;
  }

  /**
   * The next thumbnail starts a new reference (and scores 1.0, as any first frame),
   * keeping the learned floor. For a change of WHAT is sampled — a new crop, a reopened
   * camera — rather than of the scene: differencing across it would read the
   * switch as one frame of whole-field motion and open a burst for nothing.
   */
  rebase(): void {
    this.prev = null;
  }
}

/**
 * Each thumbnail pixel is the mean of a `MOTION_OVERSAMPLE²` block of the image the
 * page draws, so the browser thumbnail is an AREA average like the edge's
 * `cv2.INTER_AREA`. Drawing a ~950 px crop straight to 32×32 is not one: a canvas
 * downscale samples a few source pixels per output pixel, keeps the sensor noise,
 * and on the 2026-11-24 take that put the resting difference at ~0.02 — about 8×
 * an area average. The learned floor sat there, a hand never reached the
 * tolerance above it, and no move opened a burst. At 8× (a 256² readback) the
 * resting difference measured within 20 % of `INTER_AREA` on that take's frames.
 */
export const MOTION_OVERSAMPLE = 8;

/** A canvas readback (an `ImageData`): RGBA bytes, row-major, `width × height`. */
export interface RgbaImage {
  readonly data: ArrayLike<number>;
  readonly width: number;
  readonly height: number;
}

/**
 * A canvas readback -> a `side × side` gray thumbnail in [0,1]: luma by
 * `toGray`'s cv2 BGR2GRAY weights, then the mean of each block, so a scene scores
 * the same here as on the edge. Each side of `image` must be a whole multiple of
 * `side` (the page draws at `side × MOTION_OVERSAMPLE`).
 *
 * A fresh array every call, deliberately: `MotionMeter.update` keeps what it is
 * given as the next frame's reference, and handing back one reused buffer is the
 * exact aliasing failure that once made the tracker report a confident lock on a
 * frozen quad. It is 4 kB per frame.
 */
export function sampleThumbnail(image: RgbaImage, side: number): Float32Array {
  const { width, height } = image;
  if (side < 1 || width % side !== 0 || height % side !== 0) {
    throw new RangeError(`a ${width}×${height} image does not divide into ${side}×${side} blocks`);
  }
  const gray = toGray(image.data, new Float32Array(width * height));
  const bw = width / side;
  const bh = height / side;
  const out = new Float32Array(side * side);
  for (let y = 0; y < height; y++) {
    const row = Math.floor(y / bh) * side;
    for (let x = 0; x < width; x++) out[row + Math.floor(x / bw)]! += gray[y * width + x]!;
  }
  const area = bw * bh;
  for (let i = 0; i < out.length; i++) out[i]! /= area;
  return out;
}

/** What the forwarder decided to attach to a frame it is sending. */
export interface ForwardDecision {
  readonly stability: number;
  /** Non-null for every frame of a burst — the consumer's handle on one transition. */
  readonly burstId: string | null;
}

/**
 * Fed EVERY sampled frame — dropping frames upstream would make stability lie.
 * Every rate decision is made against CAPTURE time, never a separate clock
 * (DESIGN D5), so a stream replayed from a file paces itself the same way.
 */
export class AdaptiveForwarder {
  private burstId: string | null = null;
  /** Never reset: a burst id must not recur within a session. */
  private burstCount = 0;
  private calmSince: number | null = null;
  private lastForwardTs: number | null = null;

  constructor(readonly meter: MotionMeter) {}

  get bursting(): boolean {
    return this.burstId !== null;
  }

  /** Visible for the status line; the id itself is the forwarder's to mint. */
  get currentBurstId(): string | null {
    return this.burstId;
  }

  private trackBurst(stability: number, captureTs: number, boardId: string): void {
    if (this.burstId === null) {
      // Eager entry: the transition starts NOW, and every frame of the burst carries
      // this id. Named from its own counter, not from seq — seq advances only on
      // frames actually sent, so a fully-dropped burst would reuse its id and the
      // consumer would see two transitions fused under one.
      if (stability < ENTER_BURST) {
        this.burstId = `${boardId}:b${this.burstCount++}`;
        this.calmSince = null;
      }
      return;
    }
    // Patient exit: sustained calm, or the pause between two strides of one move
    // would split it into two bursts.
    if (stability <= EXIT_BURST) this.calmSince = null;
    else if (this.calmSince === null) this.calmSince = captureTs;
    else if (captureTs - this.calmSince >= EXIT_HOLD_S - DUE_EPS) {
      this.burstId = null;
      this.calmSince = null;
    }
  }

  /** Is another frame due at the rate that applies right now? */
  due(captureTs: number): boolean {
    if (this.lastForwardTs === null) return true;
    const elapsed = captureTs - this.lastForwardTs;
    if (elapsed < 0) return true; // clock stepped backwards: re-anchor, never stall
    // Read at decision time, so entering a burst raises the rate immediately instead
    // of waiting out the keepalive interval already pending.
    const fps = this.bursting ? BURST_FPS : KEEPALIVE_FPS;
    return elapsed >= 1 / fps - DUE_EPS;
  }

  /**
   * Meter this frame and say whether to send it. `null` means "metered, not due" —
   * which is the common answer at the keepalive rate and is not a drop.
   */
  decide(
    thumbnail: Float32Array,
    context: { readonly boardId: string; readonly captureTs: number }
  ): ForwardDecision | null {
    const stability = this.meter.update(thumbnail);
    this.trackBurst(stability, context.captureTs, context.boardId);
    if (!this.due(context.captureTs)) return null;
    this.lastForwardTs = context.captureTs;
    return { stability, burstId: this.burstId };
  }

  reset(): void {
    this.meter.reset();
    this.burstId = null;
    this.calmSince = null;
    this.lastForwardTs = null;
  }
}
