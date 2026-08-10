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
  readonly gain: number;
  /** Fast attack: a hand entering frame must drop stability at once. */
  readonly attack: number;
  /** Slower release, so rest is not declared between two strides of one move. */
  readonly release: number;
  readonly noiseFloor: number;
}

export const MOTION_DEFAULTS: MotionMeterOptions = {
  downscale: 32,
  gain: 80,
  attack: 0.7,
  release: 0.4,
  noiseFloor: 0.004
};

/**
 * Mean absolute inter-frame difference of a tiny grayscale thumbnail, through a
 * saturating gain curve and an asymmetric smoother.
 *
 * **Invariant: the FIRST frame reports 1.0.** There is nothing to difference
 * against, and an unknown board reports at rest — which matches D3's "stability is
 * a hint, never a gate". It also means a bare stability gate passes on frame one,
 * before autoexposure and autofocus have settled: anything that wants an
 * exemplar-grade frame (automatic detection, calibration) must add its own settle
 * delay rather than trusting this number alone.
 */
export class MotionMeter {
  readonly options: MotionMeterOptions;
  stability = 1;
  private prev: Float32Array | null = null;

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
    const { gain, noiseFloor, attack, release } = this.options;
    const instant = Math.exp(-gain * Math.max(diff - noiseFloor, 0));
    const weight = instant < this.stability ? attack : release;
    this.stability = Math.min(1, Math.max(0, (1 - weight) * this.stability + weight * instant));
    return this.stability;
  }

  reset(): void {
    this.prev = null;
    this.stability = 1;
  }
}

/**
 * RGBA thumbnail bytes -> gray in [0,1], with cv2's BGR2GRAY weights so a given
 * scene scores the same here as on the edge.
 *
 * A fresh array every call, deliberately: `MotionMeter.update` keeps what it is
 * given as the next frame's reference, and handing back one reused buffer is the
 * exact aliasing failure that once made the tracker report a confident lock on a
 * frozen quad. It is 1 kB per frame.
 */
export function sampleThumbnail(rgba: ArrayLike<number>, pixels: number): Float32Array {
  const out = new Float32Array(pixels);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (0.299 * rgba[p]! + 0.587 * rgba[p + 1]! + 0.114 * rgba[p + 2]!) / 255;
  }
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
