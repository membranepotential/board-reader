import { describe, expect, it } from 'vitest';
import {
  AdaptiveForwarder,
  BURST_FPS,
  ENTER_BURST,
  EXIT_BURST,
  EXIT_HOLD_S,
  KEEPALIVE_FPS,
  MOTION_DEFAULTS,
  MOTION_OVERSAMPLE,
  MotionMeter,
  sampleThumbnail,
  type RgbaImage
} from './motion';

const PIXELS = 32 * 32;

/** A uniform thumbnail — the still board. */
function flat(value: number): Float32Array {
  return new Float32Array(PIXELS).fill(value);
}

/** A thumbnail that differs from `flat(base)` by `delta` on a fraction of its pixels. */
function disturbed(base: number, delta: number, fraction: number): Float32Array {
  const out = new Float32Array(PIXELS).fill(base);
  for (let i = 0; i < Math.round(PIXELS * fraction); i++) out[i] = base + delta;
  return out;
}

describe('MotionMeter', () => {
  it('reports 1.0 on its FIRST frame — there is nothing to difference against', () => {
    // Load-bearing: a bare stability gate therefore passes on frame one, before
    // autoexposure and autofocus have settled, which is why anything wanting an
    // exemplar-grade frame must add its own settle delay.
    expect(new MotionMeter().update(flat(0.4))).toBe(1);
  });

  it('stays at 1.0 while nothing moves', () => {
    const meter = new MotionMeter();
    meter.update(flat(0.5));
    expect(meter.update(flat(0.5))).toBeCloseTo(1, 6);
  });

  it('drops the moment the scene changes', () => {
    const meter = new MotionMeter();
    meter.update(flat(0.5));
    // The gain curve saturates on a whole-field change, so one attack step takes
    // stability to (1 - attack) exactly.
    expect(meter.update(disturbed(0.5, 0.5, 1))).toBeCloseTo(1 - MOTION_DEFAULTS.attack, 6);
  });

  it('ignores sensor noise under the floor', () => {
    const meter = new MotionMeter();
    meter.update(flat(0.5));
    expect(meter.update(flat(0.5 + 0.003))).toBeCloseTo(1, 6);
  });

  it('attacks fast and releases slowly — a move must not read as rest between strides', () => {
    const meter = new MotionMeter();
    meter.update(flat(0.5));
    const moving = disturbed(0.5, 0.5, 1);
    meter.update(moving);
    const afterTheChange = meter.stability;
    const drop = 1 - afterTheChange;
    // The scene now HOLDS still (the same pixels twice), which is what recovery is.
    meter.update(moving);
    const rise = meter.stability - afterTheChange;
    expect(meter.stability).toBeGreaterThan(afterTheChange);
    expect(meter.stability).toBeLessThan(1);
    // One frame of movement costs more than one frame of stillness earns back, so
    // the pause between two strides of a move cannot be mistaken for rest.
    expect(drop).toBeGreaterThan(rise);
  });

  it("learns a noisy rig's resting difference and reads its rest as rest", () => {
    // The 2026-08-27 finding: on the deployment phone the resting inter-frame
    // difference measured ~6x the old fixed floor, so stability lived under 0.35
    // and every threshold above it was dead. The meter must converge to >= 0.9
    // on that rig's REST — and still drop when something actually moves.
    const meter = new MotionMeter();
    // resting frames whose alternation yields a constant mean diff of ~0.024
    const restA = flat(0.5);
    const restB = flat(0.5 + 0.024);
    for (let i = 0; i < 80; i++) meter.update(i % 2 === 0 ? restA : restB);
    expect(meter.stability).toBeGreaterThanOrEqual(0.9);
    // a real move differences several times the resting scatter
    meter.update(disturbed(0.5, 0.5, 1));
    expect(meter.stability).toBeLessThan(0.7);
  });

  it('rebase() starts a new reference but keeps the learned floor', () => {
    const meter = new MotionMeter();
    const restA = flat(0.5);
    const restB = flat(0.5 + 0.024);
    for (let i = 0; i < 80; i++) meter.update(i % 2 === 0 ? restA : restB);
    // A new crop: a different picture, not a moving one.
    meter.rebase();
    expect(meter.update(flat(0.1))).toBe(1);
    // The floor survived: the same resting scatter on the new picture is still rest
    // (a reset floor, back at the noise bound, would read it as heavy motion).
    meter.update(flat(0.1 + 0.024));
    expect(meter.stability).toBeGreaterThanOrEqual(0.9);
  });

  it('re-reports 1.0 after a reset, as on a fresh camera', () => {
    const meter = new MotionMeter();
    meter.update(flat(0.5));
    meter.update(disturbed(0.5, 0.5, 1));
    meter.reset();
    expect(meter.update(flat(0.9))).toBe(1);
  });

  it('never returns a value outside [0, 1]', () => {
    const meter = new MotionMeter();
    for (const thumb of [flat(0), disturbed(0, 1, 1), flat(1), disturbed(1, -1, 0.5)]) {
      const s = meter.update(thumb);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});

/** An RGBA image whose every pixel is gray `values[i]` (bytes). */
function grayRgba(width: number, height: number, values: ArrayLike<number>): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) data.fill(values[i]!, 4 * i, 4 * i + 3);
  return { data, width, height };
}

describe('sampleThumbnail', () => {
  it('uses cv2 BGR2GRAY weights, so a scene scores the same as on the edge', () => {
    const data = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    const gray = sampleThumbnail({ data, width: 3, height: 1 }, 1);
    expect(gray[0]).toBeCloseTo((0.299 + 0.587 + 0.114) / 3, 6);
  });

  it('averages each block — an area downscale, like the edge\'s INTER_AREA', () => {
    // 4×2 -> 2×1 blocks of 2×2... as a 4×4 image into a 2×2 thumbnail
    const values = [0, 255, 51, 51, 255, 0, 51, 51, 102, 102, 0, 0, 102, 102, 0, 0];
    const thumb = sampleThumbnail(grayRgba(4, 4, values), 2);
    expect(Array.from(thumb).map((v) => Math.round(v * 255))).toEqual([128, 51, 102, 0]);
  });

  it('refuses an image that does not divide into whole blocks', () => {
    expect(() => sampleThumbnail(grayRgba(5, 4, new Array(20).fill(0)), 2)).toThrow(RangeError);
  });

  it('allocates a fresh buffer every call — the meter RETAINS what it is given', () => {
    const image = grayRgba(4, 4, new Array(16).fill(0));
    expect(sampleThumbnail(image, 4)).not.toBe(sampleThumbnail(image, 4));
  });
});

/**
 * The 2026-11-24 live take, reproduced: a phone on a holder, metered at
 * CAPTURE_FPS. The canvas the page draws on carries the camera's per-pixel sensor
 * noise (σ chosen so a point-sampled 32×32 thumbnail rests at the ~0.02 mean
 * difference that take's startup ramp implies; an area thumbnail then rests at
 * ~0.0024, as that take's recorded frames do), and a hand is a dark blob crossing
 * the board at ~1.4 % of its width per frame: ~0.007 per frame on an area
 * thumbnail, ~0.017 between frames a second apart (the take: 0.014–0.047).
 */
describe('a hand on a noisy phone camera', () => {
  const SIDE = MOTION_DEFAULTS.downscale;
  const DRAW = SIDE * MOTION_OVERSAMPLE;
  const SIGMA = 0.017;

  /** A seeded uniform PRNG (mulberry32), so the scene is the same on every run. */
  function prng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  type Hand = { readonly x: number; readonly y: number } | null;

  /** One drawn frame: a chessboard at `gain` exposure, the hand, the sensor noise. */
  function drawnFrame(random: () => number, gain: number, hand: Hand): Float32Array {
    const out = new Float32Array(DRAW * DRAW);
    const square = DRAW / 8;
    const radius = DRAW * 0.12;
    for (let y = 0; y < DRAW; y++) {
      for (let x = 0; x < DRAW; x++) {
        const light = (Math.floor(x / square) + Math.floor(y / square)) % 2 === 0;
        let v = gain * (light ? 0.62 : 0.38);
        if (hand !== null && (x - hand.x) ** 2 + (y - hand.y) ** 2 < radius ** 2) v = 0.18;
        // Box–Muller: Gaussian sensor noise on every pixel the canvas holds.
        const n = Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
        out[y * DRAW + x] = v + SIGMA * n;
      }
    }
    return out;
  }

  /** Area average — what `sampleThumbnail` computes, on gray input. */
  function areaThumbnail(drawn: Float32Array): Float32Array {
    const k = MOTION_OVERSAMPLE;
    const out = new Float32Array(SIDE * SIDE);
    for (let y = 0; y < DRAW; y++)
      for (let x = 0; x < DRAW; x++) out[Math.floor(y / k) * SIDE + Math.floor(x / k)]! += drawn[y * DRAW + x]!;
    return out.map((v) => v / (k * k));
  }

  /** The OLD path: a canvas drawn straight at 32×32 keeps one sample per pixel. */
  function pointThumbnail(drawn: Float32Array): Float32Array {
    const k = MOTION_OVERSAMPLE;
    const out = new Float32Array(SIDE * SIDE);
    for (let y = 0; y < SIDE; y++)
      for (let x = 0; x < SIDE; x++) out[y * SIDE + x] = drawn[(y * k + k / 2) * DRAW + x * k + k / 2]!;
    return out;
  }

  /** 2 s exposure ramp, 3 s rest, a 1.5 s hand crossing, 1 s rest — at 20 fps. */
  function session(): { frames: Float32Array[]; handFrames: Set<number> } {
    const random = prng(20261124);
    const frames: Float32Array[] = [];
    const handFrames = new Set<number>();
    for (let i = 0; i < 40; i++) frames.push(drawnFrame(random, 0.75 + (0.25 * i) / 39, null));
    for (let i = 0; i < 60; i++) frames.push(drawnFrame(random, 1, null));
    for (let i = 0; i < 30; i++) {
      handFrames.add(frames.length);
      frames.push(drawnFrame(random, 1, { x: DRAW * (0.1 + 0.8 * (i / 29)), y: DRAW * 0.55 }));
    }
    for (let i = 0; i < 20; i++) frames.push(drawnFrame(random, 1, null));
    return { frames, handFrames };
  }

  const { frames, handFrames } = session();

  function lowest(thumbnail: (drawn: Float32Array) => Float32Array): { rest: number; hand: number } {
    const meter = new MotionMeter();
    let rest = 1;
    let hand = 1;
    frames.forEach((drawn, i) => {
      const s = meter.update(thumbnail(drawn));
      if (handFrames.has(i)) hand = Math.min(hand, s);
      else if (i >= 70 && i < 100) rest = Math.min(rest, s); // settled rest, before the hand
    });
    return { rest, hand };
  }

  it('point-sampled thumbnails bury the hand in sensor noise — the live bug', () => {
    const { hand } = lowest(pointThumbnail);
    expect(hand).toBeGreaterThan(ENTER_BURST); // no burst would ever open
  });

  it('area-averaged thumbnails open a burst for the hand and read rest as rest', () => {
    const { rest, hand } = lowest(areaThumbnail);
    expect(hand).toBeLessThan(ENTER_BURST);
    expect(rest).toBeGreaterThan(EXIT_BURST);
  });

  it('sampleThumbnail is that area average', () => {
    const drawn = frames[120]!;
    const bytes = drawn.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255));
    const ours = sampleThumbnail(grayRgba(DRAW, DRAW, bytes), SIDE);
    const reference = areaThumbnail(bytes.map((v) => v / 255));
    for (let i = 0; i < ours.length; i++) expect(ours[i]).toBeCloseTo(reference[i]!, 3);
  });
});

/** Drives the forwarder without any pixels: the meter is stubbed to a fixed reading. */
class FixedMeter extends MotionMeter {
  constructor(private value: number) {
    super();
  }
  set(value: number): void {
    this.value = value;
  }
  override update(): number {
    this.stability = this.value;
    return this.value;
  }
}

describe('AdaptiveForwarder', () => {
  const thumb = flat(0.5);

  it('forwards the first frame it ever sees', () => {
    const forwarder = new AdaptiveForwarder(new FixedMeter(1));
    expect(forwarder.decide(thumb, { boardId: 'b', captureTs: 100 })).not.toBeNull();
  });

  it('paces the keepalive at 1 fps against CAPTURE time', () => {
    const forwarder = new AdaptiveForwarder(new FixedMeter(1));
    forwarder.decide(thumb, { boardId: 'b', captureTs: 100 });
    // Sampled at 20 fps; only the frame a full keepalive interval later goes out.
    let sent = 0;
    for (let i = 1; i <= 20; i++) {
      if (forwarder.decide(thumb, { boardId: 'b', captureTs: 100 + i / 20 }) !== null) sent++;
    }
    expect(sent).toBe(KEEPALIVE_FPS);
  });

  it('jumps to the burst rate the instant stability falls', () => {
    const meter = new FixedMeter(1);
    const forwarder = new AdaptiveForwarder(meter);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0 });
    meter.set(ENTER_BURST - 0.01);
    // One twentieth of a second later: too soon for the keepalive, due at burst rate.
    const decision = forwarder.decide(thumb, { boardId: 'b', captureTs: 1 / BURST_FPS });
    expect(decision).not.toBeNull();
    expect(forwarder.bursting).toBe(true);
    expect(decision?.burstId).toBe('b:b0');
  });

  it('holds the burst through a pause between two strides of one move (hysteresis)', () => {
    const meter = new FixedMeter(1);
    const forwarder = new AdaptiveForwarder(meter);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0 });
    meter.set(0.2);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.1 });
    expect(forwarder.bursting).toBe(true);
    // Calm, but not for long enough: still one burst.
    meter.set(EXIT_BURST + 0.05);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.2 });
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.2 + EXIT_HOLD_S / 2 });
    expect(forwarder.bursting).toBe(true);
    // Sustained calm ends it.
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.2 + EXIT_HOLD_S });
    expect(forwarder.bursting).toBe(false);
  });

  it('restarts the calm hold on any dip back below the exit threshold', () => {
    const meter = new FixedMeter(1);
    const forwarder = new AdaptiveForwarder(meter);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0 });
    meter.set(0.2);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.1 });
    meter.set(EXIT_BURST + 0.05);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.2 });
    meter.set(EXIT_BURST); // <= EXIT_BURST resets the hold
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.3 });
    meter.set(EXIT_BURST + 0.05);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0.3 + EXIT_HOLD_S - 0.01 });
    expect(forwarder.bursting).toBe(true);
  });

  it('never reuses a burst id within a session', () => {
    const meter = new FixedMeter(1);
    const forwarder = new AdaptiveForwarder(meter);
    const ids: string[] = [];
    let ts = 0;
    for (let burst = 0; burst < 3; burst++) {
      meter.set(0.2);
      ids.push(forwarder.decide(thumb, { boardId: 'b', captureTs: (ts += 1) })?.burstId ?? '');
      meter.set(1);
      forwarder.decide(thumb, { boardId: 'b', captureTs: (ts += 0.1) });
      forwarder.decide(thumb, { boardId: 'b', captureTs: (ts += EXIT_HOLD_S) });
      expect(forwarder.bursting).toBe(false);
    }
    expect(new Set(ids).size).toBe(3);
  });

  it('re-anchors rather than stalling when the clock steps backwards', () => {
    const forwarder = new AdaptiveForwarder(new FixedMeter(1));
    forwarder.decide(thumb, { boardId: 'b', captureTs: 1000 });
    expect(forwarder.due(500)).toBe(true);
  });

  it('meters every frame, including the ones it does not send', () => {
    const meter = new FixedMeter(1);
    const forwarder = new AdaptiveForwarder(meter);
    forwarder.decide(thumb, { boardId: 'b', captureTs: 0 });
    meter.set(0.1);
    // Not due at the keepalive rate — but the burst still started from this frame.
    const decision = forwarder.decide(thumb, { boardId: 'b', captureTs: 0.001 });
    expect(decision).toBeNull();
    expect(forwarder.bursting).toBe(true);
  });

  it('forgets the pacing on reset, but not the burst counter', () => {
    const meter = new FixedMeter(0.2);
    const forwarder = new AdaptiveForwarder(meter);
    const first = forwarder.decide(thumb, { boardId: 'b', captureTs: 0 })?.burstId;
    forwarder.reset();
    const second = forwarder.decide(thumb, { boardId: 'b', captureTs: 0 })?.burstId;
    expect(second).not.toBe(first);
  });
});
