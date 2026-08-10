import { describe, expect, it } from 'vitest';
import {
  AdaptiveForwarder,
  BURST_FPS,
  ENTER_BURST,
  EXIT_BURST,
  EXIT_HOLD_S,
  KEEPALIVE_FPS,
  MOTION_DEFAULTS,
  MotionMeter,
  sampleThumbnail
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

describe('sampleThumbnail', () => {
  it('uses cv2 BGR2GRAY weights, so a scene scores the same as on the edge', () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    const gray = sampleThumbnail(rgba, 3);
    expect(gray[0]).toBeCloseTo(0.299, 6);
    expect(gray[1]).toBeCloseTo(0.587, 6);
    expect(gray[2]).toBeCloseTo(0.114, 6);
  });

  it('allocates a fresh buffer every call — the meter RETAINS what it is given', () => {
    const rgba = new Uint8ClampedArray(4 * 4);
    expect(sampleThumbnail(rgba, 4)).not.toBe(sampleThumbnail(rgba, 4));
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
