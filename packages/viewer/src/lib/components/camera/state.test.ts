import { describe, expect, it } from 'vitest';
import { DEFAULT_CAMERA_CONFIG } from '#lib/protocol/index.ts';
import type { CameraConfig } from '#lib/protocol/generated.ts';
import {
  cropBox,
  fitLongSide,
  freshFrameClock,
  nextCapture,
  noteFrameClock,
  ratePerSecond,
  STALL_PLAY_MS,
  STALL_REOPEN_MAX_MS,
  STALL_REOPEN_MS,
  videoConstraints,
  withinRateWindow,
  type Capture
} from './state';

const FRAME = { width: 1920, height: 1080 } as const;
const LENSES = [
  { id: 'a', label: 'Back Camera' },
  { id: 'b', label: 'Back Ultra Wide Camera' }
] as const;
const DEFAULT_CAPTURE: Capture = { device: null, resolution: '1080p' };

function config(overrides: Partial<CameraConfig>): CameraConfig {
  return { ...DEFAULT_CAMERA_CONFIG, ...overrides };
}

describe('applying a config to the camera', () => {
  it('asks for nothing when the camera already is what the config says', () => {
    expect(nextCapture(DEFAULT_CAPTURE, DEFAULT_CAMERA_CONFIG, LENSES)).toBeNull();
  });

  it('asks for the configured lens and resolution together', () => {
    expect(
      nextCapture(DEFAULT_CAPTURE, config({ device: 'Back Ultra Wide Camera', resolution: '2160p' }), LENSES)
    ).toEqual({ device: 'Back Ultra Wide Camera', resolution: '2160p' });
  });

  it('keeps the current lens when the configured one is not on this phone', () => {
    const current: Capture = { device: 'Back Camera', resolution: '1080p' };
    expect(nextCapture(current, config({ device: 'Telephoto' }), LENSES)).toBeNull();
    expect(nextCapture(current, config({ device: 'Telephoto', resolution: '720p' }), LENSES)).toEqual({
      device: 'Back Camera',
      resolution: '720p'
    });
  });

  it('goes back to the default camera when the config clears the lens', () => {
    const current: Capture = { device: 'Back Camera', resolution: '1080p' };
    expect(nextCapture(current, DEFAULT_CAMERA_CONFIG, LENSES)).toEqual(DEFAULT_CAPTURE);
  });

  it('asks for a size as an ideal, and for a named lens exactly', () => {
    expect(videoConstraints('720p', null)).toEqual({
      facingMode: { ideal: 'environment' },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    });
    expect(videoConstraints('2160p', 'b')).toMatchObject({ deviceId: { exact: 'b' } });
  });
});

describe('the crop', () => {
  it('is the whole frame when none is configured', () => {
    expect(cropBox(null, FRAME)).toEqual({ x: 0, y: 0, w: 1920, h: 1080 });
  });

  it('maps fractions of the frame to whole pixels inside it', () => {
    expect(cropBox({ x: 0.25, y: 0.5, w: 0.5, h: 0.5 }, FRAME)).toEqual({ x: 480, y: 540, w: 960, h: 540 });
    // The same fractions on a different resolution: the rectangle follows the board.
    expect(cropBox({ x: 0.25, y: 0.5, w: 0.5, h: 0.5 }, { width: 1280, height: 720 })).toEqual({
      x: 320,
      y: 360,
      w: 640,
      h: 360
    });
  });

  it('is scaled down to the send size but never up', () => {
    expect(fitLongSide({ w: 1920, h: 1080 }, 960)).toEqual({ w: 960, h: 540 });
    expect(fitLongSide({ w: 400, h: 300 }, 960)).toEqual({ w: 400, h: 300 });
  });
});

describe('send rate', () => {
  it('measures (n-1)/span, so a 1 fps keepalive reads as 1.0', () => {
    expect(ratePerSecond([0, 1000, 2000, 3000], true)).toBeCloseTo(1.0, 6);
  });

  it('is zero when not streaming and nothing has been sent', () => {
    expect(ratePerSecond([], false)).toBe(0);
  });

  it('drops stamps that have left the window', () => {
    expect(withinRateWindow([0, 2000, 4000], 4000)).toEqual([2000, 4000]);
  });
});

describe('camera stall watchdog', () => {
  it('an advancing clock is healthy and resets every escalation', () => {
    let clock = freshFrameClock(0);
    for (let now = 50, t = 0.05; now < 500; now += 50, t += 0.05) {
      const step = noteFrameClock(clock, t, now, false);
      clock = step.clock;
      expect(step.action).toBe('none');
    }
    expect(clock.reopens).toBe(0);
  });

  it('escalates play -> reopen -> backed-off reopen while the clock is stuck', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock; // first reading

    let step = noteFrameClock(clock, 1.0, STALL_PLAY_MS, false);
    expect(step.action).toBe('play');
    clock = step.clock;

    // play() did nothing; still stalled at the reopen threshold
    step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS, false);
    expect(step.action).toBe('reopen');
    clock = step.clock;
    expect(clock.reopens).toBe(1);
    expect(clock.nextReopenAt).toBe(STALL_REOPEN_MS + STALL_REOPEN_MS);

    // before the backoff expires: quiet, not a reopen storm
    step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS + 100, false);
    expect(step.action).toBe('none');

    // after it: the second reopen, with a doubled backoff
    step = noteFrameClock(clock, 1.0, clock.nextReopenAt, false);
    expect(step.action).toBe('reopen');
    expect(step.clock.nextReopenAt - clock.nextReopenAt).toBe(2 * STALL_REOPEN_MS);
  });

  it('caps the reopen backoff', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock;
    clock = noteFrameClock(clock, 1.0, STALL_PLAY_MS, false).clock; // play, spent
    for (let i = 0; i < 8; i++) {
      const at = Math.max(clock.nextReopenAt, STALL_REOPEN_MS);
      const step = noteFrameClock(clock, 1.0, at, false);
      expect(step.action).toBe('reopen');
      expect(step.clock.nextReopenAt - at).toBeLessThanOrEqual(STALL_REOPEN_MAX_MS);
      clock = step.clock;
    }
  });

  it('a hidden page is never a stall, and returning starts a fresh grace period', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock;
    // hidden through what would have been a stall
    let step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS * 2, true);
    expect(step.action).toBe('none');
    clock = step.clock;
    // just visible again: still no action — the grace period restarted
    step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS * 2 + 100, false);
    expect(step.action).toBe('none');
  });

  it('recovery is proven by the clock advancing, which zeroes the backoff', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock;
    clock = noteFrameClock(clock, 1.0, STALL_REOPEN_MS, false).clock; // reopen #1
    const step = noteFrameClock(clock, 0.1, STALL_REOPEN_MS + 500, false); // new stream
    expect(step.action).toBe('none');
    expect(step.clock.reopens).toBe(0);
    expect(step.clock.nextReopenAt).toBe(0);
  });
});
