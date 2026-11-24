import { describe, expect, it } from 'vitest';
import { DEFAULT_CAMERA_CONFIG } from '$lib/protocol';
import type { CameraView } from './client';
import { formatAge, frameAge, frameRate, phoneConnected, reportAge } from './status';

function view(overrides: Partial<CameraView> = {}): CameraView {
  return {
    board_id: 'phone',
    config: DEFAULT_CAMERA_CONFIG,
    report: null,
    seen_at: null,
    preview: null,
    frame: null,
    now: 1000,
    ...overrides
  };
}

describe('ages on the node clock', () => {
  it('measures the report and the last frame against view.now, not the browser', () => {
    const v = view({ seen_at: 999.5, frame: { seq: 4, received_at: 990, width: 960, height: 540 } });
    expect(reportAge(v)).toBe(0.5);
    expect(frameAge(v)).toBe(10);
    expect(phoneConnected(v)).toBe(true);
  });

  it('calls a phone that never reported, or went quiet, not connected', () => {
    expect(phoneConnected(view())).toBe(false);
    expect(reportAge(view())).toBeNull();
    expect(phoneConnected(view({ seen_at: 990 }))).toBe(false);
  });

  it('formats an age for the operator', () => {
    expect(formatAge(null)).toBe('never');
    expect(formatAge(2.34)).toBe('2.3 s ago');
    expect(formatAge(240)).toBe('4 min ago');
  });
});

describe('frameRate', () => {
  it('is frames over node-clock seconds between two polls', () => {
    expect(frameRate({ seq: 10, now: 100 }, { seq: 22, now: 102 })).toBe(6);
    expect(frameRate({ seq: 10, now: 100 }, { seq: 10, now: 101 })).toBe(0);
  });

  it('has nothing to say without two samples, or when the phone restarted its count', () => {
    expect(frameRate(null, { seq: 1, now: 1 })).toBeNull();
    expect(frameRate({ seq: 50, now: 100 }, { seq: 2, now: 101 })).toBeNull();
    expect(frameRate({ seq: 1, now: 100 }, { seq: 2, now: 100 })).toBeNull();
  });
});
