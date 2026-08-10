import { describe, expect, it } from 'vitest';
import { calibBlockReason, calibSuccessMessage, calibTitle } from './calibration';
import type { CalibrateResponse } from '../../protocol';

describe('calibBlockReason', () => {
  it('is available on a live, settled, ok reading', () => {
    expect(calibBlockReason({ stale: false, status: 'ok', stability: 0.95 })).toBeNull();
  });

  it('is available when stability is unknown (a hint, never a gate)', () => {
    expect(calibBlockReason({ stale: false, status: 'ok', stability: null })).toBeNull();
  });

  it('blocks on a stale stream first, regardless of the other fields', () => {
    expect(calibBlockReason({ stale: true, status: 'recalibrating', stability: 0.1 })).toBe(
      'the stream is stale — no live frame to calibrate from'
    );
  });

  it('blocks on a non-ok status', () => {
    expect(calibBlockReason({ stale: false, status: 'needs_setup', stability: 1 })).toBe(
      'the reading is "needs_setup" — calibrate from an ok reading'
    );
  });

  it('blocks on a moving board', () => {
    expect(calibBlockReason({ stale: false, status: 'ok', stability: 0.5 })).toBe(
      'the board is moving — calibrate with the pieces at rest'
    );
  });
});

describe('calibTitle', () => {
  it('reads "calibrating…" while busy, even if otherwise blocked', () => {
    expect(calibTitle('the board is moving — calibrate with the pieces at rest', true)).toBe(
      'calibrating…'
    );
  });

  it('surfaces the block reason as the title', () => {
    expect(calibTitle('the stream is stale — no live frame to calibrate from', false)).toBe(
      'the stream is stale — no live frame to calibrate from'
    );
  });

  it('explains the action when nothing blocks it', () => {
    expect(calibTitle(null, false)).toContain('rebuild the piece bank');
  });
});

describe('calibSuccessMessage', () => {
  const base: CalibrateResponse = {
    classes_calibrated: 13,
    backbone: 'resnet18',
    saved: null,
    exemplars: 26,
    label_rotation: 0
  };

  it('reports classes calibrated and exemplar count', () => {
    expect(calibSuccessMessage(base)).toBe('Calibrated 13/13 classes — bank now holds 26 exemplars');
  });

  it('mentions a non-zero label rotation in degrees', () => {
    expect(calibSuccessMessage({ ...base, label_rotation: 1 })).toBe(
      'Calibrated 13/13 classes (corner labels auto-corrected by 90 degrees) — bank now holds 26 exemplars'
    );
  });
});
