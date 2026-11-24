import { describe, expect, it } from 'vitest';
import { MOTION_STALE_MS, motionReadout } from './positions-motion';

describe('motionReadout', () => {
  const sample = { moving: false, stability: 0.99, at: 1000 };

  it('is none for a board that never sent motion', () => {
    expect(motionReadout(undefined, 1000)).toEqual({ kind: 'none' });
  });

  it('says moving or still while the last sample is fresh', () => {
    const fresh = 1000 + MOTION_STALE_MS;
    expect(motionReadout(sample, fresh)).toEqual({ kind: 'still', stability: 0.99 });
    const moving = { ...sample, moving: true };
    expect(motionReadout(moving, 1000)).toEqual({ kind: 'moving', stability: 0.99 });
  });

  it('goes stale once no sample arrived for MOTION_STALE_MS', () => {
    const moving = { ...sample, moving: true };
    expect(motionReadout(moving, 1001 + MOTION_STALE_MS)).toEqual({ kind: 'stale' });
  });
});
