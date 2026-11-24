import { describe, expect, it } from 'vitest';
import { MOTION_STALE_MS, motionReadout, parseMotionMessage } from './positions-motion';

const MESSAGE = { type: 'motion', board_id: 'phone', moving: true, stability: 0.42, ts: 12.5 };

describe('parseMotionMessage', () => {
  it('accepts the node message as sent', () => {
    expect(parseMotionMessage(MESSAGE)).toEqual(MESSAGE);
  });

  it('clamps stability into [0, 1]', () => {
    expect(parseMotionMessage({ ...MESSAGE, stability: 1.7 })?.stability).toBe(1);
    expect(parseMotionMessage({ ...MESSAGE, stability: -0.2 })?.stability).toBe(0);
  });

  it('refuses a position event, a wrong type and a malformed field', () => {
    const position = { board_id: 'phone', seq: 1, placement: '8/8/8/8/8/8/8/8' };
    expect(parseMotionMessage(position)).toBeNull();
    expect(parseMotionMessage({ ...MESSAGE, type: 'frame' })).toBeNull();
    expect(parseMotionMessage({ ...MESSAGE, moving: 'yes' })).toBeNull();
    expect(parseMotionMessage({ ...MESSAGE, board_id: '' })).toBeNull();
    expect(parseMotionMessage({ ...MESSAGE, ts: null })).toBeNull();
    expect(parseMotionMessage(undefined)).toBeNull();
  });
});

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
