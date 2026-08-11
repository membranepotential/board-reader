import { describe, expect, it } from 'vitest';
import { nextRotation, parseStoredRotation, rotationKey } from './rotation';

describe('nextRotation', () => {
  it('steps through the four quarter turns in order', () => {
    expect(nextRotation(0)).toBe(90);
    expect(nextRotation(90)).toBe(180);
    expect(nextRotation(180)).toBe(270);
  });

  it('wraps back to 0 after a full turn', () => {
    expect(nextRotation(270)).toBe(0);
  });
});

describe('rotationKey', () => {
  it('scopes the storage key to one board id', () => {
    expect(rotationKey('alpha')).toBe('board-reader.live.rotation.alpha');
    expect(rotationKey('beta')).toBe('board-reader.live.rotation.beta');
  });
});

describe('parseStoredRotation', () => {
  it('defaults to unrotated when nothing was ever stored', () => {
    expect(parseStoredRotation(null)).toBe(0);
  });

  it('accepts any of the four valid quarter turns', () => {
    expect(parseStoredRotation('0')).toBe(0);
    expect(parseStoredRotation('90')).toBe(90);
    expect(parseStoredRotation('180')).toBe(180);
    expect(parseStoredRotation('270')).toBe(270);
  });

  it('falls back to 0 on garbage — a corrupted or hand-edited value never crashes the page', () => {
    expect(parseStoredRotation('45')).toBe(0);
    expect(parseStoredRotation('not-a-number')).toBe(0);
    expect(parseStoredRotation('')).toBe(0);
  });
});
