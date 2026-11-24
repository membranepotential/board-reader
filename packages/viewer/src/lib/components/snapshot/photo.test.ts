import { describe, expect, it } from 'vitest';
import { meanLuma } from './photo';

describe('meanLuma', () => {
  it('is the mean of the channels, subsampled', () => {
    const width = 8;
    const height = 8;
    const data = new Uint8ClampedArray(width * height * 4).fill(120);
    expect(meanLuma({ data, width, height })).toBeCloseTo(120, 9);
  });

  it('is 0 for an empty image rather than NaN', () => {
    expect(meanLuma({ data: new Uint8ClampedArray(0), width: 0, height: 0 })).toBe(0);
  });
});
