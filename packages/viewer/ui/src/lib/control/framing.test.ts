import { describe, expect, it } from 'vitest';
import { framingWarnings } from './framing';

const LANDSCAPE = { width: 1920, height: 1080 };
const PORTRAIT = { width: 1080, height: 1920 };

const kinds = (...args: Parameters<typeof framingWarnings>) => framingWarnings(...args).map((w) => w.kind);

describe('framingWarnings', () => {
  it('is quiet for a landscape crop on a landscape phone', () => {
    // 0.5 x 0.6 of 16:9 is 960 x 648 px: wider than tall, well under 2:1.
    expect(kinds({ x: 0.25, y: 0.2, w: 0.5, h: 0.6 }, LANDSCAPE)).toEqual([]);
  });

  it('says so when the phone is upright, once, however the rectangle is drawn', () => {
    expect(kinds({ x: 0, y: 0.2, w: 1, h: 0.4 }, PORTRAIT)).toEqual(['portrait-phone']);
    expect(kinds({ x: 0.2, y: 0.2, w: 0.3, h: 0.6 }, PORTRAIT)).toEqual(['portrait-phone']);
  });

  it('measures the rectangle in pixels, not fractions', () => {
    // Equal fractions are 16:9 on a landscape frame (fine), but 0.3 x 0.6 is
    // 576 x 648 px — taller than wide.
    expect(kinds({ x: 0, y: 0, w: 0.5, h: 0.5 }, LANDSCAPE)).toEqual([]);
    expect(kinds({ x: 0, y: 0, w: 0.3, h: 0.6 }, LANDSCAPE)).toEqual(['portrait-crop']);
  });

  it('warns about a strip so wide the board fills under half of it', () => {
    // 1 x 0.4 of 16:9 is 1920 x 432 px: 4.4 to 1.
    expect(kinds({ x: 0, y: 0.3, w: 1, h: 0.4 }, LANDSCAPE)).toEqual(['too-wide']);
  });

  it('flags the whole frame being sent, and is silent with no camera yet', () => {
    expect(kinds(null, LANDSCAPE)).toEqual(['whole-frame']);
    expect(kinds(null, null)).toEqual([]);
  });
});
