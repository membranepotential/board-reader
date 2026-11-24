import { describe, expect, it } from 'vitest';
import { MIN_CROP_SIDE, WHOLE_FRAME, dragRect, isValidCrop } from './crop';

const RECT = { x: 0.2, y: 0.3, w: 0.4, h: 0.5 };

describe('dragRect', () => {
  it('moves the whole rectangle by the delta', () => {
    expect(dragRect(RECT, 'move', { dx: 0.1, dy: -0.1 })).toEqual({ x: 0.3, y: 0.2, w: 0.4, h: 0.5 });
  });

  it('stops a moved rectangle at the frame edge with its size intact', () => {
    expect(dragRect(RECT, 'move', { dx: 5, dy: -5 })).toEqual({ x: 0.6, y: 0, w: 0.4, h: 0.5 });
  });

  it('moves only the dragged corner; the opposite corner stays', () => {
    expect(dragRect(RECT, 'nw', { dx: -0.1, dy: -0.1 })).toEqual({ x: 0.1, y: 0.2, w: 0.5, h: 0.6 });
    expect(dragRect(RECT, 'se', { dx: 0.1, dy: 0.1 })).toEqual({ x: 0.2, y: 0.3, w: 0.5, h: 0.6 });
    expect(dragRect(RECT, 'ne', { dx: 0.1, dy: 0.1 })).toEqual({ x: 0.2, y: 0.4, w: 0.5, h: 0.4 });
    expect(dragRect(RECT, 'sw', { dx: 0.1, dy: 0.1 })).toEqual({ x: 0.3, y: 0.3, w: 0.3, h: 0.6 });
  });

  it('never shrinks a side below the minimum, nor lets a corner cross its opposite', () => {
    const squashed = dragRect(RECT, 'se', { dx: -1, dy: -1 });
    expect(squashed).toEqual({ x: 0.2, y: 0.3, w: MIN_CROP_SIDE, h: MIN_CROP_SIDE });
    expect(dragRect(RECT, 'nw', { dx: 1, dy: 1 })).toEqual({
      x: 0.55,
      y: 0.75,
      w: MIN_CROP_SIDE,
      h: MIN_CROP_SIDE
    });
  });

  it('clamps a corner dragged past the frame onto its edge', () => {
    expect(dragRect(WHOLE_FRAME, 'nw', { dx: -0.3, dy: -0.3 })).toEqual(WHOLE_FRAME);
    expect(dragRect(WHOLE_FRAME, 'nw', { dx: 0.25, dy: 0.1 })).toEqual({ x: 0.25, y: 0.1, w: 0.75, h: 0.9 });
  });

  it('always returns a rectangle the node accepts, whatever the drag', () => {
    // Odd fractions on purpose: unsnapped float edges are what land a hair past 1.
    const deltas = [-2, -0.3337, -0.0001, 0, 0.1234567, 0.7, 3];
    for (const handle of ['move', 'nw', 'ne', 'se', 'sw'] as const) {
      for (const dx of deltas) {
        for (const dy of deltas) {
          const out = dragRect({ x: 0.1111, y: 0.2222, w: 0.3333, h: 0.4444 }, handle, { dx, dy });
          expect(isValidCrop(out), `${handle} ${dx},${dy} -> ${JSON.stringify(out)}`).toBe(true);
        }
      }
    }
  });
});

describe('isValidCrop', () => {
  it('mirrors the node: inside the frame, both sides at least the minimum', () => {
    expect(isValidCrop(WHOLE_FRAME)).toBe(true);
    expect(isValidCrop({ x: 0.5, y: 0, w: 0.6, h: 0.5 })).toBe(false);
    expect(isValidCrop({ x: -0.01, y: 0, w: 0.5, h: 0.5 })).toBe(false);
    expect(isValidCrop({ x: 0, y: 0, w: MIN_CROP_SIDE - 0.001, h: 0.5 })).toBe(false);
  });
});
