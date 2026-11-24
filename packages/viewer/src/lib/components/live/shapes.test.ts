import { describe, expect, it } from 'vitest';
import { boardShapes, pendingShapes } from './shapes';
import type { ChangeRing } from '../../stores/liveness';

describe('boardShapes', () => {
  it('rings a fresh change green and a fading one paleGreen', () => {
    const rings: ChangeRing[] = [
      { square: 'a1', phase: 'fresh' },
      { square: 'h8', phase: 'fading' }
    ];
    expect(boardShapes(rings)).toEqual([
      { orig: 'a1', brush: 'green' },
      { orig: 'h8', brush: 'paleGreen' }
    ]);
  });

  it('draws nothing for a settled board with no recent change', () => {
    expect(boardShapes([])).toEqual([]);
  });
});

describe('pendingShapes', () => {
  it('draws a dashed customSvg ring, not a named brush, for each pending square', () => {
    const shapes = pendingShapes(['e4', 'e2']);
    expect(shapes).toHaveLength(2);
    for (const [i, square] of ['e4', 'e2'].entries()) {
      expect(shapes[i]!.orig).toBe(square);
      expect(shapes[i]!.brush).toBeUndefined();
      expect(shapes[i]!.customSvg?.center).toBe('orig');
      expect(shapes[i]!.customSvg?.html).toMatch(/stroke-dasharray/);
    }
  });

  it('draws nothing for no pending squares', () => {
    expect(pendingShapes([])).toEqual([]);
  });
});

describe('settled pending', () => {
  it('rings a pending square ahead of the change rings', () => {
    const rings: ChangeRing[] = [{ square: 'd4', phase: 'fresh' }];
    const shapes = boardShapes(rings, ['c3']);
    expect(shapes.map((s) => s.orig)).toEqual(['c3', 'd4']);
    expect(shapes[0]!.brush).toBeUndefined();
    expect(shapes[0]!.customSvg).toBeDefined();
  });
});
