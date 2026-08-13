import { describe, expect, it } from 'vitest';
import { boardShapes, pendingShapes } from './shapes';
import type { ChangeRing } from '../../stores/liveness';

describe('boardShapes', () => {
  it('ring a low-confidence square yellow', () => {
    expect(boardShapes(['e4'], [])).toEqual([{ orig: 'e4', brush: 'yellow' }]);
  });

  it('rings a fresh change green and a fading one paleGreen', () => {
    const rings: ChangeRing[] = [
      { square: 'a1', phase: 'fresh' },
      { square: 'h8', phase: 'fading' }
    ];
    expect(boardShapes([], rings)).toEqual([
      { orig: 'a1', brush: 'green' },
      { orig: 'h8', brush: 'paleGreen' }
    ]);
  });

  it('combines low-confidence and change rings, low-confidence first', () => {
    const rings: ChangeRing[] = [{ square: 'd4', phase: 'fresh' }];
    expect(boardShapes(['e4'], rings)).toEqual([
      { orig: 'e4', brush: 'yellow' },
      { orig: 'd4', brush: 'green' }
    ]);
  });

  it('draws nothing for a settled, confident board', () => {
    expect(boardShapes([], [])).toEqual([]);
  });
});

describe('prior flips', () => {
  it('rings a repaired square red, after the low rings', () => {
    expect(boardShapes(['e4'], [], ['e8'])).toEqual([
      { orig: 'e4', brush: 'yellow' },
      { orig: 'e8', brush: 'red' }
    ]);
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
  it('rings a pending square after the repair rings, ahead of the change rings', () => {
    const rings: ChangeRing[] = [{ square: 'd4', phase: 'fresh' }];
    const shapes = boardShapes(['e4'], rings, ['e8'], ['c3']);
    expect(shapes.map((s) => s.orig)).toEqual(['e4', 'e8', 'c3', 'd4']);
    expect(shapes[2]!.brush).toBeUndefined();
    expect(shapes[2]!.customSvg).toBeDefined();
  });

  it('draws no pending ring when nothing is pending', () => {
    expect(boardShapes(['e4'], [])).toEqual([{ orig: 'e4', brush: 'yellow' }]);
  });
});
