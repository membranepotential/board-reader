import { describe, expect, it } from 'vitest';
import { boardShapes } from '#lib/components/snapshot/board-shapes.ts';
import { withCorrection, NO_CORRECTIONS } from '#lib/components/snapshot/reading.ts';
import { CLASSES, NUM_CLASSES, type SquareDist } from '#lib/protocol/index.ts';
import { PositionEvent } from '#lib/protocol/generated.ts';

/** Every square certain of "empty", except `doubtful` squares that split three ways. */
function distWith(doubtful: ReadonlySet<string>): SquareDist {
  const sure = CLASSES.map((cls) => (cls === 'empty' ? 1 : 0));
  const split = CLASSES.map((_, i) => (i < 3 ? 1 / 3 : 0));
  expect(sure).toHaveLength(NUM_CLASSES);
  const raw = Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => (doubtful.has(`${r},${c}`) ? split : sure))
  );
  return PositionEvent.shape.square_dist.parse(raw);
}

describe('the rings on the read board', () => {
  it('draws nothing without a distribution or a selection', () => {
    expect(boardShapes(null, NO_CORRECTIONS, null)).toEqual([]);
  });

  it('rings a doubtful square red', () => {
    const shapes = boardShapes(distWith(new Set(['0,0'])), NO_CORRECTIONS, null);
    expect(shapes).toEqual([{ orig: 'a8', brush: 'red' }]);
  });

  it('lets a hand correction win over doubt, and paints the selection last', () => {
    const corrections = withCorrection(NO_CORRECTIONS, 'a8', 'P');
    const shapes = boardShapes(distWith(new Set(['0,0'])), corrections, { r: 0, c: 0 });
    expect(shapes).toEqual([
      { orig: 'a8', brush: 'blue' },
      { orig: 'a8', brush: 'green' }
    ]);
  });
});
