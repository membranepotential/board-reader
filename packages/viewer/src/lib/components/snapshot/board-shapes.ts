import { squareKey } from '#lib/board/fen.ts';
import {
  confidenceOf,
  scoresAt,
  type Corrections
} from '#lib/components/snapshot/reading.ts';
import { GRID, LOW_CONFIDENCE, type SquareDist } from '#lib/protocol/index.ts';
import type { DrawShape } from '@lichess-org/chessground/draw';

/**
 * The rings on the read board: red under `LOW_CONFIDENCE`, blue for a hand correction
 * (which wins over red), and green last so the inspected square sits on top.
 */
export function boardShapes(
  dist: SquareDist | null,
  corrections: Corrections,
  selected: { readonly r: number; readonly c: number } | null
): DrawShape[] {
  const shapes: DrawShape[] = [];
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      const square = squareKey(r, c);
      if (corrections.has(square)) {
        shapes.push({ orig: square, brush: 'blue' });
      } else if (confidenceOf(scoresAt(dist, r, c)) < LOW_CONFIDENCE) {
        shapes.push({ orig: square, brush: 'red' });
      }
    }
  }
  if (selected !== null) shapes.push({ orig: squareKey(selected.r, selected.c), brush: 'green' });
  return shapes;
}
