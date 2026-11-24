/**
 * FEN field 1 ⇄ 8×8 grid, and the per-square reads that go with it.
 *
 * All three hand-written pages carried their own near-identical copy of this; the
 * differences between them were the bugs. This is the single one.
 *
 * **The sanitizing is the point.** The grid is indexed positionally — the change
 * diff compares `prevGrid[r][c]` to `grid[r][c]`, and the string handed to
 * chessground is rebuilt from it — so a short, over-long or malformed rank must
 * never shift the grid, leave holes in it, or throw. Chessground's own FEN reader
 * skips characters it does not recognise, which SHIFTS every later square in the
 * rank: a single stray byte on the wire would silently redraw the board wrong.
 * So a placement is read here, clamped to exactly 8×8, and re-emitted from the
 * clean grid; whatever arrives, the board renders and the diff stays aligned.
 */

import {
  FILE_LETTERS,
  GRID,
  isPieceSymbol,
  ROW_RANKS,
  toGrid8,
  type Board8,
  type Grid8,
  type PieceSymbol,
  type SquareKey
} from '../protocol';

/** One square of the sanitized grid: a piece letter, or `''` for empty. */
export type Cell = PieceSymbol | '';

export type BoardGrid = Board8<Cell>;

/**
 * FEN field 1 → an 8×8 grid, always, whatever the input looks like.
 *
 * Row 0 is rank 8 and column 0 is file a, matching the frame every `square_dist`
 * is expressed in. Digits expand to empties, short ranks are padded, long ranks
 * are truncated, and a symbol that is not one of the twelve piece letters becomes
 * an empty square rather than being passed through to the renderer.
 */
export function fenToGrid(placement: string | null | undefined): BoardGrid {
  const ranks = String(placement ?? '').split('/');
  const rows: Grid8<Cell>[] = [];
  for (let r = 0; r < GRID; r++) {
    const cells: Cell[] = [];
    for (const ch of ranks[r] ?? '') {
      if (ch >= '0' && ch <= '9') {
        for (let i = 0; i < Number(ch) && cells.length < GRID; i++) cells.push('');
      } else if (cells.length < GRID) {
        cells.push(isPieceSymbol(ch) ? ch : '');
      }
    }
    while (cells.length < GRID) cells.push('');
    rows.push(toGrid8(cells));
  }
  return toGrid8(rows);
}

/**
 * Inverse of `fenToGrid`. Chessground is fed from the sanitized grid and never
 * from the raw wire string, so a malformed placement degrades to empty squares
 * instead of throwing inside the board renderer and freezing the card.
 */
export function gridToFen(grid: BoardGrid): string {
  return grid
    .map((rank) => {
      let out = '';
      let empty = 0;
      for (const cell of rank) {
        if (cell === '') {
          empty++;
          continue;
        }
        if (empty) {
          out += empty;
          empty = 0;
        }
        out += cell;
      }
      return empty ? out + empty : out;
    })
    .join('/');
}

/**
 * Algebraic name of grid cell `(row, col)` — also its chessground key.
 *
 * Off-board indices throw: they can only come from a loop bound that is wrong
 * here, never from the wire (every parsed board is 8×8 by construction).
 */
export function squareKey(row: number, col: number): SquareKey {
  const file = FILE_LETTERS[col];
  const rank = ROW_RANKS[row];
  if (file === undefined || rank === undefined) {
    throw new RangeError(`(${row}, ${col}) is off an ${GRID}x${GRID} board`);
  }
  return `${file}${rank}`;
}

/** Squares where `next` disagrees with `previous` — the change diff, and nothing more. */
export function changedSquares(previous: BoardGrid | null, next: BoardGrid): SquareKey[] {
  if (previous === null) return [];
  const changed: SquareKey[] = [];
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      if (previous[r]![c] !== next[r]![c]) changed.push(squareKey(r, c));
    }
  }
  return changed;
}
