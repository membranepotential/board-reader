/**
 * The position the node read, the hand corrections layered over it, and the square under
 * inspection.
 *
 * The node's read is never mutated: `placement` plus `square_dist` is the baseline, and
 * corrections are a separate layer over it. So "reset corrections" restores exactly what
 * the node served, and everything shown is derived from the two.
 */
import { squareKey } from '#lib/board/fen.ts';
import { boardShapes } from '#lib/components/snapshot/board-shapes.ts';
import {
  NO_CORRECTIONS,
  assignmentFor,
  baselineGrid,
  correctedGrid,
  cycleTarget,
  placementOfGrid,
  scoresAt,
  shownClassAt,
  weakestSquares,
  withCorrection,
  type Corrections
} from '#lib/components/snapshot/reading.ts';
import {
  GRID,
  WARN_LOW_LIGHT,
  type ClassName,
  type EstimateResponse,
  type SquareKey
} from '#lib/protocol/index.ts';

export interface SelectedSquare {
  readonly r: number;
  readonly c: number;
}

const clampIndex = (n: number): number => Math.min(7, Math.max(0, n));

export class BoardState {
  // `$state<T>(null)`, never `x: T | null = $state(null)`: with the annotation on the
  // variable, TypeScript's flow analysis narrows every `$derived` that reads it to the
  // initial `null`.
  reading = $state<EstimateResponse | null>(null);
  corrections = $state<Corrections>(NO_CORRECTIONS);
  selected = $state<SelectedSquare | null>(null);

  readonly dist = $derived(this.reading?.event.square_dist ?? null);
  readonly baseGrid = $derived(this.reading === null ? null : baselineGrid(this.reading));
  readonly shownGrid = $derived(
    this.baseGrid === null ? null : correctedGrid(this.baseGrid, this.corrections)
  );
  readonly shownPlacement = $derived(
    this.shownGrid === null ? null : placementOfGrid(this.shownGrid)
  );
  /** The rings drawn on the board: low confidence, corrected, inspected. */
  readonly shapes = $derived(
    this.baseGrid === null ? [] : boardShapes(this.dist, this.corrections, this.selected)
  );
  readonly lowLight = $derived(this.reading?.warnings.includes(WARN_LOW_LIGHT) ?? false);
  /**
   * Everything the inspector needs, or nothing at all. Bundling it into one value is
   * what makes "an inspected square with no distribution behind it" unrepresentable —
   * the panel cannot be rendered without the scores it is a view of.
   */
  readonly inspection = $derived.by(() => {
    const at = this.selected;
    const grid = this.baseGrid;
    if (at === null || grid === null || this.dist === null) return null;
    const scores = scoresAt(this.dist, at.r, at.c);
    if (scores === null) return null;
    return {
      square: squareKey(at.r, at.c),
      scores,
      shown: shownClassAt(grid, this.corrections, at.r, at.c)
    };
  });

  readonly #onCorrect: (next: Corrections) => void;

  /** `onCorrect` runs after every change to the corrections, so the owner can mirror it. */
  constructor(onCorrect: (next: Corrections) => void) {
    this.#onCorrect = onCorrect;
  }

  /** Forget the read: a new photo invalidates everything read off the old one. */
  clear(): void {
    this.reading = null;
    this.corrections = NO_CORRECTIONS;
    this.selected = null;
  }

  /** A fresh read has no corrections; a restored one brings back its own. */
  load(response: EstimateResponse, corrections: Corrections): void {
    this.reading = response;
    this.selected = null;
    this.corrections = corrections;
  }

  setCorrections(next: Corrections): void {
    this.corrections = next;
    this.#onCorrect(next);
  }

  resetCorrections(): void {
    this.setCorrections(NO_CORRECTIONS);
  }

  revertCorrection(square: SquareKey): void {
    this.setCorrections(withCorrection(this.corrections, square, null));
  }

  assign(cls: ClassName): void {
    if (this.selected === null) return;
    const scores = scoresAt(this.dist, this.selected.r, this.selected.c);
    if (scores === null) return;
    const square = squareKey(this.selected.r, this.selected.c);
    this.setCorrections(withCorrection(this.corrections, square, assignmentFor(scores, cls)));
  }

  /** Opening a square never changes it; clicking it again toggles through its top-3. */
  selectSquare(r: number, c: number): void {
    if (this.dist === null) return;
    this.selected = { r: clampIndex(r), c: clampIndex(c) };
  }

  deselect(): void {
    this.selected = null;
  }

  pickSquare(r: number, c: number): void {
    if (this.selected === null || this.selected.r !== r || this.selected.c !== c) {
      this.selectSquare(r, c);
      return;
    }
    const scores = scoresAt(this.dist, r, c);
    if (scores === null) return;
    const square = squareKey(r, c);
    const target = cycleTarget(scores, this.corrections.get(square));
    this.setCorrections(withCorrection(this.corrections, square, target));
  }

  stepSquare(delta: number): void {
    if (this.selected === null) return;
    const index = (this.selected.r * GRID + this.selected.c + delta + 64) % 64;
    this.selectSquare(Math.floor(index / GRID), index % GRID);
  }

  nextLowConfidence(): void {
    const low = weakestSquares(this.dist);
    if (low.length === 0) return;
    const current = this.selected === null ? -1 : this.selected.r * GRID + this.selected.c;
    const next = low.find((entry) => entry.r * GRID + entry.c > current) ?? low[0];
    if (next !== undefined) this.selectSquare(next.r, next.c);
  }
}
