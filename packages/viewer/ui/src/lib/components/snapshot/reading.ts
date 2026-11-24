/**
 * What one read means, as pure functions over the parsed `/estimate` reply.
 *
 * The rule this module exists to keep: **the node's own read is never mutated.** The
 * baseline is `prior.placement ?? event.placement` plus `square_dist`, and every hand
 * correction is a separate layer on top of it — so "reset corrections" restores exactly
 * what the node served, a correction can always be described as `read → assigned`, and
 * the ground-truth label is a derived value rather than a second copy of the board.
 *
 * Everything here is a function of its arguments, which is why it can be tested without
 * a browser and why the page's render path holds no derived state at all.
 */

import { fenToGrid, gridToFen, squareKey, type BoardGrid, type Cell } from '$lib/board/fen';
import {
  CLASSES,
  GRID,
  LOW_CONFIDENCE,
  isPieceSymbol,
  toGrid8,
  type ClassName,
  type ClassScores,
  type EstimateResponse,
  type SquareDist,
  type SquareKey
} from '$lib/protocol';

/** Operator-facing names for the 13 classes — the wording every panel and list uses. */
export const CLASS_NAMES: Readonly<Record<ClassName, string>> = {
  empty: 'empty',
  P: 'white pawn',
  N: 'white knight',
  B: 'white bishop',
  R: 'white rook',
  Q: 'white queen',
  K: 'white king',
  p: 'black pawn',
  n: 'black knight',
  b: 'black bishop',
  r: 'black rook',
  q: 'black queen',
  k: 'black king'
};

/**
 * Row 1 = White, row 2 = Black, same piece type in the same column — the board's own
 * colour symmetry, so a miscoloured read is a one-glance vertical scan, not a hunt.
 */
export const PANEL_TYPES = ['K', 'Q', 'R', 'B', 'N', 'P'] as const;
export type PanelType = (typeof PANEL_TYPES)[number];

/** chessground's own role names, for the `<piece>` elements in the switcher. */
export const ROLE_NAMES: Readonly<Record<PanelType, string>> = {
  K: 'king',
  Q: 'queen',
  R: 'rook',
  B: 'bishop',
  N: 'knight',
  P: 'pawn'
};

/** Hand corrections: square → the class the operator says is really there. */
export type Corrections = ReadonlyMap<SquareKey, ClassName>;

export const NO_CORRECTIONS: Corrections = new Map();

/** `''` is how a grid spells an empty square; `'empty'` is how a distribution does. */
export function cellToClass(cell: Cell): ClassName {
  return cell === '' ? 'empty' : cell;
}

export function classToCell(cls: ClassName): Cell {
  return cls === 'empty' ? '' : cls;
}

/** Grid indices of an algebraic square — the inverse of `squareKey`. */
export function indexOfSquare(square: SquareKey): { readonly r: number; readonly c: number } {
  return { r: GRID - Number(square[1]), c: square.charCodeAt(0) - 'a'.charCodeAt(0) };
}

/** Index of a class in the canonical `square_dist` order. */
export function classIndex(cls: ClassName): number {
  return CLASSES.indexOf(cls);
}

/** A single character typed on the keyboard, as a class — or null if it is not one. */
export function classOfKey(key: string): ClassName | null {
  if (key === '.' || key === '0' || key === 'Delete' || key === 'Backspace') return 'empty';
  return isPieceSymbol(key) ? key : null;
}

// ---- the board as shown ------------------------------------------------------

/** The node's read with the corrections layered on top — the board actually on screen. */
export function correctedGrid(base: BoardGrid, corrections: Corrections): BoardGrid {
  if (corrections.size === 0) return base;
  return toGrid8(
    base.map((rank, r) =>
      toGrid8(
        rank.map((cell, c) => {
          const fix = corrections.get(squareKey(r, c));
          return fix === undefined ? cell : classToCell(fix);
        })
      )
    )
  );
}

export function placementOfGrid(grid: BoardGrid): string {
  return gridToFen(grid);
}

/** What is shown at `(r, c)` right now: the correction if there is one, else the read. */
export function shownClassAt(base: BoardGrid, corrections: Corrections, r: number, c: number): ClassName {
  const fix = corrections.get(squareKey(r, c));
  if (fix !== undefined) return fix;
  const row = base[r];
  return row === undefined ? 'empty' : cellToClass(row[c] ?? '');
}

// ---- reading one square's distribution ---------------------------------------

/** A square's scores, or null when the read carried no distribution at all. */
export function scoresAt(dist: SquareDist | null, r: number, c: number): ClassScores | null {
  return dist?.[r]?.[c] ?? null;
}

export function argmaxClass(scores: ClassScores): ClassName {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if ((scores[i] as number) > (scores[best] as number)) best = i;
  }
  return CLASSES[best] as ClassName;
}

export function confidenceOf(scores: ClassScores | null): number {
  // A read with no distribution is not a read that doubts every square: absent reads
  // as fully confident, exactly as `squareConfidence` does for the live page.
  return scores === null ? 1 : Math.max(...scores);
}

export interface RankedClass {
  readonly cls: ClassName;
  readonly p: number;
}

/** The full 13-class distribution for a square, most likely first. */
export function rankedClasses(scores: ClassScores): readonly RankedClass[] {
  return CLASSES.map((cls, i) => ({ cls, p: scores[i] as number })).sort((a, b) => b.p - a.p);
}

/**
 * Clicking the already-selected square advances it to its next-most-probable reading:
 * argmax → 2nd → 3rd → back to argmax. Fixing a misread piece is usually one extra
 * click without ever opening the grid.
 *
 * Returns the class to assign, or `null` for "no correction" — a correction picked from
 * the grid that is not in the top 3 is still a valid state to toggle from, and the next
 * click resets it to the argmax.
 */
export function cycleTarget(scores: ClassScores, current: ClassName | undefined): ClassName | null {
  const top3 = rankedClasses(scores).slice(0, 3).map((r) => r.cls);
  const head = top3[0] as ClassName;
  const at = top3.indexOf(current ?? head);
  const next = top3[(at + 1) % top3.length] as ClassName;
  return next === head ? null : next;
}

/**
 * Assigning a class from the grid or the keyboard: picking the node's own read is "no
 * correction", so the map only ever holds genuine disagreements with the node.
 */
export function assignmentFor(scores: ClassScores, cls: ClassName): ClassName | null {
  return cls === argmaxClass(scores) ? null : cls;
}

/** A correction set with one square restated (`null` clears it). */
export function withCorrection(
  corrections: Corrections,
  square: SquareKey,
  cls: ClassName | null
): Corrections {
  const next = new Map(corrections);
  if (cls === null) next.delete(square);
  else next.set(square, cls);
  return next;
}

// ---- whole-board statistics --------------------------------------------------

export interface SquareRead {
  readonly r: number;
  readonly c: number;
  readonly square: SquareKey;
  readonly p: number;
}

/** Per-square confidences in row-major order, rank 8 first. */
export function squareReads(dist: SquareDist | null): readonly SquareRead[] {
  const out: SquareRead[] = [];
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      out.push({ r, c, square: squareKey(r, c), p: confidenceOf(scoresAt(dist, r, c)) });
    }
  }
  return out;
}

/** The eight weakest squares, weakest first — the chips, and the "next low" walk. */
export function weakestSquares(dist: SquareDist | null, count = 8): readonly SquareRead[] {
  return [...squareReads(dist)].sort((a, b) => a.p - b.p).slice(0, count);
}

export interface ConfidenceSummary {
  readonly mean: number;
  readonly min: number;
  /** How many squares fall under `LOW_CONFIDENCE`. */
  readonly low: number;
}

export function confidenceSummary(dist: SquareDist | null): ConfidenceSummary {
  const ps = squareReads(dist).map((s) => s.p);
  return {
    mean: ps.reduce((a, b) => a + b, 0) / ps.length,
    min: Math.min(...ps),
    low: ps.filter((p) => p < LOW_CONFIDENCE).length
  };
}

/**
 * A material census is the cheapest sanity check on a read: two kings, at most eight
 * pawns a side. The CV layer never applies chess rules (DESIGN D1) — this is the
 * consumer noticing, not the estimator constraining.
 */
export interface MaterialCensus {
  readonly white: string;
  readonly black: string;
  readonly occupied: number;
  /** Counts no chess position can have. Empty means the census is legal. */
  readonly implausible: readonly string[];
}

export function materialCensus(grid: BoardGrid): MaterialCensus {
  const counts = new Map<string, number>();
  for (const rank of grid) {
    for (const cell of rank) {
      if (cell !== '') counts.set(cell, (counts.get(cell) ?? 0) + 1);
    }
  }
  const count = (ch: string): number => counts.get(ch) ?? 0;
  const line = (set: readonly string[]): string => set.map((ch) => `${ch}${count(ch)}`).join(' ');
  const implausible: string[] = [];
  if (count('K') !== 1) implausible.push(`white kings: ${count('K')}`);
  if (count('k') !== 1) implausible.push(`black kings: ${count('k')}`);
  if (count('P') > 8) implausible.push(`white pawns: ${count('P')}`);
  if (count('p') > 8) implausible.push(`black pawns: ${count('p')}`);
  return {
    white: line(['K', 'Q', 'R', 'B', 'N', 'P']),
    black: line(['k', 'q', 'r', 'b', 'n', 'p']),
    occupied: [...counts.values()].reduce((a, b) => a + b, 0),
    implausible
  };
}

// ---- the FEN this page hands to a human --------------------------------------

export type SideToMove = 'w' | 'b';

/**
 * Castling rights are a downstream convenience, derived here in the consumer: grant a
 * right only if that king and rook still stand on their start squares. The estimator
 * never says anything about them — it reports placements, not game state (DESIGN D1).
 */
export function castlingRights(grid: BoardGrid): string {
  const at = (r: number, c: number): Cell => grid[r]?.[c] ?? '';
  let rights = '';
  if (at(7, 4) === 'K') {
    if (at(7, 7) === 'R') rights += 'K';
    if (at(7, 0) === 'R') rights += 'Q';
  }
  if (at(0, 4) === 'k') {
    if (at(0, 7) === 'r') rights += 'k';
    if (at(0, 0) === 'r') rights += 'q';
  }
  return rights || '-';
}

export function fullFen(placement: string, stm: SideToMove): string {
  return `${placement} ${stm} ${castlingRights(fenToGrid(placement))} - 0 1`;
}

/** Lichess takes a FEN with underscores for spaces in the path. */
export function lichessUrls(placement: string, stm: SideToMove): {
  readonly analysis: string;
  readonly editor: string;
} {
  const fen = fullFen(placement, stm).replace(/ /g, '_');
  return {
    analysis: `https://lichess.org/analysis/standard/${fen}`,
    editor: `https://lichess.org/editor/${fen}`
  };
}

// ---- ground truth ------------------------------------------------------------

/**
 * Whether the board on screen can be recorded as the true position for its photo.
 *
 * The correction panel is the only place a human ever states what a board REALLY holds,
 * and until this existed that answer died with the page. Keeping it is what makes an
 * uncontaminated eval set possible at all. Confirmed labels are scoring data only — the
 * node never fits on them.
 *
 *  - `no-retention` — the node kept no photo (`BR_SAVE_UPLOADS` unset), so a labelling
 *    would have no image to belong to. Shown and explained, never hidden: an absent
 *    button read as a missing feature.
 *  - `unsaved` — nothing recorded for this photo yet. Confirming with zero corrections is
 *    just as much a label as fixing four squares: "the model got this one exactly right"
 *    is the observation the perfect-board rate is made of.
 *  - `saved` — what is on screen is what was recorded.
 *  - `stale` — a square was edited after saving, so the recorded label no longer
 *    describes the board and this is an update.
 */
export type LabelState = 'no-retention' | 'unsaved' | 'saved' | 'stale';

export function labelState(
  upload: string | null,
  shownPlacement: string,
  savedPlacement: string | null
): LabelState {
  if (upload === null) return 'no-retention';
  if (savedPlacement === null) return 'unsaved';
  return savedPlacement === shownPlacement ? 'saved' : 'stale';
}

// ---- presentation helpers ----------------------------------------------------

export function fmt(value: number, digits = 2): string {
  return value.toFixed(digits);
}

export function pct(p: number): string {
  return p * 100 < 1 ? '<1%' : `${Math.round(p * 100)}%`;
}

/** The same red → yellow → green ramp the node draws on its debug images. */
export function heatColor(p: number): string {
  const [from, to, t] =
    p >= 0.8
      ? ([[255, 200, 60], [130, 210, 130], (p - 0.8) / 0.2] as const)
      : p >= 0.5
        ? ([[226, 85, 85], [255, 200, 60], (p - 0.5) / 0.3] as const)
        : ([[150, 45, 45], [226, 85, 85], Math.max(0, p / 0.5)] as const);
  const mix = from.map((v, i) => Math.round(v + ((to[i] as number) - v) * Math.min(1, t)));
  return `rgb(${mix.join(',')})`;
}

/**
 * The read as JSON, for the raw-event block and the "copy raw event" button.
 *
 * This is the *parsed* event re-serialised, not the bytes off the wire: everything the
 * page renders comes through `parseEstimateResponse`, so showing anything else would be
 * showing a different value than the one the stages describe. `square_dist` is
 * replaced by a note — 8×8×13 floats are what the confidence map is for.
 */
export function rawEventJson(response: EstimateResponse): string {
  const body = {
    board_id: response.event.board_id,
    seq: response.event.seq,
    utc: response.event.utc,
    placement: response.event.placement,
    confidence: response.event.confidence,
    stability: response.event.stability,
    status: response.event.status,
    orientation: response.event.orientation,
    clock: response.event.clock,
    square_dist: '[8 x 8 x 13 floats — see the confidence map]',
    prior: response.prior,
    upload: response.upload,
    warnings: response.warnings
  };
  return JSON.stringify(body, null, 2);
}

/**
 * The baseline board a reading establishes.
 *
 * `event.placement` is the raw argmax and stays that way in the event JSON, while
 * `prior` carries the nearest board that is actually possible — when the node decoded
 * one, THAT is the answer, and corrections layer on top of it.
 */
export function baselinePlacement(response: EstimateResponse): string {
  return response.prior?.placement ?? response.event.placement;
}

export function baselineGrid(response: EstimateResponse): BoardGrid {
  return fenToGrid(baselinePlacement(response));
}
