/**
 * Optional per-board display rotation on the live page (ccbf-r5 remainder,
 * user-approved): a purely cosmetic control so a board's on-screen picture can
 * be turned to match whichever side of the physical board the operator is
 * standing at when they look at the screen next to it.
 *
 * This is deliberately NOT chessground's `orientation` prop. `orientation` is
 * a white/black flip through the board's centre — chessground computes file
 * and rank layout for exactly those two sides, so it can never produce a
 * 90°-turned picture, and `Board.svelte` dropped that prop already (unused,
 * see its doc comment) rather than half-support a rotation it cannot do.
 * Turning the whole rendered card (grid, coordinates, pieces and the
 * confidence/change rings alike) by a multiple of 90° is instead a CSS
 * `transform: rotate()` on the wrapper in `BoardCard.svelte` — the placement
 * itself, always calibrated to white's view (DESIGN §4.5, `a8` first), is
 * never touched. At 90°/270° the coordinate labels and piece sprites turn
 * sideways with the rest of the picture; that is the point, not a bug — it is
 * the same thing that happens when you turn a printed board diagram in your
 * hands.
 */

export type BoardRotation = 0 | 90 | 180 | 270;

const ROTATIONS: readonly BoardRotation[] = [0, 90, 180, 270];

function isBoardRotation(value: number): value is BoardRotation {
  return (ROTATIONS as readonly number[]).includes(value);
}

/** The next step in the cycle, wrapping 270 -> 0. */
export function nextRotation(rotation: BoardRotation): BoardRotation {
  const i = ROTATIONS.indexOf(rotation);
  const found = ROTATIONS[(i + 1) % ROTATIONS.length];
  // ROTATIONS is a fixed non-empty tuple; the modulo index is always in range.
  return found as BoardRotation;
}

/** Where one board's chosen rotation is persisted — keyed per board id. */
export function rotationKey(boardId: string): string {
  return `board-reader.live.rotation.${boardId}`;
}

/** Parses a stored value back into a `BoardRotation`, defaulting to unrotated. */
export function parseStoredRotation(raw: string | null): BoardRotation {
  if (raw === null) return 0;
  const n = Number(raw);
  return isBoardRotation(n) ? n : 0;
}
