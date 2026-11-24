/**
 * The crop rectangle on the control page: dragging it and keeping it valid.
 *
 * A `CropRect` is in FRACTIONS of the full camera frame (`board_reader_core.control`),
 * so the editor works in fractions too: a pointer delta is divided by the drawn
 * picture's size once, at the caller, and everything here is resolution-free.
 *
 * Every rectangle this module returns passes the node's own `CropRect` checks
 * (`x, y >= 0`, `x + w <= 1`, `y + h <= 1`, both sides `>= MIN_CROP_SIDE`), so a
 * Save can never be refused for a rectangle the operator was allowed to draw.
 */

import type { CropRect } from '$lib/protocol';

/** `board_reader_core.control.MIN_CROP_SIDE`: below this it is a slip, not a board. */
export const MIN_CROP_SIDE = 0.05;

/** What the editor shows when the config sends the whole frame (`crop: null`). */
export const WHOLE_FRAME: CropRect = { x: 0, y: 0, w: 1, h: 1 };

/** The body moves the rectangle; a corner moves itself, the opposite corner stays. */
export type Handle = 'move' | 'nw' | 'ne' | 'se' | 'sw';
export const CORNERS = ['nw', 'ne', 'se', 'sw'] as const satisfies readonly Handle[];

/** A pointer movement, in fractions of the frame. */
export interface Delta {
  readonly dx: number;
  readonly dy: number;
}

/**
 * Edges are snapped to whole units of 1/10 000 of the frame.
 *
 * Integer units keep the arithmetic exact: `x + w` is `(X0 + (X1 - X0)) / UNITS`,
 * which never rounds past 1, whereas a float edge `x1 = 0.7 + 0.3000000001` can
 * land a hair outside the frame and be refused by the node.
 */
const UNITS = 10_000;
const MIN_UNITS = Math.round(MIN_CROP_SIDE * UNITS);

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/** Edges in fractions -> a valid rectangle, snapped to `UNITS`. */
function fromEdges(x0: number, y0: number, x1: number, y1: number): CropRect {
  const [left, right] = span(x0, x1);
  const [top, bottom] = span(y0, y1);
  return { x: left / UNITS, y: top / UNITS, w: (right - left) / UNITS, h: (bottom - top) / UNITS };
}

/** One axis in whole units, inside the frame and at least `MIN_UNITS` long. */
function span(lo: number, hi: number): readonly [number, number] {
  const start = clamp(Math.round(lo * UNITS), 0, UNITS - MIN_UNITS);
  const end = clamp(Math.round(hi * UNITS), start + MIN_UNITS, UNITS);
  return [start, end];
}

/** `start` after dragging `handle` by `delta`, kept inside the frame and above the minimum. */
export function dragRect(start: CropRect, handle: Handle, delta: Delta): CropRect {
  const x0 = start.x;
  const y0 = start.y;
  const x1 = start.x + start.w;
  const y1 = start.y + start.h;
  if (handle === 'move') {
    // Clamp the movement, not the edges: a rectangle pushed into a wall stops
    // there with its size intact instead of being squashed against it.
    const dx = clamp(delta.dx, -x0, 1 - x1);
    const dy = clamp(delta.dy, -y0, 1 - y1);
    return fromEdges(x0 + dx, y0 + dy, x1 + dx, y1 + dy);
  }
  const west = handle === 'nw' || handle === 'sw';
  const north = handle === 'nw' || handle === 'ne';
  return fromEdges(
    west ? clamp(x0 + delta.dx, 0, x1 - MIN_CROP_SIDE) : x0,
    north ? clamp(y0 + delta.dy, 0, y1 - MIN_CROP_SIDE) : y0,
    west ? x1 : clamp(x1 + delta.dx, x0 + MIN_CROP_SIDE, 1),
    north ? y1 : clamp(y1 + delta.dy, y0 + MIN_CROP_SIDE, 1)
  );
}

/** Whether the node would accept `rect` — the same four rules as `CropRect.__post_init__`. */
export function isValidCrop(rect: CropRect): boolean {
  const { x, y, w, h } = rect;
  return (
    [x, y, w, h].every(Number.isFinite) &&
    x >= 0 &&
    y >= 0 &&
    w >= MIN_CROP_SIDE &&
    h >= MIN_CROP_SIDE &&
    x + w <= 1 &&
    y + h <= 1
  );
}

export function sameCrop(a: CropRect | null, b: CropRect | null): boolean {
  if (a === null || b === null) return a === b;
  return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}
