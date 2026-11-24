/**
 * The four corners: how they are labelled, what is cropped around them, and when
 * they cannot possibly be a board.
 *
 * **Labelling is the operator's, never geometry's** (DESIGN decision #15). A
 * detector returns an *unlabelled* quad in geometric TL,TR,BR,BL order; which
 * physical corner is a8 cannot be recovered from pixels. So the quad and its
 * labelling are two separate pieces of state — four points in storage order, plus
 * a rotation — and every page that has ever shown corners has carried exactly
 * that pair. A rotation is a relabelling; it never moves a point.
 *
 * The validity checks mirror `board_reader_edge.quadcheck` (which guards the
 * edge's `--corners`) and `board_reader_core.boarddetect.quad_aspect`. They are
 * mirrored rather than approximated because the failure they catch is silent: a
 * transposed digit or a swapped pair of points produces confidently-wrong geometry
 * for as long as the run lasts, with no error anywhere.
 */

import { LOW_ANGLE_ASPECT, type CornerQuad, type Point } from '../protocol';
import { quadGeometry } from './homography';

/** = TL,TR,BR,BL of the canonical crop, and the order every endpoint takes. */
export const CORNER_ORDER = ['a8', 'h8', 'h1', 'a1'] as const;
export type CornerLabel = (typeof CORNER_ORDER)[number];

/** Quarter-turns of the labelling. Four values, so a fifth cannot be written down. */
export type LabelRotation = 0 | 1 | 2 | 3;

/** Any integer (or a negative one) folded into the four legal rotations. */
export function asRotation(turns: number): LabelRotation {
  switch ((((Math.round(turns) % 4) + 4) % 4) as number) {
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return 3;
    default:
      return 0;
  }
}

/** What the corner at storage index `i` is called under `rot`. */
export function labelOf(i: number, rot: LabelRotation): CornerLabel {
  const label = CORNER_ORDER[(((i + rot) % 4) + 4) % 4];
  if (label === undefined) throw new RangeError(`corner ${i} is not one of four`);
  return label;
}

/**
 * The quad in canonical a8,h8,h1,a1 order, whatever the storage order and rotation
 * — the inverse of `labelOf`, and the only form that ever goes on the wire.
 */
export function orderedCorners(points: CornerQuad, rot: LabelRotation): CornerQuad {
  const at = (k: number): Point => points[(((k - rot) % 4) + 4) % 4] as Point;
  return [at(0), at(1), at(2), at(3)];
}

/** Pressing "rotate labels": the next quarter-turn, points untouched. */
export function nextRotation(rot: LabelRotation): LabelRotation {
  return asRotation(rot + 1);
}

/**
 * Absorb a rotation the node applied to a reading into this page's own labelling.
 *
 * A read posted with `?autorotate=1` can come back turned by `k` quarter-turns.
 * Left alone the page then holds TWO orientations at once — the board showing the
 * rotated position while the handles, the corner table and the rectified preview
 * still describe the pre-rotation frame. Reindexing the 8×8 lattice by `rot90(·,k)`
 * is exactly a `+k` cyclic shift of the ordered corners, and since
 * `orderedCorners()[i] = points[(i - rot + 4) % 4]`, that is `rot -= k`.
 */
export function absorbRotation(rot: LabelRotation, quarterTurns: number): LabelRotation {
  return asRotation(rot - quarterTurns);
}

/** `?autorotate` reports degrees; the page thinks in quarter-turns. */
export function quarterTurnsOf(degrees: number): LabelRotation {
  return asRotation(degrees / 90);
}

// ---- the crop that goes on the wire ------------------------------------------

/** Margin on each side of the quad's bounding box, as a fraction of its size. */
export const CROP_MARGIN = 0.1;
/**
 * Extra above the top edge, on top of `CROP_MARGIN`.
 *
 * **Pieces are TALL.** A box drawn on the playing surface decapitates the back
 * rank at any real camera angle — the same reason recognition prefers tall boxes
 * over warped squares — and the top of the frame is where the pieces stand up out
 * of the board. This headroom is what keeps them in the picture.
 */
export const CROP_HEADROOM = 0.3;

export interface CropBox {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

export interface FrameSize {
  readonly width: number;
  readonly height: number;
}

/** The crop actually sent, clamped to the frame. Coordinates are frame pixels. */
export function cropBox(points: CornerQuad, frame: FrameSize): CropBox {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const left = Math.max(0, Math.floor(x0 - CROP_MARGIN * w));
  const top = Math.max(0, Math.floor(y0 - (CROP_MARGIN + CROP_HEADROOM) * h));
  const right = Math.min(frame.width, Math.ceil(x1 + CROP_MARGIN * w));
  const bottom = Math.min(frame.height, Math.ceil(y1 + CROP_MARGIN * h));
  return { x: left, y: top, w: Math.max(1, right - left), h: Math.max(1, bottom - top) };
}

/**
 * Rebase a quad into the pixels of a crop that has also been scaled.
 *
 * The node needs the corners in the image it actually receives and only there: it
 * never sees the full frame, so corners in camera coordinates would point outside
 * the image it holds. Compute this beside the draw that fixed the pixels, never
 * after an `await` — a handle dragged during encoding must not pair new corners
 * with old pixels.
 */
export function cornersInCrop(
  points: CornerQuad,
  box: CropBox,
  scaled: { readonly w: number; readonly h: number }
): CornerQuad {
  const sx = scaled.w / box.w;
  const sy = scaled.h / box.h;
  const round = (v: number) => Math.round(v * 10) / 10;
  const at = (p: Point): Point => [round((p[0] - box.x) * sx), round((p[1] - box.y) * sy)];
  return [at(points[0]), at(points[1]), at(points[2]), at(points[3])];
}

// ---- is this still a board? --------------------------------------------------

export type QuadDefectKind =
  | 'not_a_quad'
  | 'coincident_points'
  | 'self_intersecting'
  | 'not_convex'
  | 'degenerate'
  | 'out_of_frame';

/**
 * Why a set of points cannot be a board outline. `message` reads as a predicate of
 * the quad ("is not convex — …"), so a caller can prefix it with whatever named
 * the quad ("the board corners", "these handles").
 */
export interface QuadDefect {
  readonly kind: QuadDefectKind;
  readonly message: string;
}

/** Two corners closer than this are one tap entered twice, not a quad. */
const MIN_SEPARATION_PX = 1.0;
/** Below this the playing surface would put each square under 8×8 px. */
const MIN_AREA_PX = 64 * 64;
/**
 * Enclosed area as a fraction of the quad's own longest span, squared. A square
 * scores 0.5 and even a brutally foreshortened board (aspect 0.2) still scores
 * ~0.19, so below this the points are near-collinear: a sliver, not a board.
 */
const MIN_FILL = 0.05;
/**
 * How far outside the frame a corner may sit. Deliberately not zero: a board whose
 * near corner is cropped by the frame edge is a real field case, and extrapolating
 * it slightly off-frame is the right thing to do. What this catches — a quad
 * measured at a different resolution, a transposed digit — overshoots by far more.
 */
const FRAME_MARGIN = 0.1;

const cross = (o: Point, a: Point, b: Point): number =>
  (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

/** Do segments `ab` and `cd` properly cross? (Touching does not count.) */
function segmentsCross(a: Point, b: Point, c: Point, d: Point): boolean {
  const d1 = cross(c, d, a);
  const d2 = cross(c, d, b);
  const d3 = cross(a, b, c);
  const d4 = cross(a, b, d);
  return d1 > 0 !== d2 > 0 && d3 > 0 !== d4 > 0;
}

/** Shoelace area, unsigned — winding is the caller's problem, not ours. */
function area(quad: CornerQuad): number {
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i] as Point;
    const b = quad[(i + 1) % 4] as Point;
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(sum) / 2;
}

const point = (p: Point): string => `(${p[0]}, ${p[1]})`;

/**
 * Everything about a board quad that is knowable without a frame — the port of
 * `quadcheck.check_quad_shape`, including its ORDER, which is load-bearing: a
 * repeated point makes every later predicate meaningless; a bowtie's shoelace area
 * nearly cancels, so it must be caught before the area test calls it degenerate;
 * and four collinear points should fail as a sliver rather than as the "not convex"
 * they technically also are.
 *
 * Convexity is the check that earns its keep: the four points are a *walk around
 * the board* in a8,h8,h1,a1 order, so an ordering slip — classically entering them
 * as a8,h8,a1,h1 — folds the quad into a bowtie, which a sign-consistent turn test
 * refuses.
 */
export function checkQuadShape(quad: CornerQuad): QuadDefect | null {
  const flat = quad.flatMap((p) => [p[0], p[1]]);
  if (flat.length !== 8 || !flat.every(Number.isFinite)) {
    return { kind: 'not_a_quad', message: 'contains a value that is not a finite number' };
  }

  for (const [i, j] of [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]] as const) {
    const a = quad[i] as Point;
    const b = quad[j] as Point;
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) < MIN_SEPARATION_PX) {
      return {
        kind: 'coincident_points',
        message: `repeats a point: ${i} and ${j} are both ${point(a)}`
      };
    }
  }

  if (
    segmentsCross(quad[0], quad[1], quad[2], quad[3]) ||
    segmentsCross(quad[1], quad[2], quad[3], quad[0])
  ) {
    return {
      kind: 'self_intersecting',
      message:
        'crosses itself — the four points walk around the board in a8,h8,h1,a1 order; ' +
        'a bowtie usually means the last two were swapped'
    };
  }

  const enclosed = area(quad);
  if (enclosed < MIN_AREA_PX) {
    return {
      kind: 'degenerate',
      message:
        `encloses only ${enclosed.toFixed(0)} px^2 — far too small to be a board ` +
        `(64 squares need at least ${MIN_AREA_PX} px^2)`
    };
  }
  let span = 0;
  for (let i = 0; i < 4; i++) {
    for (let j = i + 1; j < 4; j++) {
      const a = quad[i] as Point;
      const b = quad[j] as Point;
      span = Math.max(span, Math.hypot(a[0] - b[0], a[1] - b[1]));
    }
  }
  if (enclosed < MIN_FILL * span * span) {
    return {
      kind: 'degenerate',
      message:
        `is a sliver: ${enclosed.toFixed(0)} px^2 across a ${span.toFixed(0)} px span, ` +
        'i.e. the points are nearly collinear'
    };
  }

  const turns = [0, 1, 2, 3].map((i) =>
    cross(quad[i] as Point, quad[(i + 1) % 4] as Point, quad[(i + 2) % 4] as Point)
  );
  if (!(turns.every((t) => t > 0) || turns.every((t) => t < 0))) {
    return {
      kind: 'not_convex',
      message:
        'is not convex — a board outline turns the same way at all four corners, ' +
        'so one of the coordinates is wrong'
    };
  }
  return null;
}

/**
 * The one check that needs a frame: are the corners actually on the picture?
 * Callers run it once the frame size is known — a camera only reveals it after the
 * first frame.
 */
export function checkQuadInFrame(quad: CornerQuad, frame: FrameSize): QuadDefect | null {
  const mx = FRAME_MARGIN * frame.width;
  const my = FRAME_MARGIN * frame.height;
  const outside = quad
    .map((p, i) => ({ p, i }))
    .filter(
      ({ p }) =>
        !(-mx <= p[0] && p[0] <= frame.width + mx && -my <= p[1] && p[1] <= frame.height + my)
    );
  if (outside.length === 0) return null;
  const where = outside.map(({ p, i }) => `${i}=${point(p)}`).join(', ');
  return {
    kind: 'out_of_frame',
    message:
      `lies outside the ${frame.width}x${frame.height} source frame: ${where} — ` +
      'corners measured on a different resolution are the usual cause'
  };
}

/**
 * Camera-too-flat warning, or `null` when the angle is usable.
 *
 * **Warn, never block.** The aspect is a proxy for camera elevation, the operator
 * may know exactly why theirs is unusual, and blocking on a proxy is worse than an
 * honest heads-up — the node makes the same call, tagging an estimate
 * `very_low_angle` rather than refusing it. The point of saying it here is
 * *timing*: before the shot, not after a bad read.
 */
export function lowAngleWarning(quad: CornerQuad): string | null {
  const { aspect } = quadGeometry(quad);
  if (aspect >= LOW_ANGLE_ASPECT) return null;
  return (
    `the corners project to a board aspect of ${aspect.toFixed(2)}, below ` +
    `${LOW_ANGLE_ASPECT.toFixed(2)}: the camera is near board level, where pieces occlude ` +
    'the rank behind them and recognition collapses. Raise the camera if you can ' +
    '(docs/operating-envelope.md).'
  );
}
