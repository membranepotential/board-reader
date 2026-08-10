/**
 * The projective maths the snapshot page rectifies with.
 *
 * A chess board is a plane, so four labelled corners are enough to warp the
 * playing surface to a square — no libraries, one 8-unknown DLT system, solved by
 * Gaussian elimination with partial pivoting.
 *
 * This is a **preview** rectification only. The node re-derives its own geometry
 * from the same corners and reads whole pieces off the ORIGINAL photo: a warped
 * square decapitates the piece standing on it, which is exactly what hurt
 * real-photo accuracy. What the operator sees here is a check that the corners
 * are on the board, not the input recognition will use.
 *
 * (`geometry/tracker.ts` carries its own, separate homography solver. It is not a
 * duplicate to be merged away: that one is normalised, least-squares over dozens
 * of RANSAC correspondences, and its numerics are pinned by ported checks. This
 * one is the exact-4-point DLT the snapshot page has always used.)
 */

import type { CornerQuad, Point } from '../protocol';

/** Row-major 3×3, `h33` last. */
export type Matrix3 = readonly [
  number, number, number,
  number, number, number,
  number, number, number
];

/**
 * The homography taking `from` to `to`, from four exact correspondences.
 *
 * `h33` is fixed to 1, which only fails for a homography sending the source origin
 * to infinity — no camera looking at a board in front of it produces one. Returns
 * `null` when the system is singular (three collinear corners, a repeated point),
 * rather than a matrix full of `NaN` that would silently paint an empty preview.
 */
export function homography(from: CornerQuad, to: CornerQuad): Matrix3 | null {
  const a: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = from[i]!;
    const [u, v] = to[i]!;
    a.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -x * v, -y * v]);
    b.push(v);
  }
  // Indices below are `!`-asserted: every one is a loop counter bounded by the 8×8
  // system built just above, so a bounds check here would only obscure the algebra.
  for (let c = 0; c < 8; c++) {
    let pivot = c;
    for (let r = c + 1; r < 8; r++) {
      if (Math.abs(a[r]![c]!) > Math.abs(a[pivot]![c]!)) pivot = r;
    }
    [a[c], a[pivot]] = [a[pivot]!, a[c]!];
    [b[c], b[pivot]] = [b[pivot]!, b[c]!];
    const lead = a[c]![c]!;
    if (Math.abs(lead) < 1e-12) return null;
    for (let r = 0; r < 8; r++) {
      if (r === c || a[r]![c] === 0) continue;
      const k = a[r]![c]! / lead;
      for (let j = c; j < 8; j++) a[r]![j] = a[r]![j]! - k * a[c]![j]!;
      b[r] = b[r]! - k * b[c]!;
    }
  }
  const h = b.map((value, i) => value / a[i]![i]!);
  if (!h.every(Number.isFinite)) return null;
  return [h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!, 1];
}

/** Push a point through `h`. `null` when it maps onto the line at infinity. */
export function applyMatrix3(h: Matrix3, x: number, y: number): Point | null {
  const w = h[6] * x + h[7] * y + h[8];
  if (Math.abs(w) < 1e-12) return null;
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

// ---- what the quad says about the camera -------------------------------------

export interface QuadGeometry {
  /** a8→h8, in source pixels. */
  readonly top: number;
  /** a1→h1. */
  readonly bottom: number;
  /** a8→a1. */
  readonly left: number;
  /** h8→h1. */
  readonly right: number;
  /** Mean edge width and height. */
  readonly w: number;
  readonly h: number;
  /**
   * Short axis / long axis, in (0, 1]. A calibration-free proxy for camera
   * elevation: ~1 is top-down, small values mean a nearly board-level camera where
   * pieces occlude the rank behind them.
   */
  readonly aspect: number;
}

const distance = (p: Point, q: Point): number => Math.hypot(p[0] - q[0], p[1] - q[1]);

/**
 * Mirror of the node's `boarddetect.quad_aspect` (and of the snapshot page's own
 * `quadGeometry`), so the page's verdict and the node's `very_low_angle` warning
 * can never disagree. `corners` are in canonical a8,h8,h1,a1 order.
 */
export function quadGeometry(corners: CornerQuad): QuadGeometry {
  const [a8, h8, h1, a1] = corners;
  const top = distance(a8, h8);
  const bottom = distance(a1, h1);
  const left = distance(a8, a1);
  const right = distance(h8, h1);
  const w = (top + bottom) / 2;
  const h = (left + right) / 2;
  const aspect = w > 0 && h > 0 ? Math.min(h / w, w / h) : 1;
  return { top, bottom, left, right, w, h, aspect };
}

// ---- the rectified preview ---------------------------------------------------

export interface RgbaImage {
  readonly data: Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
}

/**
 * Warp the quad to a `size`×`size` top-down view.
 *
 * The homography is solved the other way round — output pixel → source pixel — so
 * every destination pixel is filled exactly once and no gaps appear; sampling is
 * nearest-neighbour, which is all a check-the-corners preview needs. Pixels whose
 * pre-image falls outside the photo stay black and opaque rather than transparent,
 * so the operator sees the board's edge run off the picture instead of a hole.
 *
 * Returns `null` when the corners do not define a homography at all.
 */
export function rectifyQuad(
  image: RgbaImage,
  corners: CornerQuad,
  size: number
): { readonly data: Uint8ClampedArray; readonly size: number } | null {
  const output: CornerQuad = [
    [0, 0],
    [size, 0],
    [size, size],
    [0, size]
  ];
  const h = homography(output, corners);
  if (h === null) return null;
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const w = h[6] * x + h[7] * y + h[8];
      const u = Math.round((h[0] * x + h[1] * y + h[2]) / w);
      const v = Math.round((h[3] * x + h[4] * y + h[5]) / w);
      const di = (y * size + x) * 4;
      out[di + 3] = 255;
      if (u >= 0 && v >= 0 && u < image.width && v < image.height) {
        const si = (v * image.width + u) * 4;
        out[di] = image.data[si]!;
        out[di + 1] = image.data[si + 1]!;
        out[di + 2] = image.data[si + 2]!;
      }
    }
  }
  return { data: out, size };
}

/**
 * Mean luma of an RGBA image, subsampled — the page's own mirror of the node's
 * low-light check (`server.LOW_LIGHT_MEAN`). Stepping rather than reading every
 * pixel keeps a 12 MP phone photo instant, and a mean does not need every sample.
 */
export function meanLuma(image: RgbaImage): number {
  const step = Math.max(1, Math.round(Math.max(image.width, image.height) / 240));
  let sum = 0;
  let n = 0;
  for (let y = 0; y < image.height; y += step) {
    for (let x = 0; x < image.width; x += step) {
      const i = (y * image.width + x) * 4;
      sum += (image.data[i]! + image.data[i + 1]! + image.data[i + 2]!) / 3;
      n++;
    }
  }
  return n ? sum / n : 0;
}
