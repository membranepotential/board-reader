/* ---------------------------------------------------------------------------
   Planar board tracking for the browser edge — DESIGN §4.4.1(a), "bump / drift".

   A chess board is a PLANE, so every camera movement that keeps it in frame maps
   the old view to the new one by a single homography. That is the whole algorithm:
   sprinkle points over the board, follow them with pyramidal Lucas-Kanade, fit the
   homography they agree on, and push the four corners through it.

   Why this and not "detect the board again every frame": DESIGN decision #15 says
   geometry can never tell which corner is a8 — a fresh detection returns an
   unlabelled quad, so re-detecting *loses the operator's orientation* and turns a
   bump into a §4.4.1(b) reposition. Tracking carries the labelling along with the
   corners, which is what lets the phone be walked all the way around the board
   (the calibration orbit) without ever re-declaring which corner is which.

   No DOM in this module, deliberately: it takes a grayscale Float32Array and
   returns numbers, so `node` can test the mathematics directly
   (packages/viewer/tests/test_tracker_js.py).

   Coordinates: the tracker speaks NORMALISED frame coordinates ([0,1] across the
   frame) at its boundary, so the caller may change capture resolution — a
   different phone, a different lens — without rebasing anything.
   --------------------------------------------------------------------------- */

// ---- image pyramid -----------------------------------------------------------

/** RGBA bytes -> gray in [0,1], using cv2's BGR2GRAY weights (as the edge does). */
export function toGray(rgba, out) {
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]) / 255;
  }
  return out;
}

/** Central differences. Computed once per level per frame, not once per point. */
function gradients(level) {
  const { data, w, h } = level;
  const ix = new Float32Array(w * h);
  const iy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      ix[i] = 0.5 * (data[x < w - 1 ? i + 1 : i] - data[x > 0 ? i - 1 : i]);
      iy[i] = 0.5 * (data[y < h - 1 ? i + w : i] - data[y > 0 ? i - w : i]);
    }
  }
  level.ix = ix;
  level.iy = iy;
  return level;
}

/** Half-size by 2x2 box average — the decimation and its anti-alias in one pass. */
export function halve({ data, w, h }) {
  const w2 = Math.max(1, w >> 1);
  const h2 = Math.max(1, h >> 1);
  const out = new Float32Array(w2 * h2);
  for (let y = 0; y < h2; y++) {
    const r0 = Math.min(2 * y, h - 1) * w;
    const r1 = Math.min(2 * y + 1, h - 1) * w;
    for (let x = 0; x < w2; x++) {
      const c0 = Math.min(2 * x, w - 1);
      const c1 = Math.min(2 * x + 1, w - 1);
      out[y * w2 + x] = 0.25 * (data[r0 + c0] + data[r0 + c1] + data[r1 + c0] + data[r1 + c1]);
    }
  }
  return { data: out, w: w2, h: h2 };
}

/**
 * Level 0 is full size; each further level halves it. Coarse levels are what let a fast
 * pan be tracked at all: LK alone only sees motion inside its window.
 *
 * Level 0 is COPIED, and that copy is load-bearing. A caller capturing video reuses one
 * buffer per frame, so a pyramid that merely referenced it would, one frame later, hold
 * the *current* pixels as its "previous" image — every point would then measure its
 * displacement against itself, converge on zero, and report a flawless fit while the
 * quad sat perfectly still. It fails as a silent wrong answer, not as an error.
 */
export function buildPyramid(gray, w, h, levels = 3) {
  const pyramid = [gradients({ data: gray.slice(), w, h })];
  for (let i = 1; i < levels; i++) pyramid.push(gradients(halve(pyramid[i - 1])));
  return pyramid;
}

function sample(data, w, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * w + x0;
  return (1 - fx) * (1 - fy) * data[i] + fx * (1 - fy) * data[i + 1]
    + (1 - fx) * fy * data[i + w] + fx * fy * data[i + w + 1];
}

// ---- Lucas-Kanade ------------------------------------------------------------

const LK_DEFAULTS = {
  half: 5,            // 11x11 window: wide enough to span a piece edge, cheap enough for 20fps
  iterations: 12,
  minEigenvalue: 2e-5, // reject points on flat felt/table — they would invent motion
  maxResidual: 0.14,  // mean |difference| after convergence, in [0,1] gray units
};

/** One point, one pyramid level. `gx,gy` is the guess inherited from the coarser
 *  level; the return is the total displacement at THIS level. */
function trackAtLevel(prev, next, px, py, gx, gy, o) {
  const { w, h, data: A, ix, iy } = prev;
  const { w: nw, h: nh, data: B } = next;
  const half = o.half;
  if (px < half + 1 || py < half + 1 || px > w - half - 2 || py > h - half - 2) return null;

  const n = (2 * half + 1) * (2 * half + 1);
  const win = new Float32Array(n);
  const winX = new Float32Array(n);
  const winY = new Float32Array(n);
  let gxx = 0, gxy = 0, gyy = 0;
  for (let dy = -half, k = 0; dy <= half; dy++) {
    for (let dx = -half; dx <= half; dx++, k++) {
      const sx = px + dx, sy = py + dy;
      const a = sample(ix, w, sx, sy);
      const b = sample(iy, w, sx, sy);
      win[k] = sample(A, w, sx, sy);
      winX[k] = a;
      winY[k] = b;
      gxx += a * a; gxy += a * b; gyy += b * b;
    }
  }
  const det = gxx * gyy - gxy * gxy;
  // Smallest eigenvalue of the structure tensor = Shi-Tomasi trackability. A window
  // with texture in only one direction (a board edge) slides along it — the aperture
  // problem — and would contribute a confident lie to the homography fit.
  const eig = 0.5 * (gxx + gyy - Math.sqrt(Math.max(0, (gxx - gyy) ** 2 + 4 * gxy * gxy))) / n;
  if (det <= 1e-12 || eig < o.minEigenvalue) return null;

  let vx = gx, vy = gy;
  let residual = Infinity;
  for (let it = 0; it < o.iterations; it++) {
    let bx = 0, by = 0, sum = 0;
    for (let dy = -half, k = 0; dy <= half; dy++) {
      for (let dx = -half; dx <= half; dx++, k++) {
        const sx = px + vx + dx, sy = py + vy + dy;
        if (sx < 0 || sy < 0 || sx > nw - 2 || sy > nh - 2) return null;
        const d = win[k] - sample(B, nw, sx, sy);
        sum += Math.abs(d);
        bx += d * winX[k];
        by += d * winY[k];
      }
    }
    residual = sum / n;
    const stepX = (gyy * bx - gxy * by) / det;
    const stepY = (gxx * by - gxy * bx) / det;
    vx += stepX;
    vy += stepY;
    if (stepX * stepX + stepY * stepY < 1e-4) break;
  }
  // A window that converged onto something that does not actually look like the
  // original patch (a hand swept over it, the piece it sat on was lifted) is not a
  // correspondence — better dropped than voted with.
  if (!(residual <= o.maxResidual)) return null;
  return [vx, vy];
}

/** Pyramidal LK over a flat [x0,y0,x1,y1,...] array, coarse level first. */
export function trackPoints(prevPyramid, nextPyramid, points, opts = {}) {
  const o = { ...LK_DEFAULTS, ...opts };
  const count = points.length >> 1;
  const tracked = new Float64Array(points.length);
  const ok = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const x = points[2 * i], y = points[2 * i + 1];
    let gx = 0, gy = 0, lost = false;
    for (let l = prevPyramid.length - 1; l >= 0; l--) {
      const s = 1 / (1 << l);
      const v = trackAtLevel(prevPyramid[l], nextPyramid[l], x * s, y * s, gx, gy, o);
      if (v === null) { lost = true; break; }
      gx = l > 0 ? 2 * v[0] : v[0];
      gy = l > 0 ? 2 * v[1] : v[1];
    }
    if (lost) continue;
    ok[i] = 1;
    tracked[2 * i] = x + gx;
    tracked[2 * i + 1] = y + gy;
  }
  return { points: tracked, ok };
}

// ---- homography --------------------------------------------------------------

/** Gaussian elimination with partial pivoting. `a` (n*n, row-major) and `b` are
 *  consumed. Returns null when the system is singular. */
function solveLinear(a, b, n) {
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r * n + col]) > Math.abs(a[pivot * n + col])) pivot = r;
    }
    if (Math.abs(a[pivot * n + col]) < 1e-12) return null;
    if (pivot !== col) {
      for (let c = 0; c < n; c++) {
        const t = a[col * n + c]; a[col * n + c] = a[pivot * n + c]; a[pivot * n + c] = t;
      }
      const t = b[col]; b[col] = b[pivot]; b[pivot] = t;
    }
    const d = a[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = a[r * n + col] / d;
      if (f === 0) continue;
      for (let c = col; c < n; c++) a[r * n + c] -= f * a[col * n + c];
      b[r] -= f * b[col];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= a[r * n + c] * x[c];
    x[r] = s / a[r * n + r];
  }
  return x;
}

/** Hartley normalisation: centroid to the origin, mean radius to sqrt(2). Without it
 *  the 8x8 system mixes pixel-squared and pixel terms and conditions badly. */
function normalisation(points, indices) {
  let cx = 0, cy = 0;
  for (const i of indices) { cx += points[2 * i]; cy += points[2 * i + 1]; }
  cx /= indices.length;
  cy /= indices.length;
  let d = 0;
  for (const i of indices) d += Math.hypot(points[2 * i] - cx, points[2 * i + 1] - cy);
  d /= indices.length;
  return { s: d > 1e-9 ? Math.SQRT2 / d : 1, cx, cy };
}

function matmul3(m, n) {
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = m[r * 3] * n[c] + m[r * 3 + 1] * n[3 + c] + m[r * 3 + 2] * n[6 + c];
    }
  }
  return out;
}

/**
 * Least-squares homography from >=4 correspondences, h33 fixed to 1.
 *
 * Fixing h33 only fails for a homography that sends the source origin to infinity,
 * which no camera looking at a board in front of it produces.
 */
export function homographyFromPairs(src, dst, indices) {
  const idx = indices ?? Array.from({ length: src.length >> 1 }, (_, i) => i);
  if (idx.length < 4) return null;
  const ns = normalisation(src, idx);
  const nd = normalisation(dst, idx);

  const ata = new Float64Array(64);
  const atb = new Float64Array(8);
  const row = new Float64Array(8);
  const accumulate = (target) => {
    for (let i = 0; i < 8; i++) {
      atb[i] += row[i] * target;
      for (let j = 0; j < 8; j++) ata[i * 8 + j] += row[i] * row[j];
    }
  };
  for (const i of idx) {
    const x = (src[2 * i] - ns.cx) * ns.s, y = (src[2 * i + 1] - ns.cy) * ns.s;
    const u = (dst[2 * i] - nd.cx) * nd.s, v = (dst[2 * i + 1] - nd.cy) * nd.s;
    row.set([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    accumulate(u);
    row.set([0, 0, 0, x, y, 1, -v * x, -v * y]);
    accumulate(v);
  }
  const h = solveLinear(ata, atb, 8);
  if (h === null) return null;
  for (const value of h) if (!Number.isFinite(value)) return null;

  const hn = new Float64Array([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1]);
  const tSrc = new Float64Array([ns.s, 0, -ns.s * ns.cx, 0, ns.s, -ns.s * ns.cy, 0, 0, 1]);
  const tDstInv = new Float64Array([1 / nd.s, 0, nd.cx, 0, 1 / nd.s, nd.cy, 0, 0, 1]);
  const H = matmul3(matmul3(tDstInv, hn), tSrc);
  if (Math.abs(H[8]) < 1e-12) return null;
  for (let i = 0; i < 9; i++) H[i] /= H[8];
  return H;
}

export function applyHomography(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  if (Math.abs(w) < 1e-12) return null;
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

/** Deterministic RNG: a tracker that behaves differently on replay is untestable. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * RANSAC homography. The outliers are not noise — they are the *other* things in the
 * scene: a hand crossing the board, a piece being lifted, and every point that landed
 * on a tall piece rather than the playing surface (a piece top is metres out of the
 * board plane in projective terms, so it moves differently under parallax).
 */
export function ransacHomography(src, dst, opts = {}) {
  const { threshold = 2.0, iterations = 160, seed = 1, earlyRatio = 0.85 } = opts;
  const count = src.length >> 1;
  if (count < 4) return null;
  const random = mulberry32(seed);
  const limit = threshold * threshold;
  const pick = new Int32Array(4);
  let best = null;
  let bestCount = 0;

  for (let it = 0; it < iterations; it++) {
    for (let k = 0; k < 4; k++) {
      let candidate;
      let clash;
      do {
        candidate = Math.floor(random() * count) % count;
        clash = false;
        for (let j = 0; j < k; j++) if (pick[j] === candidate) clash = true;
      } while (clash);
      pick[k] = candidate;
    }
    const H = homographyFromPairs(src, dst, pick);
    if (H === null) continue;
    let inliers = 0;
    for (let i = 0; i < count; i++) {
      const p = applyHomography(H, src[2 * i], src[2 * i + 1]);
      if (p === null) continue;
      const ex = p[0] - dst[2 * i], ey = p[1] - dst[2 * i + 1];
      if (ex * ex + ey * ey <= limit) inliers++;
    }
    if (inliers > bestCount) { bestCount = inliers; best = H; }
    if (bestCount >= earlyRatio * count) break;
  }
  if (best === null || bestCount < 4) return null;

  // Refit on the consensus set: the 4-point model that won only had to be good
  // enough to find the inliers, not to be the best fit to them.
  const inliers = [];
  for (let i = 0; i < count; i++) {
    const p = applyHomography(best, src[2 * i], src[2 * i + 1]);
    if (p === null) continue;
    const ex = p[0] - dst[2 * i], ey = p[1] - dst[2 * i + 1];
    if (ex * ex + ey * ey <= limit) inliers.push(i);
  }
  const refined = inliers.length >= 4 ? homographyFromPairs(src, dst, inliers) : null;
  return {
    H: refined ?? best,
    inliers: inliers.length,
    ratio: inliers.length / count,
  };
}

/** The homography taking the unit square's (0,0),(1,0),(1,1),(0,1) to `quad`. */
export function unitSquareToQuad(quad) {
  const src = new Float64Array([0, 0, 1, 0, 1, 1, 0, 1]);
  const dst = new Float64Array(8);
  for (let i = 0; i < 4; i++) { dst[2 * i] = quad[i][0]; dst[2 * i + 1] = quad[i][1]; }
  return homographyFromPairs(src, dst);
}

/**
 * Would this quad still be a board seen by a camera? A tracker that has drifted onto
 * the table cloth reports a bow-tie, a sliver, or something off in the next county —
 * and DESIGN §4.4.1 is explicit that streaming through a mis-localised camera is
 * worse than an honest gap, because the crops are then *confidently* wrong.
 */
export function isSaneQuad(quad, { minArea = 0.01, maxArea = 4, bounds = 0.6 } = {}) {
  if (!Array.isArray(quad) || quad.length !== 4) return false;
  for (const p of quad) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return false;
    if (p[0] < -bounds || p[0] > 1 + bounds || p[1] < -bounds || p[1] > 1 + bounds) return false;
  }
  let area = 0;
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i], b = quad[(i + 1) % 4], c = quad[(i + 2) % 4];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    const turn = Math.sign(cross);
    if (turn === 0) return false;
    if (sign === 0) sign = turn;
    else if (turn !== sign) return false;   // a bow-tie or a reflex corner
    area += a[0] * b[1] - b[0] * a[1];
  }
  area = Math.abs(area) / 2;
  return area >= minArea && area <= maxArea;
}

// ---- the tracker -------------------------------------------------------------

export const TRACKER_STATES = ["idle", "seeded", "locked", "lost"];

/**
 * Frame-to-frame planar tracking of one quad.
 *
 * Deliberately frame-to-frame rather than against a keyframe: an orbit around the
 * board leaves no keyframe recognisable for long, and the operator can always
 * re-anchor by touching a handle. The cost is slow drift, which is why the page
 * shows the lock state instead of pretending the quad is exact.
 */
export class PlaneTracker {
  /**
   * `subdivisions` is seed points per SQUARE, not per frame — 1 puts one point on every
   * square corner (9x9 = 81), 2 adds the square centres and edge midpoints (17x17).
   * Expressed this way on purpose: the seed lattice has to land on the board's own
   * lattice, and an arbitrary NxN grid silently misses it. A 8x8 grid measured half the
   * surviving points of a 9x9 one on the same frames, with no error to say why.
   */
  constructor({ width, height, levels = 3, subdivisions = 1, minInliers = 12,
                minRatio = 0.35, maxStep = 0.2, seed = 1, lk = {} } = {}) {
    const steps = Math.max(1, Math.round(subdivisions));
    Object.assign(this, { width, height, levels, minInliers, minRatio, maxStep, seed, lk });
    this.subdivisions = steps;
    this.grid = 8 * steps + 1;
    this.corners = null;
    this.state = "idle";
    this.ratio = 0;
    this.inliers = 0;
    this.previous = null;
  }

  /** Anchor on the operator's quad (normalised). Also the "re-lock" path: any
   *  handle drag, detection or reset calls this, and tracking resumes from there. */
  reset(corners) {
    this.corners = corners.map(([x, y]) => [x, y]);
    this.state = "seeded";
    this.ratio = 0;
    this.inliers = 0;
    this.previous = null;
    return this.corners;
  }

  /**
   * Points on the board's OWN lattice — the 9x9 intersections of the 8x8 squares — in
   * level-0 pixels.
   *
   * This is the one place the tracker knows it is looking at a chess board, and it is
   * worth it: a board's square corners are its highest-gradient points *by
   * construction*, while the square interiors are flat colour that the trackability
   * gate rightly throws away. Measured on a real board frame, moving the same 81 points
   * onto the lattice took the surviving-point count from 18 to 31 and halved the drift
   * (1.66px -> 0.87px over ten frames of movement).
   */
  seedPoints(corners) {
    const H = unitSquareToQuad(corners);
    if (H === null) return null;
    const n = this.grid;
    const points = new Float64Array(n * n * 2);
    let k = 0;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const p = applyHomography(H, i / (n - 1), j / (n - 1));
        if (p === null) return null;
        points[k++] = p[0] * this.width;
        points[k++] = p[1] * this.height;
      }
    }
    return points;
  }

  /**
   * Advance one frame. `gray` is width*height in [0,1]. Returns the current state;
   * `corners` are ALWAYS the best available quad — on a loss they are the last good
   * one, frozen, never a half-fitted guess.
   */
  update(gray) {
    if (this.corners === null) return this.report();
    const pyramid = buildPyramid(gray, this.width, this.height, this.levels);
    const previous = this.previous;
    this.previous = pyramid;
    if (previous === null) {
      this.state = this.state === "lost" ? "lost" : "seeded";
      return this.report();
    }

    const seeded = this.seedPoints(this.corners);
    if (seeded === null) return this.lose();
    const { points, ok } = trackPoints(previous, pyramid, seeded, this.lk);

    let pairs = 0;
    for (let i = 0; i < ok.length; i++) if (ok[i]) pairs++;
    if (pairs < this.minInliers) return this.lose();
    const src = new Float64Array(pairs * 2);
    const dst = new Float64Array(pairs * 2);
    for (let i = 0, k = 0; i < ok.length; i++) {
      if (!ok[i]) continue;
      src[2 * k] = seeded[2 * i]; src[2 * k + 1] = seeded[2 * i + 1];
      dst[2 * k] = points[2 * i]; dst[2 * k + 1] = points[2 * i + 1];
      k++;
    }

    const fit = ransacHomography(src, dst, { seed: this.seed });
    if (fit === null) return this.lose();
    this.ratio = fit.ratio;
    this.inliers = fit.inliers;
    if (fit.inliers < this.minInliers || fit.ratio < this.minRatio) return this.lose();

    const moved = [];
    for (const [x, y] of this.corners) {
      const p = applyHomography(fit.H, x * this.width, y * this.height);
      if (p === null) return this.lose();
      moved.push([p[0] / this.width, p[1] / this.height]);
    }
    // One frame cannot move the board across the viewfinder. A jump this big is a
    // bad fit that happened to find friends, not a camera movement.
    for (let i = 0; i < 4; i++) {
      if (Math.hypot(moved[i][0] - this.corners[i][0],
                     moved[i][1] - this.corners[i][1]) > this.maxStep) return this.lose();
    }
    if (!isSaneQuad(moved)) return this.lose();

    this.corners = moved;
    this.state = "locked";
    return this.report();
  }

  /** Freeze, and keep trying: the next frame is tracked from the frozen quad, so a
   *  hand sweeping over the board recovers on its own once it leaves. */
  lose() {
    this.state = "lost";
    this.ratio = 0;
    this.inliers = 0;
    return this.report();
  }

  report() {
    return { state: this.state, corners: this.corners, ratio: this.ratio, inliers: this.inliers };
  }
}
