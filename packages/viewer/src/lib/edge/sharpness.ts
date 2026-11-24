/**
 * How sharp is this frame? — the focus score the camera page stamps on every
 * streamed frame (`FrameHeader.sharpness`).
 *
 * A phone is handheld: micro-shake smears roughly half of its frames, so the node
 * prefers the crisp frame of a shaky second, and "crisp" is this number.
 *
 * **Tenengrad**, the classic autofocus measure: mean squared Sobel gradient
 * magnitude. Blur is a low-pass filter and gradients are the high-pass half, so
 * blurring a scene strictly lowers this while leaving everything else about the
 * scene alone — which is exactly the comparison being made, the same board seen
 * frame after frame.
 *
 * It is normalised by the mean intensity, squared, because the alternative is a
 * meter that prefers whichever candidate the autoexposure happened to brighten:
 * scaling every pixel by `c` scales every gradient by `c` and the raw sum by `c²`.
 * The scores are therefore comparable across a window in which exposure drifts, and
 * meaningless in absolute terms — only their ORDER is ever used.
 *
 * No DOM here, deliberately, like `motion.ts` beside it: it takes a
 * grayscale `Float32Array` and returns a number, so it is tested directly
 * (`sharpness.test.ts`).
 */

/**
 * A grayscale image as every module in `#lib/edge` passes one:
 * row-major values in [0,1], `width * height` of them.
 */
export interface GrayImage {
  readonly data: Float32Array;
  readonly width: number;
  readonly height: number;
}

/** RGBA bytes (a canvas readback) -> luma in [0,1], written into `out`. */
export function toGray(rgba: ArrayLike<number>, out: Float32Array): Float32Array {
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (0.299 * rgba[p]! + 0.587 * rgba[p + 1]! + 0.114 * rgba[p + 2]!) / 255;
  }
  return out;
}

/**
 * Floor on the squared mean intensity, so a frame captured with the lens covered
 * cannot divide its way to the top of the ranking.
 */
const MIN_MEAN_SQ = 1e-4;

/**
 * Mean squared Sobel gradient magnitude over the interior, divided by the mean
 * intensity squared. Higher is sharper; a flat field scores 0.
 *
 * Comparable only WITHIN one burst of the same scene — it is a ranking key, not a
 * calibrated quantity. Anything smaller than 3x3 has no interior to measure and
 * scores 0 rather than throwing: a candidate that cannot be scored simply never
 * wins.
 */
export function tenengrad({ data, width, height }: GrayImage): number {
  if (width < 3 || height < 3 || data.length < width * height) return 0;

  let energy = 0;
  let sum = 0;
  // The 1 px border is skipped: a 3x3 kernel has no neighbourhood there, and
  // clamping it would invent an edge where the crop happens to end.
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const tl = data[i - width - 1]!;
      const tc = data[i - width]!;
      const tr = data[i - width + 1]!;
      const ml = data[i - 1]!;
      const mr = data[i + 1]!;
      const bl = data[i + width - 1]!;
      const bc = data[i + width]!;
      const br = data[i + width + 1]!;
      const gx = tr + 2 * mr + br - (tl + 2 * ml + bl);
      const gy = bl + 2 * bc + br - (tl + 2 * tc + tr);
      energy += gx * gx + gy * gy;
      sum += data[i]!;
    }
  }

  const pixels = (width - 2) * (height - 2);
  const mean = sum / pixels;
  return energy / pixels / Math.max(mean * mean, MIN_MEAN_SQ);
}
