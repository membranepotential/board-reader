/**
 * Framing guidance for the crop rectangle, from the ChessQ framing measurement
 * (docs/reports/2026-11-24-chessq-framing.md).
 *
 * What that report found, reduced to what a rectangle can tell us:
 *
 *  - the board should fill at least about half the SENT image's width (50 % cost
 *    nothing measurable, 35 % collapsed perfect boards), so sending a wide frame
 *    with a small board in it is the failure to steer away from;
 *  - a portrait image lost about a quarter of the perfect boards, so landscape is
 *    preferred — both for the phone itself and for the rectangle drawn on it.
 *
 * The page cannot see where the board is inside the rectangle, so it cannot measure
 * the 50 %. It draws guides at the middle half of the rectangle instead and leaves
 * that check to the operator's eye; the warnings here are the parts that follow from
 * the rectangle's shape alone.
 */

import type { CropRect, FrameSize } from '$lib/protocol';

/**
 * Pixel width / height above which the board likely fills less than half the
 * rectangle's width. A board filmed at an angle is about as wide as it is tall, or
 * somewhat wider; a rectangle that hugs it top to bottom and is more than twice as
 * wide as tall therefore has the board spanning under half of it.
 */
export const MAX_CROP_ASPECT = 2;

export type FramingWarning =
  | { readonly kind: 'portrait-phone' }
  | { readonly kind: 'portrait-crop' }
  | { readonly kind: 'too-wide' }
  | { readonly kind: 'whole-frame' };

export const FRAMING_COPY: { readonly [K in FramingWarning['kind']]: string } = {
  'portrait-phone':
    'The phone is upright. Turn it to landscape: portrait images read about a quarter fewer boards perfectly.',
  'portrait-crop':
    'The rectangle is taller than wide. Landscape reads best: widen it, or turn the phone.',
  'too-wide':
    'The rectangle is more than twice as wide as tall, so the board likely fills less than half of it. Crop closer to the board.',
  'whole-frame':
    'The whole frame is sent. Unless the board fills at least half its width, draw a rectangle around the board.'
};

/** `crop`'s shape in pixels of `frame`: width over height. */
export function cropAspect(crop: CropRect, frame: FrameSize): number {
  return (crop.w * frame.width) / (crop.h * frame.height);
}

/**
 * The warnings for sending `crop` (`null`: the whole frame) out of a `frame`-sized
 * camera image. No frame size yet means no camera yet, and nothing to warn about.
 */
export function framingWarnings(
  crop: CropRect | null,
  frame: FrameSize | null
): readonly FramingWarning[] {
  if (frame === null) return [];
  const warnings: FramingWarning[] = [];
  if (frame.height > frame.width) warnings.push({ kind: 'portrait-phone' });
  if (crop === null) {
    warnings.push({ kind: 'whole-frame' });
    return warnings;
  }
  const aspect = cropAspect(crop, frame);
  // An upright phone already says "portrait"; the rectangle drawn on it saying so
  // again tells the operator nothing new.
  if (aspect < 1 && frame.height <= frame.width) warnings.push({ kind: 'portrait-crop' });
  if (aspect > MAX_CROP_ASPECT) warnings.push({ kind: 'too-wide' });
  return warnings;
}
