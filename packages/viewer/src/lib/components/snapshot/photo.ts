/**
 * The photo in hand, and what the capture stage says about it.
 *
 * The photo is downscaled to a 1600 px long side in the browser, and that downscaled
 * image is exactly what is POSTed to `/estimate` — so the capture stage describes the
 * very pixels the node reads.
 */

import { LOW_LIGHT_MEAN } from '#lib/protocol/index.ts';
import { kv, kvIf, type KvRow } from './kv';
import { fmt } from './reading';

/** Long side the photo is downscaled to before anything looks at it. */
export const MAX_SOURCE_PX = 1600;
/** What the posted JPEG is encoded at. */
export const POST_QUALITY = 0.9;
/** …and the much smaller copy a history entry keeps so a read can be restored. */
export const THUMB_QUALITY = 0.75;

/** The downscaled photo everything else is measured in. */
export interface SourceImage {
  readonly width: number;
  readonly height: number;
  /** JPEG data URL of the source — the history thumbnail and the restore path. */
  readonly dataUrl: string;
  /** Size of the JPEG actually posted, in bytes. */
  readonly bytes: number;
  /** This page's mirror of the node's low-light check, so it can explain the warning. */
  readonly meanLuma: number;
}

/** Where a photo came from — the only part of the capture stage that is not derivable. */
export interface CaptureFacts {
  /** "IMG_0042.jpg · 3204 kB", or the retained upload it was pulled back from. */
  readonly origin: string;
  /** When the node retained it, for a photo loaded off the node. */
  readonly retainedAt: string | null;
  readonly originalWidth: number;
  readonly originalHeight: number;
  /** Downscale factor applied to reach `SourceImage` — 1 when the photo was small. */
  readonly scale: number;
}

export function captureSummary(image: SourceImage): string {
  return (
    `${image.width}x${image.height} px · ${kb(image.bytes)} · ` +
    `mean luma ${fmt(image.meanLuma, 1)}`
  );
}

export function captureRows(image: SourceImage, facts: CaptureFacts): readonly KvRow[] {
  return [
    kv('source file', facts.origin),
    ...(facts.retainedAt === null ? [] : [kv('retained at', facts.retainedAt)]),
    kv('original size', `${facts.originalWidth}x${facts.originalHeight} px`),
    kv('posted size', `${image.width}x${image.height} px (scale ${fmt(facts.scale, 3)})`),
    kv('posted JPEG', `${kb(image.bytes)} at q${POST_QUALITY.toFixed(2)}`),
    kvIf(
      'mean luma',
      `${fmt(image.meanLuma, 1)} / 255 (server warns below ${LOW_LIGHT_MEAN})`,
      image.meanLuma < LOW_LIGHT_MEAN
    )
  ];
}

export function kb(bytes: number): string {
  return `${(bytes / 1024).toFixed(0)} kB`;
}

/**
 * Mean luma of an RGBA image, subsampled — the page's own mirror of the node's
 * low-light check (`server.LOW_LIGHT_MEAN`). Stepping rather than reading every
 * pixel keeps a 12 MP phone photo instant, and a mean does not need every sample.
 */
export function meanLuma(image: {
  readonly data: Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
}): number {
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
