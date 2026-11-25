/**
 * Browser-only canvas and image helpers for the snapshot page. Nothing here runs at
 * module scope: the page is prerendered, so every function is called from an event
 * handler or an effect.
 */
import {
  MAX_SOURCE_PX,
  POST_QUALITY,
  THUMB_QUALITY,
  meanLuma,
  type SourceImage
} from '#lib/components/snapshot/photo.ts';

export function context(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext('2d');
  // A 2d context is not optional on any browser this page runs in; a null here is a
  // dead page, not a condition to degrade around.
  if (ctx === null) throw new Error('this browser gave us no 2d canvas context');
  return ctx;
}

export function canvasJpeg(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('the browser could not encode this photo'))),
      'image/jpeg',
      quality
    );
  });
}

export function decodeImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const element = new Image();
    element.onload = () => resolve(element);
    element.onerror = () => reject(new Error('the photo could not be decoded'));
    element.src = url;
  });
}

/**
 * Draw a decoded photo into `canvas` at the posted resolution and measure what the
 * capture stage reports about it. Returns the downscale factor with the image.
 */
export async function drawSource(
  canvas: HTMLCanvasElement,
  element: HTMLImageElement
): Promise<{ readonly image: SourceImage; readonly scale: number }> {
  const scale = Math.min(1, MAX_SOURCE_PX / Math.max(element.naturalWidth, element.naturalHeight));
  canvas.width = Math.round(element.naturalWidth * scale);
  canvas.height = Math.round(element.naturalHeight * scale);
  const ctx = context(canvas);
  ctx.drawImage(element, 0, 0, canvas.width, canvas.height);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const bytes = (await canvasJpeg(canvas, POST_QUALITY)).size;
  return {
    scale,
    image: {
      width: canvas.width,
      height: canvas.height,
      dataUrl: canvas.toDataURL('image/jpeg', THUMB_QUALITY),
      bytes,
      meanLuma: meanLuma({ data: pixels.data, width: canvas.width, height: canvas.height })
    }
  };
}

/** Put a stored photo back into `canvas` at its stored size. */
export async function restoreSource(canvas: HTMLCanvasElement, image: SourceImage): Promise<void> {
  const element = await decodeImage(image.dataUrl);
  canvas.width = image.width;
  canvas.height = image.height;
  context(canvas).drawImage(element, 0, 0);
}
