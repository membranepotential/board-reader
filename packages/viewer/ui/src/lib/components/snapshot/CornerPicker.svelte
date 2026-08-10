<script lang="ts">
  /**
   * The photo as posted, with the board quad and its four draggable corners.
   *
   * Three rules this component exists to keep, all of them paid for in field bugs:
   *
   *  1. **The handles are real DOM elements over the canvas, never drawn pixels.** Each
   *     owns its own `touch-action: none` and its own pointer capture, so a drag is only
   *     ever stolen from the page scroll when it starts ON a handle. The picture itself
   *     stays `touch-action: pan-y` unconditionally and never calls `preventDefault` or
   *     `setPointerCapture` — a thumb landing anywhere that is not a handle scrolls the
   *     page like anything else.
   *  2. **The picture only ever ADDS a corner.** Placing a point is a single, uncontinued
   *     tap; fine-tuning it is a deliberately separate gesture on the handle that grew
   *     there. Touch-action is decided once, at touchstart, by whatever element was hit,
   *     and reassigning pointer capture afterwards does not retroactively change that —
   *     so a tap-then-drag-without-lifting would still be stolen as a page scroll.
   *     Splitting "place" from "adjust" avoids relying on that at all.
   *  3. **Geometry is in image pixels.** The canvas is sized in image pixels and scaled
   *     by CSS; the SVG overlay uses the image's own viewBox; only the handles' `left`/
   *     `top` are converted, at the last moment, by one uniform factor. The on-screen
   *     fit is computed in JS (not a percentage `max-height`, which resolves against an
   *     indefinite height and let a portrait photo overflow its frame) and re-run from a
   *     `ResizeObserver`, because the row re-flows whenever the inspector opens beside
   *     the board.
   */
  import type { Point } from '$lib/protocol';
  import Loupe from './Loupe.svelte';
  import { cornerHint, labelAt, pointCount, type Framing } from './framing';
  import type { SourceImage } from './photo';

  interface Props {
    /** The offscreen canvas holding the photo at its posted resolution. */
    source: HTMLCanvasElement | null;
    image: SourceImage;
    framing: Framing;
    /** Rotating labels is only meaningful when a placement can validate them. */
    showRotate: boolean;
    onAdd: (point: Point) => void;
    onMove: (index: number, point: Point) => void;
    /** A drag ended — the moment the (expensive) rectified preview is worth redrawing. */
    onCommit: () => void;
    onRotate: () => void;
    onUndo: () => void;
    onReset: () => void;
  }

  const { source, image, framing, showRotate, onAdd, onMove, onCommit, onRotate, onUndo, onReset }: Props =
    $props();

  let frame: HTMLDivElement | undefined = $state();
  let view: HTMLCanvasElement | undefined = $state();

  /** On-screen size of the photo, in CSS px, and the factor that got it there. */
  let fit = $state({ width: 0, height: 0, scale: 1 });

  let dragging: number | null = $state(null);
  let magnifierOn = $state(false);
  let pointer: { readonly x: number; readonly y: number } | null = $state(null);
  let focus: Point | null = $state(null);

  const points = $derived(framing.placement.points);
  const hint = $derived(cornerHint(framing));

  /** Paint the photo once per photo — the quad and handles are DOM, not pixels. */
  $effect(() => {
    const ctx = view?.getContext('2d');
    // `image` is read so a new photo (same dimensions or not) repaints.
    if (!ctx || source === null || image.width === 0) return;
    ctx.clearRect(0, 0, image.width, image.height);
    ctx.drawImage(source, 0, 0);
  });

  $effect(() => {
    const box = frame;
    if (!box) return;
    const measure = () => {
      const rect = box.getBoundingClientRect();
      if (!rect.width || !image.width) return;
      const k = Math.min(rect.width / image.width, rect.height / image.height);
      fit = {
        width: Math.floor(image.width * k),
        height: Math.floor(image.height * k),
        scale: k
      };
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(box);
    return () => observer.disconnect();
  });

  /** Client coordinates → image pixels, through the canvas's own live rect. */
  function imagePos(event: PointerEvent): Point {
    const rect = view?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return [0, 0];
    return [
      Math.round((event.clientX - rect.left) * (image.width / rect.width)),
      Math.round((event.clientY - rect.top) * (image.height / rect.height))
    ];
  }

  function handleScale(): number {
    return image.width ? fit.width / image.width : 1;
  }

  function placeCorner(event: PointerEvent): void {
    if (framing.placement.kind === 'placed') return;
    onAdd(imagePos(event));
    onCommit();
  }

  function startDrag(index: number, event: PointerEvent): void {
    event.preventDefault();
    event.stopPropagation();
    dragging = index;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    pointer = { x: event.clientX, y: event.clientY };
    focus = points[index] ?? null;
  }

  function moveDrag(index: number, event: PointerEvent): void {
    if (dragging !== index) return;
    event.preventDefault();
    const point = imagePos(event);
    onMove(index, point);
    pointer = { x: event.clientX, y: event.clientY };
    focus = point;
  }

  function endDrag(index: number, commit: boolean): void {
    if (dragging !== index) return;
    dragging = null;
    pointer = null;
    focus = null;
    if (commit) onCommit();
  }

  /**
   * Hover magnifier: opt-in, and deliberately mouse-only. Touch has no hover, and the
   * scroll invariants above must hold for a thumb no matter what this toggle says — so
   * this listener only ever observes. A drag already owns the loupe and aims it at the
   * corner itself, which is more useful than the raw cursor position; do not fight it.
   */
  function hover(event: PointerEvent): void {
    if (!magnifierOn || event.pointerType !== 'mouse' || dragging !== null || source === null) return;
    pointer = { x: event.clientX, y: event.clientY };
    focus = imagePos(event);
  }

  function leave(): void {
    if (dragging !== null) return;
    pointer = null;
    focus = null;
  }
</script>

<div class="col-label">Photo as posted</div>
<div class="photo-frame" bind:this={frame}>
  <div
    class="corner-stage"
    onpointermove={hover}
    onpointerleave={leave}
    role="presentation"
  >
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <canvas
      bind:this={view}
      width={image.width}
      height={image.height}
      style:width="{fit.width}px"
      style:height="{fit.height}px"
      onpointerdown={placeCorner}
      oncontextmenu={(event) => event.preventDefault()}
      ondragstart={(event) => event.preventDefault()}
    ></canvas>
    {#if points.length >= 2}
      <svg
        class="quad"
        viewBox="0 0 {image.width} {image.height}"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <polyline
          points={[...points, ...(points.length === 4 ? [points[0] as Point] : [])]
            .map((p) => `${p[0]},${p[1]}`)
            .join(' ')}
        />
      </svg>
    {/if}
    {#each points as point, index (index)}
      <div
        class="corner-handle"
        style:left="{point[0] * handleScale()}px"
        style:top="{point[1] * handleScale()}px"
        onpointerdown={(event) => startDrag(index, event)}
        onpointermove={(event) => moveDrag(index, event)}
        onpointerup={() => endDrag(index, true)}
        onpointercancel={() => endDrag(index, false)}
        role="presentation"
      >
        {labelAt(framing, index)}
      </div>
    {/each}
  </div>
</div>

<p class="hint">
  {#each hint as segment, i (i)}{#if segment.strong}<b>{segment.text}</b>{:else}{segment.text}{/if}{/each}
</p>

<div class="row">
  {#if showRotate}
    <button type="button" class="secondary" onclick={onRotate}>Rotate labels</button>
  {/if}
  <button type="button" class="secondary" disabled={pointCount(framing) === 0} onclick={onUndo}>
    Undo
  </button>
  <button type="button" class="secondary" disabled={pointCount(framing) === 0} onclick={onReset}>
    Reset corners
  </button>
  <button
    type="button"
    class="secondary"
    aria-pressed={magnifierOn}
    onclick={() => {
      magnifierOn = !magnifierOn;
      if (!magnifierOn && dragging === null) pointer = null;
    }}
  >
    {magnifierOn ? 'Magnifier on' : 'Magnifier'}
  </button>
</div>

<Loupe {source} anchor={pointer} {focus} {points} rot={framing.rot} />

<style>
  .col-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--dim);
    margin-bottom: 6px;
  }

  /* A square frame, so the photo block matches the board block exactly whatever the
     photo's aspect ratio. */
  .photo-frame {
    width: 100%;
    aspect-ratio: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f1216;
    border: 1px solid var(--line);
    border-radius: 6px;
    overflow: hidden;
  }

  /* Unconditionally pan-y: page scroll always wins over the picture. */
  .corner-stage {
    position: relative;
    line-height: 0;
    touch-action: pan-y;
  }

  canvas {
    display: block;
    max-width: none;
    border: 0;
    border-radius: 0;
  }

  .quad {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    fill: none;
    stroke: var(--accent);
    stroke-width: 2;
    /* The photo is scaled down to fit its frame; a stroke in image units would be a
       hairline on a phone. */
    vector-effect: non-scaling-stroke;
  }

  /* 34px: still a comfortable thumb target, but small enough that four of them on a
     phone-sized photo no longer hide the board edges they are meant to align with.
     Precision comes from the loupe, not from bulk. */
  .corner-handle {
    position: absolute;
    width: 34px;
    height: 34px;
    margin: -17px 0 0 -17px;
    border-radius: 50%;
    background: rgba(255, 207, 51, 0.9);
    border: 2px solid var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 12px;
    line-height: 1;
    color: #14171c;
    touch-action: none;
    -webkit-user-select: none;
    user-select: none;
    -webkit-touch-callout: none;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
  }

  .corner-handle:active {
    background: #ffcf33;
  }

  .hint {
    color: var(--muted);
    font-size: 13px;
    margin: 6px 0;
  }

  .hint b {
    color: var(--warn);
  }

  .row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin: 10px 0;
  }

  button {
    background: #333842;
    color: #fff;
    border: 0;
    border-radius: 9px;
    padding: 10px 16px;
    font: inherit;
    font-size: 15px;
    font-weight: 600;
    min-height: 48px;
    touch-action: manipulation;
    cursor: pointer;
  }

  /* A toggle carries its state in aria-pressed, and the styling reads straight off it —
     no parallel class to keep in sync with the attribute screen readers are told about. */
  button[aria-pressed='true'] {
    background: var(--accent);
  }

  button:disabled {
    opacity: 0.4;
    cursor: default;
  }
</style>
