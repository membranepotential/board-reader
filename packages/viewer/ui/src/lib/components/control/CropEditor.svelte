<script lang="ts">
  /**
   * The phone's full-frame preview with the crop rectangle drawn on it, draggable
   * by its body and resizable by its four corners.
   *
   * The `.picture` box is sized to the frame's own aspect ratio, and the image and the
   * rectangle both sit inside it at `inset: 0` — so the box IS the picture and a pointer
   * delta divided by its size is a delta in fractions of the frame, with no letterbox
   * to subtract (the /camera viewfinder learned this the hard way with object-fit).
   *
   * The rectangle is controlled: every pointer move reports a new `CropRect` through
   * `onchange` and the parent decides what to show.
   */
  import { CORNERS, dragRect, type Handle } from '$lib/control/crop';
  import type { CropRect, FrameSize } from '$lib/protocol/generated';

  interface Props {
    /** The preview image, or `null` while the phone has sent none. */
    src: string | null;
    /** The frame's size; only its aspect ratio matters here. */
    frame: FrameSize;
    rect: CropRect;
    /** The config sends the whole frame: the rectangle is a starting point, not a crop. */
    whole: boolean;
    onchange: (rect: CropRect) => void;
  }

  const { src, frame, rect, whole, onchange }: Props = $props();

  let picture: HTMLDivElement | undefined = $state();

  /** The gesture in progress — plain data, nothing renders from it. */
  let drag: {
    readonly handle: Handle;
    readonly start: CropRect;
    readonly pointer: number;
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
  } | null = null;

  const CORNER_NAMES: { readonly [K in (typeof CORNERS)[number]]: string } = {
    nw: 'top-left',
    ne: 'top-right',
    se: 'bottom-right',
    sw: 'bottom-left'
  };

  function begin(handle: Handle, event: PointerEvent): void {
    if (picture === undefined || event.button !== 0) return;
    const box = picture.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;
    event.preventDefault();
    (event.currentTarget as Element).setPointerCapture(event.pointerId);
    drag = {
      handle,
      start: rect,
      pointer: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      width: box.width,
      height: box.height
    };
  }

  function move(event: PointerEvent): void {
    if (drag === null || event.pointerId !== drag.pointer) return;
    onchange(
      dragRect(drag.start, drag.handle, {
        dx: (event.clientX - drag.x) / drag.width,
        dy: (event.clientY - drag.y) / drag.height
      })
    );
  }

  function end(event: PointerEvent): void {
    if (drag !== null && event.pointerId === drag.pointer) drag = null;
  }
</script>

<div
  class="picture"
  bind:this={picture}
  style:--ar={frame.width / frame.height}
>
  {#if src !== null}
    <img {src} alt="the phone's full camera frame" draggable="false" />
  {:else}
    <div class="note">no preview from the phone yet</div>
  {/if}

  <div
    class="rect"
    class:whole
    style:left="{rect.x * 100}%"
    style:top="{rect.y * 100}%"
    style:width="{rect.w * 100}%"
    style:height="{rect.h * 100}%"
  >
    <!-- The middle half of the rectangle: the board should reach past both guides. -->
    <div class="guide" style:left="25%"></div>
    <div class="guide" style:left="75%"></div>
    <button
      type="button"
      class="body"
      aria-label="move the crop rectangle"
      onpointerdown={(e) => begin('move', e)}
      onpointermove={move}
      onpointerup={end}
      onpointercancel={end}
    ></button>
    {#each CORNERS as corner (corner)}
      <button
        type="button"
        class="corner {corner}"
        aria-label="resize from the {CORNER_NAMES[corner]} corner"
        onpointerdown={(e) => begin(corner, e)}
        onpointermove={move}
        onpointerup={end}
        onpointercancel={end}
      ></button>
    {/each}
  </div>
</div>

<style>
  .picture {
    position: relative;
    aspect-ratio: var(--ar);
    /* As wide as the column allows, but never taller than most of the window — an
       upright phone's 9:16 preview would otherwise push everything off screen. */
    width: min(100%, calc(70vh * var(--ar)));
    background: #0f1216;
    border: 1px solid #272b33;
    border-radius: 6px;
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
  }
  img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    pointer-events: none;
  }
  .note {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #7d8590;
    font-size: 13px;
  }
  .rect {
    position: absolute;
    border: 2px solid #6aa8ff;
    box-shadow: 0 0 0 100vmax rgb(0 0 0 / 0.45);
  }
  .rect.whole {
    border-style: dashed;
    box-shadow: none;
  }
  .guide {
    position: absolute;
    top: 0;
    bottom: 0;
    border-left: 1px dashed rgb(106 168 255 / 0.7);
    pointer-events: none;
  }
  button {
    position: absolute;
    padding: 0;
    border: 0;
    background: transparent;
    touch-action: none;
  }
  .body {
    inset: 0;
    cursor: move;
  }
  /* Inside the rectangle, not straddling its edge: a rectangle at the frame's edge
     would otherwise have half of each handle clipped off by the picture. */
  .corner {
    width: 18px;
    height: 18px;
    background: #6aa8ff;
    border: 2px solid #14171c;
    border-radius: 3px;
  }
  .corner.nw {
    left: -2px;
    top: -2px;
    cursor: nwse-resize;
  }
  .corner.ne {
    right: -2px;
    top: -2px;
    cursor: nesw-resize;
  }
  .corner.se {
    right: -2px;
    bottom: -2px;
    cursor: nwse-resize;
  }
  .corner.sw {
    left: -2px;
    bottom: -2px;
    cursor: nesw-resize;
  }
  button:focus-visible {
    outline: 2px solid #ffcf33;
    outline-offset: 1px;
  }
</style>
