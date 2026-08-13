<script lang="ts">
  /**
   * The viewfinder: the picture, the quad drawn over it, the four handles, the
   * controls beside it and the status strip under it.
   *
   * # The layout, in three sentences
   *
   * The stage is one `position: fixed` box the size of the viewport, it takes the
   * safe-area insets as PADDING, and its content is a single flex line: the frame,
   * then the rail. Landscape lays that line as a row (rail on the right), portrait as
   * a column (rail along the bottom); that one `flex-direction` is the entire
   * difference between the two orientations. Everything else — the picture, the
   * handles, the status strip — lives inside the frame and is positioned against the
   * frame alone.
   *
   * That is the whole model, and it is deliberately dumber than what it replaced.
   * The previous build floated the controls ON a full-bleed picture and had the text
   * chrome dodge them through hand-computed `--cluster-w` / `--cluster-h` reservations
   * — a stack of numbers (`calc(8px + 48px + 8px + 48px + inset)`, a 124 px pill
   * width, per-device `max-height` breakpoints) that had to agree with a layout they
   * could not see. On the phone they did not agree, and the right-hand controls ran
   * off the screen. Nothing here reserves space for anything: the rail occupies its
   * own flex track, so the frame is *by construction* the room that is left, and no
   * arithmetic can drift.
   *
   * Three consequences worth stating, because each was previously a rule somebody had
   * to remember:
   *
   * - **Nothing can be clipped by the notch.** The insets are applied once, at the
   *   stage, so no component below knows the notch exists. It costs the picture the
   *   inset strip it used to bleed into, which is the correct trade — that strip was
   *   under a sensor housing and unreadable anyway.
   * - **A corner handle can never land under a button.** The rail does not overlap
   *   the picture, so the rail does not have to be inert and the handles do not have
   *   to be recovered from beneath it.
   * - **It costs no picture.** The picture is letterboxed into whatever box it is
   *   given, and on phone aspect ratios the letterbox already exceeded the rail:
   *   measured on an 844x390 landscape stage with a 16:9 camera, the picture is
   *   693x390 whether the rail floats or reserves.
   *
   * # The one property that is correctness, not style
   *
   * **The picture is a JS-SIZED WRAPPER, not `object-fit: contain`.** `.picture` is
   * given exactly the video's aspect ratio inside the frame's box, so the element's
   * box IS the picture and `rect.width / videoWidth` is a single uniform scale.
   * `contain` would letterbox inside the element and introduce a second coordinate
   * space; the one thing that must never happen is screen space leaking into the
   * crop, because the preview would still look perfect while the corners on the wire
   * were wrong.
   *
   * Landscape is the BASE stylesheet and `@media (orientation: portrait)` the
   * override. That is as close to "landscape by default" as the platform allows:
   * iOS Safari implements no `screen.orientation.lock()`.
   */
  import type { Snippet } from 'svelte';
  import { cropBox, orderedCorners } from '$lib/geometry/corners';
  import type { Point } from '$lib/protocol';
  import CornerHandles from './CornerHandles.svelte';
  import { fitPicture, type Box, type CameraState, type CornerIndex, type TrackingState } from './state';

  interface Props {
    /** The element the page attaches the camera stream to. */
    video: HTMLVideoElement | undefined;
    camera: CameraState;
    tracking: TrackingState;
    dragging: CornerIndex | null;
    onGrab: (corner: CornerIndex) => void;
    onMove: (corner: CornerIndex, point: Point) => void;
    onRelease: () => void;
    onCancel: () => void;
    /**
     * Attached as element attributes rather than from an effect in the page: the
     * camera is opened during `onMount`, and `loadedmetadata` can fire before a
     * post-mount effect would have got a listener on.
     */
    onLoadedMetadata: () => void;
    onFrameResize: () => void;
    /**
     * Chips, banner and hint line. Rendered INSIDE the frame, along its bottom edge,
     * which is what makes "the status strip never runs under the buttons" a fact of
     * the box tree rather than an offset somebody has to keep correct.
     */
    chrome: Snippet;
    /** The controls. Its own flex track beside the frame — see the note above. */
    controls: Snippet;
  }

  let {
    video = $bindable(),
    camera,
    tracking,
    dragging,
    onGrab,
    onMove,
    onRelease,
    onCancel,
    onLoadedMetadata,
    onFrameResize,
    chrome,
    controls
  }: Props = $props();

  let frameEl: HTMLDivElement | undefined = $state();
  let picture: HTMLDivElement | undefined = $state();
  /** The frame's own box — the room left over once the rail has taken its track. */
  let inner = $state.raw<Box | null>(null);

  const frame = $derived(camera.kind === 'live' ? camera.frame : null);
  const framing = $derived(camera.kind === 'live' ? camera.framing : null);
  const layout = $derived(frame !== null && inner !== null ? fitPicture(frame, inner) : null);
  const quad = $derived(framing === null ? null : orderedCorners(framing.quad, framing.rot));
  const crop = $derived(frame !== null && quad !== null ? cropBox(quad, frame) : null);

  function measure(): void {
    if (frameEl === undefined) return;
    const box = frameEl.getBoundingClientRect();
    inner = { w: Math.max(1, box.width), h: Math.max(1, box.height) };
  }

  $effect(() => {
    if (frameEl === undefined) return;
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(frameEl);
    return () => observer.disconnect();
  });

  /**
   * Client coordinates -> video pixels. The picture element's rect IS the picture,
   * so this is one uniform scale with no second coordinate space to get wrong.
   */
  function toVideo(clientX: number, clientY: number): Point {
    if (picture === undefined || frame === null) return [0, 0];
    const box = picture.getBoundingClientRect();
    const x = ((clientX - box.left) * frame.width) / box.width;
    const y = ((clientY - box.top) * frame.height) / box.height;
    return [
      Math.max(0, Math.min(frame.width, Math.round(x))),
      Math.max(0, Math.min(frame.height, Math.round(y)))
    ];
  }
</script>

<svelte:window onresize={measure} />

<div class="camera-stage">
  <div class="frame" bind:this={frameEl}>
    <div
      class="picture"
      bind:this={picture}
      style:width={layout ? `${layout.width}px` : undefined}
      style:height={layout ? `${layout.height}px` : undefined}
      style:left={layout ? `${layout.left}px` : undefined}
      style:top={layout ? `${layout.top}px` : undefined}
    >
      <!-- svelte-ignore a11y_media_has_caption -->
      <video
        bind:this={video}
        playsinline
        muted
        autoplay
        onloadedmetadata={onLoadedMetadata}
        onresize={onFrameResize}
        oncontextmenu={(event) => event.preventDefault()}
      ></video>

      {#if frame !== null && quad !== null && crop !== null}
        <svg
          preserveAspectRatio="none"
          viewBox="0 0 {frame.width} {frame.height}"
          aria-hidden="true"
        >
          <rect class="crop-line" x={crop.x} y={crop.y} width={crop.w} height={crop.h} />
          <!-- Tracking lost: the quad is frozen where it last fitted, and says so. -->
          <polygon
            class="quad-line"
            class:lost={tracking === 'lost'}
            points={quad.map((p) => p.join(',')).join(' ')}
          />
        </svg>
      {/if}

      {#if framing !== null && layout !== null}
        <CornerHandles
          quad={framing.quad}
          rot={framing.rot}
          scale={layout.scale}
          {dragging}
          {toVideo}
          {onGrab}
          {onMove}
          {onRelease}
          {onCancel}
        />
      {/if}
    </div>

    {@render chrome()}
  </div>

  {@render controls()}
</div>

<style>
  /* The whole viewport, and nothing may leave it: `overflow: hidden` is what makes
     "cut off at an edge" unrepresentable rather than merely unlikely — a child that
     miscomputes its size is clipped by the stage instead of widening the page and
     pushing the far edge off the screen.

     `100dvh` and not just `inset: 0`: on iOS Safari a fixed element's containing
     block is the LARGE viewport, so `bottom: 0` sits behind the toolbar whenever the
     toolbar is showing. `dvh` is the one that tracks it.

     The safe-area insets are taken HERE, once, as padding. Every component inside is
     then free of `env()` entirely — which is the point, because the insets are the
     part of the geometry that only exists on a device nobody is testing on.

     Nothing on this page scrolls, so the stage is `touch-action: none` — which also
     stops iOS pull-to-refresh firing over the viewfinder. It never captures a pointer
     and never preventDefaults a touch; only a handle does that. */
  .camera-stage {
    position: fixed;
    inset: 0;
    height: 100dvh;
    background: #000;
    touch-action: none;
    overflow: hidden;
    z-index: 0;
    padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom)
      env(safe-area-inset-left);
    /* The one flex line: frame, then rail. */
    display: flex;
    flex-direction: row;
  }

  /* The entire portrait override. The rail is the last child either way, so a row
     puts it on the right and a column puts it along the bottom — one DOM order, one
     declaration, no second set of offsets to keep in agreement with the first. */
  @media (orientation: portrait) {
    .camera-stage {
      flex-direction: column;
    }
  }

  /* Whatever the rail leaves. `min-*: 0` because a flex item's automatic minimum size
     is its content, and the picture is an absolutely-positioned child sized in JS —
     without these the frame could refuse to shrink and push the rail off the edge,
     which is precisely the failure being designed out. */
  .frame {
    position: relative;
    flex: 1 1 auto;
    min-width: 0;
    min-height: 0;
  }

  .picture {
    position: absolute;
    left: 0;
    top: 0;
    line-height: 0;
  }

  .picture video {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
  }

  .picture svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  .quad-line {
    fill: rgba(42, 110, 240, 0.08);
    stroke: var(--accent-soft);
    stroke-width: 2;
    vector-effect: non-scaling-stroke;
  }

  .quad-line.lost {
    fill: rgba(255, 143, 143, 0.06);
    stroke: var(--bad);
    stroke-dasharray: 8 6;
  }

  .crop-line {
    fill: none;
    stroke: var(--warn);
    stroke-width: 1.5;
    stroke-dasharray: 7 6;
    opacity: 0.75;
    vector-effect: non-scaling-stroke;
  }
</style>
