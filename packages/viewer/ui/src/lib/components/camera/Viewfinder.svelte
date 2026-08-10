<script lang="ts">
  /**
   * The viewfinder: the picture, the quad drawn over it, the four handles, and the
   * chrome floating on top.
   *
   * Two properties of this file are correctness, not style.
   *
   * **The picture is a JS-SIZED WRAPPER, not `object-fit: contain`.** `.picture` is
   * given exactly the video's aspect ratio inside the stage's inner box, so the
   * element's box IS the picture and `rect.width / videoWidth` is a single uniform
   * scale. `contain` would letterbox inside the element and introduce a second
   * coordinate space; the one thing that must never happen is screen space leaking
   * into the crop, because the preview would still look perfect while the corners
   * on the wire were wrong.
   *
   * **The picture is INSET FROM the chrome, never underneath it.** `--rail` and
   * `--dock` are the chrome's own thickness, stated once per orientation in the
   * stylesheet below and read back by the measurement. It costs nothing (measured:
   * the same 693x390 picture either way on an 844x390 landscape phone) and it
   * guarantees a corner handle dragged to the edge of the frame can never land
   * under a button. Safe-area insets belong to the chrome, not the picture.
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
    /** The stage minus its chrome; the page parks the position panel against it. */
    inner: Box | null;
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
    /** Chips, banner, hint line and rail — they float over the picture. */
    chrome: Snippet;
  }

  let {
    video = $bindable(),
    inner = $bindable(),
    camera,
    tracking,
    dragging,
    onGrab,
    onMove,
    onRelease,
    onCancel,
    onLoadedMetadata,
    onFrameResize,
    chrome
  }: Props = $props();

  let stage: HTMLDivElement | undefined = $state();
  let picture: HTMLDivElement | undefined = $state();

  const frame = $derived(camera.kind === 'live' ? camera.frame : null);
  const framing = $derived(camera.kind === 'live' ? camera.framing : null);
  const layout = $derived(frame !== null && inner !== null ? fitPicture(frame, inner) : null);
  const quad = $derived(framing === null ? null : orderedCorners(framing.quad, framing.rot));
  const crop = $derived(frame !== null && quad !== null ? cropBox(quad, frame) : null);

  /**
   * The chrome's thickness comes from the stylesheet rather than from a second copy
   * of the layout in here, so the two orientations state it once each.
   */
  function measure(): void {
    if (stage === undefined) return;
    const style = getComputedStyle(stage);
    const rail = parseFloat(style.getPropertyValue('--rail')) || 0;
    const dock = parseFloat(style.getPropertyValue('--dock')) || 0;
    const box = stage.getBoundingClientRect();
    inner = { w: Math.max(1, box.width - rail), h: Math.max(1, box.height - dock) };
  }

  $effect(() => {
    if (stage === undefined) return;
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
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

<div class="camera-stage" bind:this={stage}>
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

<style>
  /* Nothing on this page scrolls, so the stage is `touch-action: none` — which also
     stops iOS pull-to-refresh firing over the viewfinder. It never captures a
     pointer and never preventDefaults a touch; only a handle does that. */
  .camera-stage {
    position: fixed;
    inset: 0;
    height: 100dvh;
    background: #000;
    touch-action: none;
    overflow: hidden;
    z-index: 0;
    --rail: 150px;
    --dock: 0px;
  }

  @media (orientation: portrait) {
    .camera-stage {
      --rail: 0px;
      --dock: 150px;
    }
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
