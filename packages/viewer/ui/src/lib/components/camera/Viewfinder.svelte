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
   * **The picture is FULL-BLEED**, and that is a deliberate reversal. It used to be
   * inset from the chrome so that a corner handle dragged to the extreme edge of the
   * frame could never land under a button; the operator asked for the opposite — the
   * controls float over the picture and the picture gets the whole stage, notch
   * included (`viewport-fit=cover` in `app.html`). The accepted tradeoff is exactly
   * the guarantee that was given up: a handle dragged into the last few millimetres
   * of an edge can now sit under a pill. It is recoverable — the cluster is inert
   * between its buttons, so the handle beside one is still grabbable, and the board
   * can be re-framed from the sheet — and it costs a board that is bigger on screen
   * for every second the page is used.
   *
   * `--cluster-w` / `--cluster-h` are what the cluster reserves at the right (or the
   * bottom, in portrait). The PICTURE ignores them; the banner and the hint line dodge
   * them, so a sentence never renders under a button. Stated here, once per
   * orientation, because the stage is the element all three of them live in.
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
    /** Chips, banner, hint line and the button cluster — all of it floats over the picture. */
    chrome: Snippet;
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
    chrome
  }: Props = $props();

  let stage: HTMLDivElement | undefined = $state();
  let picture: HTMLDivElement | undefined = $state();
  /** The stage's own box, which is now the whole box the picture is fitted into. */
  let inner = $state.raw<Box | null>(null);

  const frame = $derived(camera.kind === 'live' ? camera.frame : null);
  const framing = $derived(camera.kind === 'live' ? camera.framing : null);
  const layout = $derived(frame !== null && inner !== null ? fitPicture(frame, inner) : null);
  const quad = $derived(framing === null ? null : orderedCorners(framing.quad, framing.rot));
  const crop = $derived(frame !== null && quad !== null ? cropBox(quad, frame) : null);

  function measure(): void {
    if (stage === undefined) return;
    const box = stage.getBoundingClientRect();
    inner = { w: Math.max(1, box.width), h: Math.max(1, box.height) };
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
    /* What the floating controls reserve — for the TEXT chrome only; the picture
       runs underneath them. `Rail.svelte` sizes itself from the same two numbers, so
       the reservation and the thing reserved for cannot drift apart. */
    --cluster-w: calc(124px + 8px + max(8px, env(safe-area-inset-right)));
    --cluster-h: 0px;
  }

  @media (orientation: portrait) {
    .camera-stage {
      --cluster-w: 0px;
      /* The portrait row wraps to TWO rows of three (`Rail.svelte` says why), so this
         is written as the sum it has to equal rather than as a total: top padding,
         two pill rows, the row gap between them, and the home indicator. Getting this
         wrong does not clip anything — it silently floats the bottom chrome at the
         wrong distance from the buttons, which is why the arithmetic is spelled out. */
      --cluster-h: calc(8px + 48px + 8px + 48px + max(8px, env(safe-area-inset-bottom)));
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
