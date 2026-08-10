<script lang="ts">
  /**
   * The magnifier: a fixed 140 px window showing the photo at 4×, floating near the
   * finger, with the quad and its corner markers redrawn INSIDE it.
   *
   * Drawing the markers again is not decoration. The handles are DOM elements sitting
   * above the canvas, so a magnified copy of the canvas alone shows none of them — and
   * lining a corner up means seeing the handle and the board edge in the same view.
   * Markers are translucent rings with the label set outside them, so the pixels being
   * aimed at stay visible underneath.
   */
  import { labelOf, type LabelRotation } from '$lib/geometry/corners';
  import type { Point } from '$lib/protocol';

  interface Props {
    /** The offscreen source canvas — the photo in its own pixel space. */
    source: HTMLCanvasElement | null;
    /** Where the pointer is, in client px. `null` hides the loupe. */
    anchor: { readonly x: number; readonly y: number } | null;
    /** The image-pixel point under the crosshair. */
    focus: Point | null;
    points: readonly Point[];
    rot: LabelRotation;
  }

  const { source, anchor, focus, points, rot }: Props = $props();

  const SIZE = 140;
  /** Magnification, in loupe pixels per source pixel. */
  const ZOOM = 4;
  /** How far above the finger the window floats, so it is never under it. */
  const GAP = 24;

  let canvas: HTMLCanvasElement | undefined = $state();

  const shown = $derived(source !== null && anchor !== null && focus !== null);

  /**
   * Prefer floating above the finger; flip below and clamp to the viewport so it never
   * drifts off-screen when a corner sits near the top or an edge of a small phone.
   */
  const box = $derived.by(() => {
    if (anchor === null || typeof window === 'undefined') return { left: 0, top: 0 };
    let top = anchor.y - SIZE - GAP;
    if (top < 4) top = Math.min(window.innerHeight - SIZE - 4, anchor.y + GAP);
    const left = Math.max(4, Math.min(window.innerWidth - SIZE - 4, anchor.x - SIZE / 2));
    return { left, top };
  });

  $effect(() => {
    const ctx = canvas?.getContext('2d');
    if (!ctx || source === null || focus === null || !shown) return;
    const span = SIZE / ZOOM;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.drawImage(source, focus[0] - span / 2, focus[1] - span / 2, span, span, 0, 0, SIZE, SIZE);

    const centre = SIZE / 2;
    ctx.strokeStyle = '#2a6ef0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centre, 0);
    ctx.lineTo(centre, SIZE);
    ctx.moveTo(0, centre);
    ctx.lineTo(SIZE, centre);
    ctx.stroke();

    const at = (q: Point): Point => [
      (q[0] - focus[0]) * ZOOM + centre,
      (q[1] - focus[1]) * ZOOM + centre
    ];
    if (points.length >= 2) {
      ctx.strokeStyle = '#2a6ef0';
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((q, i) => {
        const [x, y] = at(q);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      if (points.length === 4) ctx.closePath();
      ctx.stroke();
    }

    const radius = 9;
    ctx.font = '700 11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    points.forEach((q, i) => {
      const [x, y] = at(q);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,207,51,.28)';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffcf33';
      ctx.stroke();
      // Dark outline first: the label stays legible over any photo underneath.
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,.8)';
      ctx.strokeText(labelOf(i, rot), x, y - radius - 2);
      ctx.fillStyle = '#ffcf33';
      ctx.fillText(labelOf(i, rot), x, y - radius - 2);
    });
  });
</script>

<canvas
  bind:this={canvas}
  class="loupe"
  width={SIZE}
  height={SIZE}
  style:display={shown ? 'block' : 'none'}
  style:left="{box.left}px"
  style:top="{box.top}px"
  aria-hidden="true"
></canvas>

<style>
  .loupe {
    position: fixed;
    width: 140px;
    height: 140px;
    border: 2px solid var(--accent);
    z-index: 40;
    border-radius: 10px;
    /* Never a target: it floats under the finger that is dragging something else. */
    pointer-events: none;
    background: #000;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
  }
</style>
