<script lang="ts">
  /**
   * The four draggable corners, laid over the picture.
   *
   * 44px, the touch minimum — this is a full-screen instrument, not a thumbnail.
   *
   * ONLY a drag that starts on a handle ever takes the pointer: the handle
   * `preventDefault`s and captures, the stage and the video never do either. That
   * contract is what lets the stage be `touch-action: none` (nothing on this page
   * scrolls, and iOS pull-to-refresh must not fire over the viewfinder) without the
   * stage ever stealing a touch meant for a corner.
   */
  import { labelOf, type LabelRotation } from '$lib/geometry/corners';
  import type { CornerQuad, Point } from '$lib/protocol';
  import type { CornerIndex } from './state';

  interface Props {
    /** Corners in VIDEO pixels, storage order. */
    quad: CornerQuad;
    rot: LabelRotation;
    /** CSS px per video px — `PictureLayout.scale`. */
    scale: number;
    /** Which handle currently owns the pointer, if any. */
    dragging: CornerIndex | null;
    /** Client coordinates -> video pixels, clamped to the frame. */
    toVideo: (clientX: number, clientY: number) => Point;
    onGrab: (corner: CornerIndex) => void;
    onMove: (corner: CornerIndex, point: Point) => void;
    onRelease: () => void;
    onCancel: () => void;
  }

  const { quad, rot, scale, dragging, toVideo, onGrab, onMove, onRelease, onCancel }: Props =
    $props();

  const INDICES: readonly CornerIndex[] = [0, 1, 2, 3];
</script>

{#each INDICES as corner (corner)}
  <!-- A handle is a pointer-drag surface with no keyboard or assistive equivalent —
       it exists to be dragged onto a physical board seen through the camera. Giving
       it `role="button"` would claim an activation it does not have, so the rule is
       suppressed rather than answered with a fiction. -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="corner-handle"
    style:left="{quad[corner][0] * scale}px"
    style:top="{quad[corner][1] * scale}px"
    onpointerdown={(event) => {
      event.preventDefault();
      event.stopPropagation();
      onGrab(corner);
      event.currentTarget.setPointerCapture(event.pointerId);
    }}
    onpointermove={(event) => {
      if (dragging !== corner) return;
      event.preventDefault();
      onMove(corner, toVideo(event.clientX, event.clientY));
    }}
    onpointerup={() => {
      if (dragging === corner) onRelease();
    }}
    onpointercancel={() => {
      if (dragging === corner) onCancel();
    }}
  >
    {labelOf(corner, rot)}
  </div>
{/each}

<style>
  .corner-handle {
    position: absolute;
    width: 44px;
    height: 44px;
    margin: -22px 0 0 -22px;
    border-radius: 50%;
    background: rgba(255, 207, 51, 0.9);
    border: 2px solid var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 13px;
    color: #14171c;
    touch-action: none;
    -webkit-user-select: none;
    user-select: none;
    -webkit-touch-callout: none;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    z-index: 3;
  }

  .corner-handle:active {
    background: #ffcf33;
  }
</style>
