<script lang="ts">
  /**
   * The live position, as a floating panel over the viewfinder.
   *
   * Rendered as a SIBLING of the stage, never a child: chessground sizes itself
   * from its element's box and re-measures through its OWN `ResizeObserver`, so
   * that box has to stay real even when the stage is gone (no camera). For the same
   * reason the panel is hidden with `visibility`, NEVER `display: none` — a 0x0 box
   * makes chessground's observer fire, and it then computes the container size as
   * 0/0 and writes `NaNpx`.
   *
   * Nothing here tells chessground it was resized. Moving the panel's width is
   * enough; `redrawAll()` would be actively harmful — it tears the board down
   * (`innerHTML = ''`) and re-observes without disconnecting the previous observer,
   * i.e. one leaked observer and one full rebuild per `pointermove`.
   *
   * The gesture has ONE degree of freedom, the width, and the board is a square
   * driven by that single number (`--board-size`). A free two-axis drag would make
   * chessground derive a non-square container and render stretched pieces.
   */
  import { untrack } from 'svelte';
  import Board from '$lib/board/Board.svelte';
  import type { DrawShape } from '@lichess-org/chessground/draw';
  import { placePanel, type PanelState, type PlacedPanel } from './state';

  interface Props {
    panel: PanelState;
    /** The stage's picture area — where the panel parks itself before it is moved. */
    stageWidth: number;
    placement: string;
    shapes: DrawShape[];
    status: string;
    /** Persist: called once a gesture ends, not per pointermove. */
    onCommit: () => void;
  }

  let { panel = $bindable(), stageWidth, placement, shapes, status, onCommit }: Props = $props();

  let element: HTMLDivElement | undefined = $state();
  let placed: PlacedPanel = $state({ left: 0, top: 0, width: panel.width, shown: panel.shown });

  function same(a: PanelState, b: PanelState): boolean {
    return a.left === b.left && a.top === b.top && a.width === b.width && a.shown === b.shown;
  }

  /**
   * Clamp back into the viewport every time, mirroring the snapshot page's loupe: a
   * panel parked in the corner of a landscape phone would otherwise be off-screen
   * in portrait, with no way to reach its own drag strip.
   */
  function place(next: PanelState): void {
    const result = placePanel(
      next,
      { innerWidth: window.innerWidth, innerHeight: window.innerHeight, stageWidth },
      element?.offsetHeight ?? 0
    );
    placed = result;
    if (!same(panel, result)) panel = result;
  }

  // Only `stageWidth` is tracked; `panel` is read untracked so writing the clamped
  // value back cannot re-enter this effect.
  $effect(() => {
    void stageWidth;
    place(untrack(() => panel));
  });

  let drag: { readonly id: number; readonly dx: number; readonly dy: number } | null = null;
  let resize: { readonly id: number; readonly x: number; readonly width: number } | null = null;
</script>

<!-- A rotation changes the viewport under a panel that was parked against an edge. -->
<svelte:window onresize={() => place(panel)} />

<div
  class="pos-panel"
  class:hidden={!placed.shown}
  bind:this={element}
  style:left="{placed.left}px"
  style:top="{placed.top}px"
  style:width="{placed.width}px"
>
  <!-- The drag strip: a pointer gesture that moves a floating panel, with no
       keyboard equivalent to declare. -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="pos-head"
    onpointerdown={(event) => {
      event.preventDefault();
      drag = { id: event.pointerId, dx: event.clientX - placed.left, dy: event.clientY - placed.top };
      event.currentTarget.setPointerCapture(event.pointerId);
    }}
    onpointermove={(event) => {
      if (drag === null || event.pointerId !== drag.id) return;
      place({ ...panel, left: event.clientX - drag.dx, top: event.clientY - drag.dy });
    }}
    onpointerup={() => {
      if (drag === null) return;
      drag = null;
      onCommit();
    }}
    onpointercancel={() => {
      drag = null;
    }}
  >
    <span>position</span><span class="spacer"></span>
    <span class="st">{status}</span>
  </div>

  <div class="board-wrap" style:--board-size="{placed.width}px">
    <Board {placement} {shapes} />
  </div>

  <div
    class="pos-grip"
    aria-hidden="true"
    onpointerdown={(event) => {
      event.preventDefault();
      event.stopPropagation();
      resize = { id: event.pointerId, x: event.clientX, width: placed.width };
      event.currentTarget.setPointerCapture(event.pointerId);
    }}
    onpointermove={(event) => {
      if (resize === null || event.pointerId !== resize.id) return;
      // ...and nothing else: chessground re-measures through its own observer.
      place({ ...panel, width: resize.width + (event.clientX - resize.x) });
    }}
    onpointerup={() => {
      if (resize === null) return;
      resize = null;
      onCommit();
    }}
    onpointercancel={() => {
      resize = null;
    }}
  ></div>
</div>

<style>
  .pos-panel {
    position: fixed;
    z-index: 30;
    touch-action: none;
    background: rgba(27, 31, 38, 0.94);
    border-radius: 10px;
    /* The outline is an INSET shadow rather than a border: `--board-size` is the
       panel's own width, so a 1px border either side would leave the board two
       pixels wider than the box it sits in. A shadow takes no layout. */
    box-shadow:
      0 8px 26px rgba(0, 0, 0, 0.55),
      inset 0 0 0 1px var(--line);
  }

  .pos-panel.hidden {
    visibility: hidden;
    pointer-events: none;
  }

  .pos-head {
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 32px;
    padding: 4px 10px;
    font-size: 12px;
    color: var(--muted);
    touch-action: none;
    cursor: move;
  }

  .pos-head .spacer {
    flex: 1;
  }

  .pos-head .st {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .board-wrap {
    width: 100%;
    touch-action: manipulation;
    -webkit-user-select: none;
    user-select: none;
  }

  /* >=44px, and pointer-driven: CSS `resize` is unusable on touch. */
  .pos-grip {
    position: absolute;
    right: 0;
    bottom: 0;
    width: 44px;
    height: 44px;
    touch-action: none;
    cursor: nwse-resize;
    background: linear-gradient(
      135deg,
      transparent 58%,
      var(--accent-soft) 58%,
      var(--accent-soft) 66%,
      transparent 66%,
      transparent 74%,
      var(--accent-soft) 74%,
      var(--accent-soft) 82%,
      transparent 82%
    );
    border-bottom-right-radius: 10px;
    opacity: 0.75;
  }
</style>
