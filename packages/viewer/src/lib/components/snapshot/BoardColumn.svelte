<script lang="ts">
  /** The position read, with the legend for its rings and the ground-truth action. */
  import type { DrawShape } from '@lichess-org/chessground/draw';
  import Board from '#lib/board/Board.svelte';
  import GroundTruthForm from '#lib/components/snapshot/GroundTruthForm.svelte';
  import type { LabelState } from '#lib/components/snapshot/reading.ts';
  import { GRID, LOW_CONFIDENCE } from '#lib/protocol/index.ts';

  interface Props {
    placement: string;
    shapes: DrawShape[];
    /** A tap on a square, as row and column from the top left. */
    onPick: (r: number, c: number) => void;
    groundTruth: LabelState;
    corrections: number;
    submittingLabel: boolean;
    onSave: () => void;
  }

  const { placement, shapes, onPick, groundTruth, corrections, submittingLabel, onSave }: Props =
    $props();

  let column: HTMLDivElement | undefined = $state();
  let boardSize = $state(360);

  $effect(() => {
    const el = column;
    if (!el) return;
    // chessground measures the box it is handed, so the board gets an explicit px size
    // and heals itself from there via its own ResizeObserver.
    const measure = () => (boardSize = Math.max(160, Math.min(460, el.clientWidth)));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  });

  /**
   * Tap = tap, not touch-capture. A plain click listener never touches `touch-action` or
   * calls `preventDefault`, so a swipe that starts on the board still scrolls the page;
   * only an actual (non-dragged) tap fires a click at all. chessground stays `viewOnly`,
   * which is also why the square has to be worked out from the rect rather than from its
   * `select` event — `viewOnly` binds no interaction handlers at all.
   */
  function boardClick(event: MouseEvent): void {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return;
    onPick(
      Math.min(7, Math.max(0, Math.floor((y / rect.height) * GRID))),
      Math.min(7, Math.max(0, Math.floor((x / rect.width) * GRID)))
    );
  }
</script>

<div class="board-col" bind:this={column}>
  <div class="col-label">Position read</div>
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div class="board-wrap" style:--board-size="{boardSize}px" data-keeps-inspector onclick={boardClick}>
    <Board {placement} {shapes} />
  </div>
  <div class="board-legend">
    <span><i style="background:#e25555"></i> confidence &lt; {LOW_CONFIDENCE}</span>
    <span><i style="background:#6aa8ff"></i> corrected by hand</span>
    <span><i style="background:#7ac77a"></i> inspecting</span>
  </div>
  <p class="hint">
    Click a square to inspect it; click it again to toggle through its most likely readings.
  </p>
  <GroundTruthForm state={groundTruth} {corrections} submitting={submittingLabel} {onSave} />
</div>

<style>
  .board-col {
    flex: 1 1 340px;
    max-width: 440px;
    min-width: 0;
  }

  .col-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--dim);
    margin-bottom: 6px;
  }

  /* viewOnly means no piece dragging, so touch may pan — a thumb landing on the board
     must still scroll the page. */
  .board-wrap {
    width: 100%;
    touch-action: manipulation;
  }

  .board-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 12px;
    color: var(--muted);
    margin: 8px 0;
  }

  .board-legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .board-legend i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }

  .hint {
    color: var(--muted);
    font-size: 13px;
    margin: 6px 0;
  }
</style>
