<script lang="ts">
  /**
   * One board's card on the live page — ported from index.html's `ensureCard` /
   * `render`, minus the DOM bookkeeping: `PositionStream` already derives the
   * sanitized grid, change rings, pending squares, motion, staleness and
   * events/sec (see `#lib/stores/positions.svelte`), so this component only lays
   * them out and owns the UI-only state index.html kept per card (whether the
   * preview is open and the display rotation).
   *
   * What it does NOT show, on purpose: a confidence bar or confidence rings (the
   * recognizer reports ~1.0 on its wrong squares too — /snapshot's inspector keeps
   * the per-square numbers), and the read frame's stability (the node reads after
   * motion ends, so that one always says "at rest"). Motion comes from the node's
   * per-frame motion messages instead, and the chip is absent when the node sends
   * none.
   *
   * Stale = the reading is old, so it stops asserting itself — except the `.live`
   * row, which is exempt: how stale it is, is exactly what you want to read on a
   * stale card. A stream that dies mid-move must also not pulse "moving" forever,
   * so the motion chip's animation is switched off once the card is stale.
   */
  import Board from '#lib/board/Board.svelte';
  import { gridToFen } from '#lib/board/fen.ts';
  import { RATE_WINDOW_MS, STALE_MS, formatAge } from '#lib/stores/liveness.ts';
  import type { BoardLiveness, BoardState } from '#lib/stores/positions.svelte.ts';
  import FramePreview from './FramePreview.svelte';
  import { nextRotation, parseStoredRotation, rotationKey, type BoardRotation } from './rotation';
  import { boardShapes } from './shapes';
  import { untrack } from 'svelte';

  interface Props {
    board: BoardState;
    liveness: BoardLiveness;
  }

  const { board, liveness }: Props = $props();

  let previewOpen = $state(false);

  // `board.id` is this instance's key in the `{#each}` on the live page (never
  // reassigned to a different board), so reading it once at construction is
  // deliberate — `untrack` says so instead of leaving it looking like a missed
  // dependency.
  let rotation = $state<BoardRotation>(
    untrack(() => parseStoredRotation(readLocal(rotationKey(board.id))))
  );

  function readLocal(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null; // private mode: preferences simply do not persist
    }
  }

  function writeLocal(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private mode */
    }
  }

  function rotateBoard(): void {
    rotation = nextRotation(rotation);
    writeLocal(rotationKey(board.id), String(rotation));
  }

  const event = $derived(board.event);
  const placement = $derived(gridToFen(board.grid));
  const shapes = $derived(boardShapes(liveness.rings, board.flipped, board.pending));
  const motion = $derived(liveness.motion);
</script>

<div class="board-card" class:stale={liveness.stale}>
  <h2>
    <span>board {board.id}</span>
    <span class="chips">
      {#if motion.kind === 'moving'}
        <span
          class="motion-chip moving"
          title="the camera sees motion on the board (a hand, or a piece in flight); the next read waits until it stops"
        >
          moving
        </span>
      {:else if motion.kind === 'still'}
        <span class="motion-chip still" title="the camera sees no motion on the board">still</span>
      {:else if motion.kind === 'stale'}
        <span class="motion-chip unknown" title="no motion message from the node for a few seconds">
          motion ?
        </span>
      {/if}
      {#if liveness.stale}
        <span class="stale-chip" title="no event for over {STALE_MS / 1000}s">stale</span>
      {/if}
      <span class="status {event.status}">{event.status}</span>
    </span>
  </h2>

  <div class="card-body">
    <FramePreview
      boardId={board.id}
      seq={event.seq}
      open={previewOpen}
      onToggle={() => (previewOpen = !previewOpen)}
    />

    <div class="board-col">
      <div class="board-wrap" style="transform: rotate({rotation}deg)">
        <Board {placement} {shapes} />
      </div>

      <div class="board-actions">
        <button
          class="rotate"
          title="rotate this board's on-screen picture — display only, saved for this board"
          onclick={rotateBoard}
        >
          ↻ {rotation}°
        </button>
        <span class="legend">
          <span><i class="ch"></i>just changed</span>
          <span><i class="pd"></i>not settled yet</span>
        </span>
      </div>

      <div class="meta">
        <div class="live">
          <span title="age of the last event">{formatAge(liveness.ageMs)}</span> ·
          <span title="events/sec over the last {RATE_WINDOW_MS / 1000}s">
            {liveness.eventsPerSecond.toFixed(1)} ev/s
          </span>
          · seq <code>{event.seq}</code>
        </div>
        <div>placement: <code class="pl">{event.placement}</code></div>
        <div>utc <code>{event.utc}</code> · clock <code>{event.clock.running}</code></div>
      </div>
    </div>
  </div>
</div>

<style>
  .board-card {
    background: #1b1f26;
    border: 1px solid #272b33;
    border-radius: 10px;
    padding: 14px;
  }
  .board-card.stale {
    border-color: #4a3a22;
  }
  h2 {
    font-size: 14px;
    margin: 0 0 10px;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
  }
  .chips {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .status,
  .motion-chip,
  .stale-chip {
    font-size: 12px;
    padding: 1px 7px;
    border-radius: 9px;
    white-space: nowrap;
  }
  .status {
    background: #243;
    color: #9e9;
  }
  .status.recalibrating,
  .status.needs_setup,
  .status.stale {
    background: #432;
    color: #fc8;
  }
  .motion-chip {
    background: #23303f;
    color: #8fb6e8;
  }
  .motion-chip.moving {
    background: #3a2f18;
    color: #ffcf7a;
    animation: pulse 1.1s ease-in-out infinite;
  }
  .motion-chip.unknown {
    background: #2b2f37;
    color: #9aa3ad;
  }
  .stale-chip {
    background: #3a2020;
    color: #ff9a9a;
  }
  /* A stream that dies mid-move must not pulse "moving" forever. */
  .board-card.stale .motion-chip {
    opacity: 0.45;
    animation: none;
  }
  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }

  /* ---- card body: [preview][board column] ------------------------------------
     The preview docks to the board's left (see FramePreview.svelte); this is just
     the two-column layout that holds it and the board next to each other. */
  .card-body {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  /* Stale = the reading is old, so it stops asserting itself. The `.live` row is
     exempt: how stale it is, is exactly what you want to read on a stale card. */
  .board-wrap,
  .meta > :not(.live) {
    transition: filter 150ms linear;
  }
  .board-card.stale .board-wrap,
  .board-card.stale .meta > :not(.live) {
    filter: saturate(0.15) brightness(0.62);
  }
  /* Square box: rotating it in 90° steps never changes its bounding box, so this
     never fights the layout above. Own rule so the transform transition is scoped
     to the board and does not also apply to the .meta rows sharing the filter one. */
  .board-wrap {
    transition:
      filter 150ms linear,
      transform 200ms ease;
  }

  .board-actions {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    flex-wrap: wrap;
    /* Unconstrained, this row grows to fit its widest content instead of wrapping —
       and `.board-wrap` above it is `width: auto`, so it grows to match and carries
       the extra width into its own rotate transform. Same cap `.meta`
       already use, so a third legend chip wraps onto its own line instead of
       widening the whole card. */
    max-width: var(--board-size, 360px);
  }
  .rotate {
    font: inherit;
    font-size: 13px;
    padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid #35404e;
    background: #1c222b;
    color: #9aa3ad;
    cursor: pointer;
  }
  .rotate:hover {
    background: #232b36;
    color: #c8cdd4;
  }
  .legend {
    font-size: 12px;
    color: #7d8590;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .legend i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    vertical-align: -1px;
    margin-right: 4px;
  }
  .legend .ch {
    background: #5ad17a;
  }
  .legend .pd {
    box-sizing: border-box;
    background: none;
    border: 1.5px dashed #e2a03f;
  }

  .meta {
    margin-top: 10px;
    font-size: 12px;
    color: #9aa3ad;
    display: grid;
    gap: 4px;
    max-width: var(--board-size, 360px);
  }
  .live {
    color: #c8cdd4;
    font-variant-numeric: tabular-nums;
  }
  .pl {
    word-break: break-all;
  }
  code {
    color: #c8cdd4;
  }
</style>
