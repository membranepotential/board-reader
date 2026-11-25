<script lang="ts">
  /**
   * One board's card on the live page — ported from index.html's `ensureCard` /
   * `render`, minus the DOM bookkeeping: `PositionStream` already derives the
   * sanitized grid, change rings, pending squares, motion, staleness and
   * events/sec (see `#lib/stores/positions.svelte`), so this component only lays
   * them out and owns the one UI-only state index.html kept per card: whether the
   * preview is open.
   *
   * What it does NOT show, on purpose: a confidence bar or confidence rings (the
   * recognizer reports ~1.0 on its wrong squares too — /snapshot's inspector keeps
   * the per-square numbers), and the read frame's stability (the node reads after
   * motion ends, so that one always says "at rest"). Motion comes from the node's
   * per-frame motion messages instead, and the chip is absent when the node sends
   * none. The "hand?" chip is the latest read's `obscured` flag: the node judged
   * something to cover the board (a hand resting still makes no motion), so that
   * read could settle a piece that moved at most, not one that only vanished.
   *
   * Stale = the reading is old, so it stops asserting itself — except the `.live`
   * row, which is exempt: how stale it is, is exactly what you want to read on a
   * stale card. A stream that dies mid-move must also not pulse "moving" forever,
   * so the motion chip's animation is switched off once the card is stale.
   */
  import Board from '#lib/board/Board.svelte';
  import Card from '#lib/components/Card.svelte';
  import { gridToFen } from '#lib/board/fen.ts';
  import { RATE_WINDOW_MS, STALE_MS, formatAge } from '#lib/stores/liveness.ts';
  import type { BoardLiveness, BoardState } from '#lib/stores/positions.svelte.ts';
  import FramePreview from './FramePreview.svelte';
  import GameDrawer from './GameDrawer.svelte';
  import GamePanel from '#lib/components/game/GamePanel.svelte';
  import { boardShapes } from './shapes';
  import { shortDeviceId } from '#lib/edge/device.ts';

  interface Props {
    board: BoardState;
    /** What people call the board: its camera's name, or its id (DESIGN D18). */
    label: string;
    liveness: BoardLiveness;
  }

  const { board, label, liveness }: Props = $props();

  let previewOpen = $state(false);
  /** Open by default: the game is what a player watches during a game. */
  let gameOpen = $state(true);

  const event = $derived(board.event);
  const placement = $derived(gridToFen(board.grid));
  const shapes = $derived(boardShapes(liveness.rings, board.pending));
  const motion = $derived(liveness.motion);
</script>

<Card class="board-card" stale={liveness.stale}>
  {#snippet heading()}
    <span class="label" data-board={board.id}>
      {#if label === board.id}
        board {board.id}
      {:else}
        {label} <span class="board-id" title="board {board.id}">{shortDeviceId(board.id)}</span>
      {/if}
    </span>
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
      {#if event.obscured}
        <span
          class="obscured-chip"
          title="the last read looks obscured (a hand over the board?), so it could show a piece that moved, not one that vanished"
        >
          hand?
        </span>
      {/if}
      {#if liveness.stale}
        <span class="stale-chip" title="no event for over {STALE_MS / 1000}s">stale</span>
      {/if}
      <span class="status {event.status}">{event.status}</span>
    </span>
  {/snippet}

  <div class="card-body">
    <FramePreview
      boardId={board.id}
      seq={event.seq}
      open={previewOpen}
      onToggle={() => (previewOpen = !previewOpen)}
    />

    <div class="board-col">
      <div class="board-wrap">
        <Board {placement} {shapes} />
      </div>

      <div class="board-actions">
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

    <GameDrawer open={gameOpen} onToggle={() => (gameOpen = !gameOpen)}>
      <GamePanel {event} board={label} />
    </GameDrawer>
  </div>
</Card>

<style>
  .board-id {
    font-weight: 400;
    font-size: 12px;
    color: #6b7280;
  }
  .chips {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .status,
  .motion-chip,
  .obscured-chip,
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
  .obscured-chip {
    background: #3a2f18;
    color: #ffcf7a;
  }
  .stale-chip {
    background: #3a2020;
    color: #ff9a9a;
  }
  /* A stream that dies mid-move must not pulse "moving" forever. */
  :global(.board-card.stale) .motion-chip {
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

  /* ---- card body: [preview][board column][game] ------------------------------
     The preview docks to the board's left (FramePreview.svelte), the game to its
     right (GameDrawer.svelte); this is the row that holds the three. */
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
  :global(.board-card.stale) .board-wrap,
  :global(.board-card.stale) .meta > :not(.live) {
    filter: saturate(0.15) brightness(0.62);
  }
  .board-actions {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    flex-wrap: wrap;
    /* Unconstrained, this row grows to fit its widest content instead of wrapping,
       and `.board-wrap` above it (`width: auto`) grows to match. Same cap `.meta`
       already use, so a third legend chip wraps onto its own line instead of
       widening the whole card. */
    max-width: var(--board-size, 360px);
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
