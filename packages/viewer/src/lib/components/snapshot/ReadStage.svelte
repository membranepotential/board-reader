<script lang="ts">
  /** The body of the read stage: the node's per-square distribution and its weak spots. */
  import type { BoardState } from '#lib/components/snapshot/board-state.svelte.ts';
  import ConfidenceMap from '#lib/components/snapshot/ConfidenceMap.svelte';
  import KvTable from '#lib/components/snapshot/KvTable.svelte';
  import { fmt, shownClassAt, weakestSquares } from '#lib/components/snapshot/reading.ts';
  import { readTable } from '#lib/components/snapshot/tables.ts';

  interface Props {
    board: BoardState;
    estimateMs: number | null;
  }

  const { board, estimateMs }: Props = $props();

  const rows = $derived(readTable(board.reading, estimateMs));
  const chips = $derived.by(() => {
    const { dist, baseGrid, corrections } = board;
    if (dist === null || baseGrid === null) return [];
    return weakestSquares(dist).map((entry) => {
      const cls = shownClassAt(baseGrid, corrections, entry.r, entry.c);
      return { ...entry, symbol: cls === 'empty' ? '.' : cls };
    });
  });
</script>

<p class="note">
  A 13-class distribution per square, read from the whole photo and always in the chess
  frame (rank 8 on top), stateless and rule-free — an illegal-looking board is still
  reported as <code>ok</code>. Click any cell to open it in the inspector.
</p>
<KvTable {rows} />
{#if board.dist !== null && board.baseGrid !== null}
  <ConfidenceMap
    dist={board.dist}
    base={board.baseGrid}
    corrections={board.corrections}
    selected={board.selected}
    onPick={(r, c) => board.pickSquare(r, c)}
  />
  <div class="note">Least confident squares</div>
  <div class="chips" data-keeps-inspector>
    {#each chips as chip (chip.square)}
      <button class="chip" type="button" onclick={() => board.selectSquare(chip.r, chip.c)}>
        {chip.square}
        {chip.symbol}
        {fmt(chip.p, 2)}
      </button>
    {/each}
  </div>
{/if}
{#if (board.reading?.warnings.length ?? 0) > 0}
  <div class="note warn">Server warnings: {board.reading?.warnings.join(', ')}</div>
{/if}

<style>
  .note {
    font-size: 12.5px;
    color: var(--muted);
    margin: 8px 0;
  }

  .note.warn {
    color: var(--warn);
  }

  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
  }

  .chip {
    font-family: ui-monospace, monospace;
    font-size: 12px;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 4px 8px;
    color: var(--fg);
    min-height: 32px;
    font-weight: 500;
    touch-action: manipulation;
    cursor: pointer;
  }
</style>
