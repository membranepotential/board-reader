<script lang="ts">
  /**
   * Every square where the operator disagreed with the node, as `read → assigned`.
   *
   * The list reads off the correction map, which only ever holds genuine disagreements
   * (assigning the model's own read clears the entry), so an empty list is a positive
   * statement: the board below is exactly what the model read.
   */
  import type { ClassName, SquareKey } from '#lib/protocol/index.ts';
  import { CLASS_NAMES, type Corrections } from './reading';

  interface Props {
    corrections: Corrections;
    /** What the node read on that square — resolved by the page, which holds the dist. */
    readAt: (square: SquareKey) => ClassName;
    onRevert: (square: SquareKey) => void;
  }

  const { corrections, readAt, onRevert }: Props = $props();

  const entries = $derived([...corrections.entries()]);
</script>

{#if entries.length === 0}
  <div class="note">none — the board below is exactly what the model read</div>
{:else}
  <div class="list">
    {#each entries as [square, cls] (square)}
      <div class="row">
        <span class="mono">{square}</span>
        <span class="muted">{CLASS_NAMES[readAt(square)]} &rarr; {CLASS_NAMES[cls]}</span>
        <span class="spacer"></span>
        <button class="text-btn" type="button" onclick={() => onRevert(square)}>revert</button>
      </div>
    {/each}
  </div>
{/if}

<style>
  .note {
    font-size: 12.5px;
    color: var(--muted);
    margin: 8px 0;
  }

  .list {
    margin: 8px 0;
    font-size: 13px;
  }

  .row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid var(--line);
  }

  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: #c8cdd4;
  }

  .muted {
    color: var(--muted);
  }

  .spacer {
    flex: 1;
  }

  .text-btn {
    background: none;
    border: 0;
    color: var(--accent-soft);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    padding: 4px 0;
    text-decoration: underline;
    touch-action: manipulation;
    cursor: pointer;
  }
</style>
