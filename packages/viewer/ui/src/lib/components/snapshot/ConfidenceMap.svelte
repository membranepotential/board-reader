<script lang="ts">
  /**
   * 8×8 of what the classifier believes, on the node's own red → yellow → green ramp.
   * Every cell opens that square in the inspector; the already-selected one cycles.
   */
  import { squareKey, type BoardGrid } from '$lib/board/fen';
  import { FILE_LETTERS, GRID, ROW_RANKS, type SquareDist } from '$lib/protocol';
  import { CLASS_NAMES, confidenceOf, fmt, heatColor, scoresAt, shownClassAt } from './reading';
  import type { Corrections } from './reading';

  interface Props {
    dist: SquareDist;
    /** The node's read, uncorrected — corrections are layered for the label only. */
    base: BoardGrid;
    corrections: Corrections;
    selected: { readonly r: number; readonly c: number } | null;
    onPick: (r: number, c: number) => void;
  }

  const { dist, base, corrections, selected, onPick }: Props = $props();

  const rows = $derived(
    Array.from({ length: GRID }, (_, r) =>
      Array.from({ length: GRID }, (_, c) => {
        const cls = shownClassAt(base, corrections, r, c);
        return {
          r,
          c,
          square: squareKey(r, c),
          p: confidenceOf(scoresAt(dist, r, c)),
          symbol: cls === 'empty' ? '.' : cls,
          name: CLASS_NAMES[cls]
        };
      })
    )
  );
</script>

<div class="confmap" data-keeps-inspector>
  {#each rows as row, r (r)}
    <div class="lbl">{ROW_RANKS[r]}</div>
    {#each row as cell (cell.square)}
      <button
        class="cell"
        class:sel={selected?.r === cell.r && selected?.c === cell.c}
        type="button"
        style:background={heatColor(cell.p)}
        title="{cell.square} — {cell.name} {fmt(cell.p * 100, 1)}%"
        onclick={() => onPick(cell.r, cell.c)}
      >
        <span>{cell.symbol}</span>
        <span class="p">{Math.round(cell.p * 100)}</span>
      </button>
    {/each}
  {/each}
  <div class="lbl"></div>
  {#each FILE_LETTERS as file (file)}
    <div class="lbl">{file}</div>
  {/each}
</div>

<style>
  .confmap {
    display: grid;
    grid-template-columns: 1.4em repeat(8, 1fr);
    gap: 2px;
    margin: 10px 0;
    font-size: 10px;
  }

  .lbl {
    color: var(--dim);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .cell {
    aspect-ratio: 1;
    border-radius: 3px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    line-height: 1.05;
    font-family: ui-monospace, monospace;
    font-size: 10px;
    color: #0d0f12;
    font-weight: 700;
    border: 1px solid transparent;
    padding: 0;
    touch-action: manipulation;
    cursor: pointer;
  }

  .cell.sel {
    border-color: #fff;
  }

  .p {
    font-size: 8px;
    font-weight: 500;
    opacity: 0.8;
  }
</style>
