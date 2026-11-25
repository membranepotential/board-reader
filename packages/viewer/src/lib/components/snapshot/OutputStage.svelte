<script lang="ts">
  /** The body of the output stage: the position as shown, its corrections, and where to take it. */
  import type { BoardState } from '#lib/components/snapshot/board-state.svelte.ts';
  import CorrectionsPanel from '#lib/components/snapshot/CorrectionsPanel.svelte';
  import KvTable from '#lib/components/snapshot/KvTable.svelte';
  import {
    argmaxClass,
    fullFen,
    indexOfSquare,
    lichessUrls,
    rawEventJson,
    scoresAt,
    type SideToMove
  } from '#lib/components/snapshot/reading.ts';
  import { materialTable } from '#lib/components/snapshot/tables.ts';

  interface Props {
    board: BoardState;
    stm: SideToMove;
  }

  let { board, stm = $bindable() }: Props = $props();

  const links = $derived(
    board.shownPlacement === null ? null : lichessUrls(board.shownPlacement, stm)
  );
  const materialRows = $derived(board.shownGrid === null ? [] : materialTable(board.shownGrid));
  const rawEvent = $derived(board.reading === null ? '' : rawEventJson(board.reading));

  async function copy(text: string | null): Promise<void> {
    if (!text) return;
    try {
      await navigator.clipboard?.writeText(text);
    } catch {
      /* a refused clipboard is not worth an error banner — the text is on screen */
    }
  }

  function readAt(square: Parameters<typeof indexOfSquare>[0]) {
    const at = indexOfSquare(square);
    const scores = scoresAt(board.dist, at.r, at.c);
    return scores === null ? 'empty' : argmaxClass(scores);
  }
</script>

<code class="fen">{board.shownPlacement ?? ''}</code>
<div class="row">
  <button
    class="text-btn"
    type="button"
    onclick={() =>
      copy(board.shownPlacement === null ? null : fullFen(board.shownPlacement, stm))}
  >
    Copy FEN
  </button>
  <button class="text-btn" type="button" onclick={() => copy(rawEvent)}>Copy raw event</button>
  {#if board.corrections.size}
    <button class="text-btn" type="button" onclick={() => board.resetCorrections()}>
      Reset corrections
    </button>
  {/if}
</div>
{#if board.reading !== null}
  <div class="stm">
    to move:
    <label><input type="radio" bind:group={stm} value="w" /> White</label>
    <label><input type="radio" bind:group={stm} value="b" /> Black</label>
  </div>
  {#if links !== null}
    <div class="links">
      <a href={links.analysis} target="_blank" rel="noopener">Open in Lichess analysis</a>
      <a class="secondary" href={links.editor} target="_blank" rel="noopener">Lichess editor</a>
    </div>
  {/if}
  <div class="note">Corrections</div>
  <CorrectionsPanel
    corrections={board.corrections}
    {readAt}
    onRevert={(square) => board.revertCorrection(square)}
  />
  <KvTable rows={materialRows} />
{/if}
<div class="note">Raw event</div>
<pre class="raw">{rawEvent}</pre>

<style>
  .note {
    font-size: 12.5px;
    color: var(--muted);
    margin: 8px 0;
  }

  .row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin: 10px 0;
  }

  /* Occasional side actions, deliberately NOT the >=48px treatment. */
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

  .fen {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: #c8cdd4;
    word-break: break-all;
    -webkit-user-select: text;
    user-select: text;
    background: #0f1216;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 8px;
    margin: 8px 0;
    display: block;
  }

  .raw {
    background: #0f1216;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    line-height: 1.4;
    max-height: 320px;
    overflow: auto;
    -webkit-user-select: text;
    user-select: text;
    margin: 8px 0;
  }

  .stm {
    margin: 10px 0;
    font-size: 13px;
    color: var(--muted);
  }

  .stm label {
    margin-right: 16px;
    padding: 6px 0;
    display: inline-block;
    touch-action: manipulation;
    color: var(--fg);
  }

  .links a {
    display: inline-flex;
    align-items: center;
    min-height: 44px;
    background: #1f3a24;
    color: var(--ok);
    text-decoration: none;
    padding: 8px 14px;
    border-radius: 9px;
    margin: 6px 8px 0 0;
    font-weight: 600;
    font-size: 14px;
    touch-action: manipulation;
  }

  .links a.secondary {
    background: #333842;
    color: #c8cdd4;
  }
</style>
