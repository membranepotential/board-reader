<script lang="ts">
  /**
   * One square: what the model saw, what it believed, and what it should have said.
   *
   * Wide enough, it is a panel beside the board so a square and its candidates are in
   * one glance; on a phone it docks as a bottom sheet above the fixed action bar. Plain
   * taps only — nothing here is a drag surface, so page scrolling is never at risk.
   *
   * Assigning the model's own read is not a correction (see `assignmentFor`), so the
   * panel can be walked square by square without inventing disagreements.
   */
  import type { ClassName, ClassScores } from '$lib/protocol';
  import {
    CLASS_NAMES,
    PANEL_TYPES,
    ROLE_NAMES,
    argmaxClass,
    classIndex,
    fmt,
    pct,
    rankedClasses
  } from './reading';

  interface Props {
    square: string;
    scores: ClassScores;
    /** What the board shows here right now — the correction if there is one. */
    shown: ClassName;
    corrected: boolean;
    /**
     * The crop the classifier actually saw, as a cell of the debug mosaic. The mosaic is
     * a uniform 8×8 tiling of equal cells, so percentage background positioning lands
     * exactly on one cell.
     */
    mosaic: { readonly url: string; readonly row: number; readonly col: number } | null;
    onAssign: (cls: ClassName) => void;
    onRevert: () => void;
    onClose: () => void;
    onStep: (delta: number) => void;
    onNextLow: () => void;
  }

  const { square, scores, shown, corrected, mosaic, onAssign, onRevert, onClose, onStep, onNextLow }: Props =
    $props();

  const read = $derived(argmaxClass(scores));
  const ranked = $derived(rankedClasses(scores).slice(0, 5));
  const probability = (cls: ClassName): number => scores[classIndex(cls)] ?? 0;
  const bar = (p: number): string => `${Math.min(100, p * 100).toFixed(0)}%`;
</script>

<!-- Marks the subtree the document-level "click outside closes" check must not close on. -->
<div class="inspector" data-keeps-inspector>
  <div class="head">
    <h3>Square <span class="sq">{square}</span></h3>
    <span class="spacer"></span>
    <button class="text-btn" type="button" onclick={onClose}>Close</button>
  </div>

  <div class="read">
    <div
      class="crop"
      style:background-image={mosaic ? `url("${mosaic.url}")` : 'none'}
      style:background-position={mosaic
        ? `${(mosaic.col / 7) * 100}% ${(mosaic.row / 7) * 100}%`
        : '0 0'}
    ></div>
    <div class="facts">
      <div>
        <span class="k">model read</span>
        {CLASS_NAMES[read]} ({read}) at {fmt(probability(read) * 100, 1)}%
      </div>
      <div>
        <span class="k">shown now</span>
        {#if corrected}
          <span class="corrected">{CLASS_NAMES[shown]} — assigned by hand</span>
        {:else}
          {CLASS_NAMES[shown]} — as read
        {/if}
      </div>
      {#if corrected}
        <div>
          <button class="text-btn" type="button" onclick={onRevert}>Revert to model read</button>
        </div>
      {/if}
    </div>
  </div>

  <!-- Empty vs piece is the most common correction, so it gets its own full-width row
       above the 12-piece grid rather than another cell in it. -->
  <div class="row-label">Assign</div>
  <button
    class="empty-row"
    class:selected={shown === 'empty'}
    type="button"
    onclick={() => onAssign('empty')}
  >
    <span>Empty</span>
    <span class="bar-track"><i style:width={bar(probability('empty'))}></i></span>
    <span class="pct">{pct(probability('empty'))}</span>
  </button>

  {#each [{ colour: 'white', label: 'White' }, { colour: 'black', label: 'Black' }] as side (side.colour)}
    <div class="row-label">{side.label}</div>
    <!-- .cg-wrap: chessground's piece sprites are declared under it, and these are the
         same sprites the board uses. -->
    <div class="grid cg-wrap">
      {#each PANEL_TYPES as type (type)}
        {@const cls = (side.colour === 'white' ? type : type.toLowerCase()) as ClassName}
        {@const p = probability(cls)}
        <button
          class="cell"
          class:faint={p < 0.01}
          class:selected={cls === shown}
          type="button"
          title="{CLASS_NAMES[cls]} — {fmt(p * 100, 1)}%"
          onclick={() => onAssign(cls)}
        >
          <piece class="{side.colour} {ROLE_NAMES[type]}"></piece>
          <span class="bar-track"><i style:width={bar(p)}></i></span>
          <span class="pct">{pct(p)}</span>
          {#if cls === read}
            <span class="argmax-dot" title="the model’s own read"></span>
          {/if}
        </button>
      {/each}
    </div>
  {/each}

  <div class="rank-list">
    {#each ranked as entry (entry.cls)}
      <div class="rank-row">
        <span class="rk-cls">{entry.cls === 'empty' ? '.' : entry.cls}</span>
        <span class="bar-track"><i style:width={bar(entry.p)}></i></span>
        <span class="rk-pct">
          {entry.p * 100 < 0.1 ? '<0.1' : fmt(entry.p * 100, 1)}%
        </span>
      </div>
    {/each}
  </div>

  <div class="nav">
    <button type="button" onclick={() => onStep(-1)}>Prev square</button>
    <button type="button" onclick={() => onStep(1)}>Next square</button>
    <button type="button" onclick={onNextLow}>Next low-confidence</button>
  </div>
  <div class="kbd-hint">
    Keys: <kbd>arrows</kbd> move · <kbd>k q r b n p</kbd> assign black,
    <kbd>shift</kbd>+key assign white · <kbd>.</kbd> empty · <kbd>esc</kbd> close
  </div>
</div>

<style>
  .inspector {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
  }

  @media (max-width: 759px) {
    .inspector {
      position: fixed;
      left: 0;
      right: 0;
      bottom: var(--bar-h);
      z-index: 29;
      background: var(--panel);
      margin: 0;
      border-radius: 12px 12px 0 0;
      border-left: 0;
      border-right: 0;
      max-height: 54vh;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      box-shadow: 0 -6px 20px rgba(0, 0, 0, 0.5);
    }
  }

  .head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }

  h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
  }

  .sq {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--warn);
    font-size: 16px;
    font-weight: 700;
  }

  .spacer {
    flex: 1;
  }

  .read {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }

  .crop {
    width: 64px;
    height: 88px;
    flex: 0 0 auto;
    border-radius: 6px;
    border: 1px solid var(--line);
    background-color: #0f1216;
    background-repeat: no-repeat;
    background-size: 800% 800%;
  }

  .facts {
    font-size: 13px;
    line-height: 1.5;
  }

  .facts .k {
    color: var(--muted);
  }

  .facts .corrected {
    color: var(--accent-soft);
    font-weight: 600;
  }

  .row-label {
    font-size: 11px;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 6px 0 3px;
  }

  .empty-row {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    background: var(--panel-2);
    border: 2px solid transparent;
    border-radius: 9px;
    padding: 8px 12px;
    margin-bottom: 8px;
    min-height: 44px;
    color: var(--fg);
    font: inherit;
    font-size: 14px;
    font-weight: 600;
    touch-action: manipulation;
    cursor: pointer;
  }

  .empty-row.selected {
    border-color: var(--accent-soft);
    background: #232f42;
  }

  .empty-row .bar-track {
    flex: 1;
  }

  /* 2 rows × 6 columns: White on top, Black below, same piece type in the same column —
     the board's own colour symmetry, so a miscoloured read is a vertical glance. */
  .grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 5px;
  }

  .cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    padding: 4px 2px 5px;
    border-radius: 8px;
    border: 2px solid transparent;
    background: var(--panel-2);
    color: inherit;
    font: inherit;
    touch-action: manipulation;
    cursor: pointer;
  }

  .cell.selected {
    border-color: var(--accent-soft);
    background: #232f42;
  }

  .cell.faint {
    opacity: 0.45;
  }

  .cell piece {
    display: block;
    position: static;
    width: 30px;
    height: 30px;
  }

  .bar-track {
    width: 100%;
    height: 5px;
    border-radius: 3px;
    background: #333842;
    overflow: hidden;
    display: block;
  }

  .bar-track > i {
    display: block;
    height: 100%;
    background: var(--accent-soft);
  }

  .pct {
    font-size: 10px;
    color: var(--muted);
  }

  .argmax-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--warn);
  }

  /* The numbers behind the grid: the ranked distribution, all of it. */
  .rank-list {
    margin: 10px 0 0;
    font-size: 12px;
  }

  .rank-row {
    display: grid;
    grid-template-columns: 3.2em 1fr 3.2em;
    gap: 8px;
    align-items: center;
    padding: 2px 0;
  }

  .rk-cls {
    font-family: ui-monospace, monospace;
    color: #c8cdd4;
  }

  .rk-pct {
    text-align: right;
    color: var(--muted);
    font-family: ui-monospace, monospace;
  }

  .nav {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
    flex-wrap: wrap;
  }

  .nav button {
    min-height: 36px;
    padding: 6px 10px;
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    background: #333842;
    color: #fff;
    border: 0;
    border-radius: 9px;
    touch-action: manipulation;
    cursor: pointer;
  }

  /* Occasional side actions, deliberately NOT the >=48px treatment. */
  .text-btn {
    background: none;
    border: 0;
    color: var(--accent-soft);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    min-height: auto;
    padding: 4px 0;
    text-decoration: underline;
    touch-action: manipulation;
    cursor: pointer;
  }

  .kbd-hint {
    font-size: 11.5px;
    color: var(--dim);
    margin-top: 8px;
  }

  .kbd-hint kbd {
    font-family: ui-monospace, monospace;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 4px;
  }
</style>
