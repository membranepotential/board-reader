<script lang="ts">
  /**
   * The board card's right-hand drawer for the game panel: the mirror image of
   * `FramePreview` (which docks to the board's left). A vertical tab at the card's
   * right edge stays where the operator clicked it, and the pane grows leftward from
   * it; only the pane's width animates.
   *
   * The panel stays mounted while the drawer is closed (the pane clips it to zero
   * width), so the decoder keeps folding in every read and the game is complete
   * whenever the drawer opens again.
   */
  import type { Snippet } from 'svelte';

  interface Props {
    open: boolean;
    onToggle: () => void;
    children: Snippet;
  }

  const { open, onToggle, children }: Props = $props();
</script>

<div class="drawer" class:open>
  <div class="drawer-pane" inert={!open}>
    <div class="drawer-content">{@render children()}</div>
  </div>
  <button class="drawer-tab" aria-expanded={open} title="show the game (PGN)" onclick={onToggle}>
    <span class="chev">‹</span>game
  </button>
</div>

<style>
  .drawer {
    display: flex;
    align-items: flex-start;
  }
  .drawer-tab {
    writing-mode: vertical-rl;
    text-orientation: mixed;
    width: 26px;
    height: var(--board-size, 360px);
    background: #232833;
    color: #9aa3ad;
    border: 1px solid #2f3540;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    letter-spacing: 0.04em;
    padding: 6px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  .drawer-tab:hover {
    color: #e6e6e6;
    border-color: #3c434f;
  }
  .drawer.open .drawer-tab {
    color: #8fb6e8;
    border-color: #35507a;
  }
  .drawer-tab .chev {
    transition: transform 200ms ease;
  }
  .drawer.open .drawer-tab .chev {
    transform: rotate(180deg);
  }
  /* Width is the only animated axis; the pane clips its contents while it grows. The
     height is the board's, so the panel inside scrolls within it and never makes the
     card taller than the board. */
  .drawer-pane {
    width: 0;
    overflow: hidden;
    transition: width 220ms ease;
    height: var(--board-size, 360px);
  }
  .drawer.open .drawer-pane {
    width: var(--game-width, 260px);
    margin-right: 8px;
  }
  /* A fixed width, so the panel's text does not reflow while the pane animates. */
  .drawer-content {
    width: var(--game-width, 260px);
    height: 100%;
  }
</style>
