<script lang="ts">
  /**
   * The card both the live page (`live/BoardCard.svelte`) and /control
   * (`control/CameraCard.svelte`) lay one board or one camera out in: the panel, its
   * stale look and the heading row. Presentational only: what goes in the heading
   * (`heading`) and below it (`children`) is the caller's.
   *
   * The root carries the caller's `class` too, so a caller that styles the card's
   * state from outside (`:global(.board-card.stale) .board-wrap`) has a name to aim at.
   */
  import type { Snippet } from 'svelte';
  import type { HTMLAttributes } from 'svelte/elements';

  interface Props extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
    /** The reading is old: the card stops asserting itself. */
    stale?: boolean;
    /** The heading row: a label on the left, chips on the right. */
    heading: Snippet;
    children: Snippet;
  }

  const { stale = false, heading, children, class: className, ...rest }: Props = $props();
</script>

<div {...rest} class={['card', className]} class:stale>
  <h2>{@render heading()}</h2>
  {@render children()}
</div>

<style>
  .card {
    background: #1b1f26;
    border: 1px solid #272b33;
    border-radius: 10px;
    padding: 14px;
    min-width: 0;
  }
  .card.stale {
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
</style>
