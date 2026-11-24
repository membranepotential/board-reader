<script lang="ts">
  /**
   * The one place the chessground stylesheets are pulled in. They used to be three
   * `<link>` tags repeated in every page, pointing at a vendored copy of chessground 9
   * under `static/vendor/`; the bundler is the reason that copy existed, so it is gone
   * and these come from the package (`@lichess-org/chessground` — the unscoped
   * `chessground` package is deprecated).
   *
   * Vite fingerprints and inlines them into each prerendered page, so no page can drift
   * from another on which board theme it loaded.
   */
  import '@lichess-org/chessground/assets/chessground.base.css';
  import '@lichess-org/chessground/assets/chessground.brown.css';
  import '@lichess-org/chessground/assets/chessground.cburnett.css';

  import type { Snippet } from 'svelte';

  const { children }: { children: Snippet } = $props();
</script>

{@render children()}

<style>
  /* Shared shell only. Each page owns its own layout — the live page is a card grid,
     /snapshot a vertical pipeline and /camera a full-screen viewfinder — so anything
     more opinionated than this belongs to the page, not here. */
  :global(:root) {
    color-scheme: dark;
  }

  :global(*) {
    box-sizing: border-box;
  }

  :global(body) {
    margin: 0;
    font: 15px/1.4 system-ui, sans-serif;
    background: #14171c;
    color: #e6e6e6;
  }
</style>
