<script lang="ts">
  /**
   * The one chessground wrapper. All three pages render boards; before the rewrite each
   * built its own instance by hand and each carried the same two ordering hazards, so
   * both are solved here once:
   *
   *  1. **The element must have its real size before `Chessground()` runs.** Chessground
   *     measures the box it is handed and sizes `cg-container` from it, so constructing
   *     against a zero-sized or not-yet-laid-out element yields a board that never
   *     recovers. `$effect` runs after the element is in the DOM, and the wrapper carries
   *     an explicit square box driven by `--board-size` rather than a percentage through
   *     an auto-height ancestor.
   *
   *  2. **One instance, updated in place — never re-created per event.** This is the
   *     load-bearing one, and it is easy to lose: the creation effect must depend on
   *     `host` and *nothing else*. Reading `placement` inside it (as an earlier version
   *     of this file did) makes every position change destroy and rebuild the board —
   *     at up to `/camera`'s 12 fps burst rate. That is not merely wasteful:
   *     `api.destroy()` does not disconnect the `ResizeObserver` chessground binds to
   *     the *host* element, and the host survives the rebuild, so each cycle leaks one
   *     observer onto a live element. It also throws away chessground's piece animation,
   *     so pieces teleport instead of sliding. Hence `untrack` around the initial config:
   *     the values are read once, for the first paint, and the effect stays keyed to the
   *     element alone.
   *
   * Consumers pass a *sanitized* FEN placement (see `$lib/board/fen`), never a raw wire
   * string: chessground silently skips symbols it does not recognise, which shifts every
   * later square in the rank instead of failing, so one stray byte would redraw the board
   * wrong with no error anywhere.
   *
   * The prop surface is deliberately two values. Orientation, coordinates, view-only and
   * an escape-hatch config were all offered once and used by nobody — every call site
   * wants the same read-only white-oriented board — so they are gone rather than left as
   * options with no caller.
   */
  import { Chessground } from '@lichess-org/chessground';
  import type { Api } from '@lichess-org/chessground/api';
  import type { DrawShape } from '@lichess-org/chessground/draw';
  import { untrack } from 'svelte';

  interface Props {
    /** Board-placement field of a FEN — ranks only, no side-to-move. */
    placement: string;
    /** Per-square rings: low confidence, just-changed, selection. */
    shapes?: DrawShape[];
  }

  const { placement, shapes = [] }: Props = $props();

  let host: HTMLDivElement | undefined = $state();
  let api: Api | undefined = $state();

  // Keyed to the element only — see hazard 2 above.
  $effect(() => {
    const el = host;
    if (!el) return;
    const instance = untrack(() =>
      Chessground(el, {
        fen: placement,
        orientation: 'white',
        coordinates: true,
        // These boards report a reading; they are not played on.
        viewOnly: true,
        // The drawable layer stays enabled even when view-only: the rings are how a
        // reading states its own uncertainty, and they are drawn as autoShapes.
        drawable: { enabled: true, visible: true, autoShapes: [] }
      })
    );
    api = instance;
    return () => {
      api = undefined;
      instance.destroy();
    };
  });

  // Split from the shape effect so a placement change does not repaint rings and a ring
  // change does not re-diff the position.
  $effect(() => {
    api?.set({ fen: placement });
  });

  $effect(() => {
    api?.setAutoShapes(shapes);
  });
</script>

<div class="board" bind:this={host}></div>

<style>
  /* Explicit square box, never a percentage: chessground reads this element's measured
     size at construction. `--board-size` is set by the page. */
  .board {
    width: var(--board-size, 360px);
    height: var(--board-size, 360px);
  }
</style>
