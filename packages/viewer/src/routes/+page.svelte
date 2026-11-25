<script lang="ts">
  /**
   * board-reader · live positions — the live page.
   *
   * Every board's bookkeeping — the sanitized grid, pending squares, motion,
   * change rings, staleness, events/sec, reconnect-with-backoff — already lives
   * in `PositionStream` (`#lib/stores/positions.svelte`); this page is wiring: one
   * `<BoardCard>` per board, in a stable order (each card holds its game drawer), plus
   * the connection chip.
   *
   * The route is prerendered (`adapter-static`), and `PositionStream`'s
   * constructor reads `performance.now()` and its `start()` opens a socket — both
   * browser-only — so the store is built inside `onMount`, never at module or
   * component-script scope, where SSR would run it.
   */
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import { VIEWER_SOCKET, wsUrl } from '#lib/api.ts';
  import { PositionStream } from '#lib/stores/positions.svelte.ts';
  import BoardCard from '#lib/components/live/BoardCard.svelte';

  let stream: PositionStream | undefined = $state();
  /** The socket URL this page reads from, shown so an operator can see it. */
  let sourceUrl = $state('');

  onMount(() => {
    const positions = new PositionStream();
    stream = positions;
    sourceUrl = wsUrl(VIEWER_SOCKET);
    positions.start();
    return () => positions.stop();
  });

  const connectionLabel = $derived(
    stream === undefined || stream.connection === 'connecting'
      ? 'connecting…'
      : stream.connection === 'live'
        ? 'live'
        : 'disconnected — retrying'
  );
</script>

<header>
  <h1>board-reader · live positions</h1>
  <span class="conn" class:up={stream?.connection === 'live'}>{connectionLabel}</span>
  <span class="source">source: <code>{sourceUrl}</code></span>
  <a class="control" href={resolve('/control')}>camera control</a>
</header>

<main>
  {#if stream}
    {#each stream.boards as board (board.id)}
      <BoardCard {board} label={stream.label(board.id)} liveness={stream.liveness(board)} />
    {/each}
  {/if}
</main>

<style>
  :global(:root) {
    /* One number drives the whole card: the board box, the preview pane and the
       frame image are all this size, so the preview lines up with the board. */
    --board-size: 360px;
  }
  @media (max-width: 900px) {
    :global(:root) {
      --board-size: 288px;
    }
  }

  header {
    padding: 14px 20px;
    border-bottom: 1px solid #272b33;
    display: flex;
    align-items: baseline;
    gap: 14px;
  }
  h1 {
    font-size: 17px;
    margin: 0;
    font-weight: 600;
  }
  .conn {
    font-size: 13px;
    padding: 2px 8px;
    border-radius: 10px;
    background: #3a2020;
    color: #ff9a9a;
  }
  .conn.up {
    background: #1f3a24;
    color: #97e0a6;
  }
  .source {
    font-size: 12px;
    color: #6b7280;
  }
  .source code {
    color: #6b7280;
  }
  .control {
    margin-left: auto;
    font-size: 13px;
    color: #6aa8ff;
  }

  main {
    display: flex;
    flex-wrap: wrap;
    gap: 22px;
    padding: 20px;
    align-items: flex-start;
  }
</style>
