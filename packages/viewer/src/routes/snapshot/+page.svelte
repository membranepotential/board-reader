<script lang="ts">
  /**
   * `/snapshot` — read one photo, all the way down.
   *
   * This is the **pipeline inspector**: the diagnostic surface for the reading algorithm.
   * One photo goes in and the page shows every stage that turned it into a position —
   * capture, read, output — each with its own state, headline and full
   * evidence. The recognizer finds the board and its orientation in the whole photo,
   * so there are no corners to set and the read is always in the chess frame (rank 8
   * on top).
   *
   * The state lives in `SnapshotSession`; this file wires it to the layout. The node's
   * read is never mutated: hand corrections are a separate layer over it (`BoardState`).
   *
   * Prerendered: nothing below touches a browser API at module scope or during SSR —
   * the canvases, observers and document listeners all live in `$effect`.
   */
  import { onMount } from 'svelte';
  import ActionBar from '#lib/components/snapshot/ActionBar.svelte';
  import BoardColumn from '#lib/components/snapshot/BoardColumn.svelte';
  import HistoryList from '#lib/components/snapshot/HistoryList.svelte';
  import PhotoColumn from '#lib/components/snapshot/PhotoColumn.svelte';
  import PhotoInput from '#lib/components/snapshot/PhotoInput.svelte';
  import PipelineSteps from '#lib/components/snapshot/PipelineSteps.svelte';
  import SquareInspector from '#lib/components/snapshot/SquareInspector.svelte';
  import StatusBanner from '#lib/components/snapshot/StatusBanner.svelte';
  import { historyNote, historyRows } from '#lib/components/snapshot/history.ts';
  import { listenInspector } from '#lib/components/snapshot/inspector-input.ts';
  import { SnapshotSession } from '#lib/components/snapshot/session.svelte.ts';
  import { EMPTY_PLACEMENT } from '#lib/protocol/index.ts';

  const session = new SnapshotSession();
  const board = session.board;

  let barHeight = $state(64);

  const rows = $derived(historyRows(session.reads, session.uploads));

  // The node's retained photos are listed before the first read.
  onMount(() => {
    void session.refreshUploads();
  });

  $effect(() => listenInspector(board));
</script>

<svelte:head>
  <title>board-reader · pipeline inspector</title>
</svelte:head>

<div class="page" style:--bar-h="{barHeight}px">
  <header>
    <h1>board-reader · pipeline inspector</h1>
    <span class="spacer"></span>
    <span class="run-state">{session.runState}</span>
  </header>

  <main>
    <PhotoInput hasPhoto={session.image !== null} onFile={(file) => void session.loadFile(file)} />

    {#if session.banner !== null}
      <StatusBanner kind={session.banner.kind} text={session.banner.text} />
    {/if}
    {#if board.lowLight}
      <StatusBanner kind="warn" text="Photo is very dark — reads degrade; add light if you can." />
    {/if}

    <div class="out-main">
      {#if session.image !== null}
        <PhotoColumn dataUrl={session.image.dataUrl} />
      {/if}

      {#if board.reading !== null}
        <BoardColumn
          placement={board.shownPlacement ?? EMPTY_PLACEMENT}
          shapes={board.shapes}
          onPick={(r, c) => board.pickSquare(r, c)}
          groundTruth={session.groundTruth}
          corrections={board.corrections.size}
          submittingLabel={session.submittingLabel}
          onSave={() => void session.saveGroundTruth()}
        />
      {/if}

      {#if board.inspection !== null}
        {@const inspection = board.inspection}
        <div class="inspector-col">
          <SquareInspector
            square={inspection.square}
            scores={inspection.scores}
            shown={inspection.shown}
            corrected={board.corrections.has(inspection.square)}
            onAssign={(cls) => board.assign(cls)}
            onRevert={() => board.revertCorrection(inspection.square)}
            onClose={() => board.deselect()}
            onStep={(delta) => board.stepSquare(delta)}
            onNextLow={() => board.nextLowConfidence()}
          />
        </div>
      {/if}
    </div>

    <ActionBar
      disabled={session.image === null || session.busy !== null}
      busy={session.busy}
      onRead={() => void session.runRead()}
      bind:height={barHeight}
    />

    <PipelineSteps {session} />

    <HistoryList
      {rows}
      note={historyNote(session.retention)}
      activeId={session.activeReadId}
      clearable={session.reads.length > 0}
      onRestore={(id) => void session.restoreRead(id)}
      onLoadUpload={(upload) => void session.loadUpload(upload)}
      onClear={() => session.clearReads()}
    />

    {#if session.failure}
      <div class="err">{session.failure}</div>
    {/if}
  </main>
</div>

<style>
  /* The snapshot page is a DIAGNOSTIC INTERFACE for the reading algorithm, not a wizard.
     One column, in pipeline order: capture → the stages → the position they produced →
     the log of earlier reads. The only thing that ever sits side-by-side is the piece
     switcher next to the board, and only when there is room for both.
     No emoji anywhere: state is text + colour. */
  .page {
    --bg: #14171c;
    --panel: #1b1f26;
    --panel-2: #232833;
    --line: #272b33;
    --fg: #e6e6e6;
    --muted: #9aa3ad;
    --dim: #7d8590;
    --accent: #2a6ef0;
    --accent-soft: #6aa8ff;
    --ok: #97e0a6;
    --warn: #ffcf33;
    --bad: #ff8f8f;
    background: var(--bg);
    color: var(--fg);
    -webkit-tap-highlight-color: transparent;
    -webkit-user-select: none;
    user-select: none;
    -webkit-touch-callout: none;
    min-height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }

  h1 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
  }

  .spacer {
    flex: 1;
  }

  .run-state {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--dim);
  }

  main {
    padding: 14px 16px calc(var(--bar-h) + 96px);
    max-width: 1240px;
    margin: 0 auto;
  }

  @media (min-width: 760px) {
    main {
      padding-bottom: 40px;
    }
  }

  /* Input and output, side by side and the same size: the photo left of the board it
     produced, with the piece switcher after them when the window has room. The photo
     never leaves the screen — it is what every stage is talking about. */
  .out-main {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-start;
    margin-top: 14px;
  }

  .inspector-col {
    flex: 1 1 320px;
    min-width: 0;
  }

  .err {
    color: var(--bad);
    font-size: 13px;
    white-space: pre-wrap;
    margin-top: 12px;
  }
</style>
