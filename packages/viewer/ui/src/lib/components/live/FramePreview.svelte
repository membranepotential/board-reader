<script lang="ts">
  /**
   * The collapsible preview of the latest frame the node ingested for one board —
   * ported from index.html's `.preview` markup and `refreshPreview`.
   *
   * Docked to the board's left, growing rightward from a vertical tab that stays
   * anchored where the operator clicked it (only the pane's width animates; the
   * tab never moves). It fetches only while open: the `<img>` is in the DOM
   * exactly when `open` is true, so a collapsed preview issues no request, and
   * reopening — or a new frame arriving while already open, via `seq` — is a
   * fresh `src` and a fresh fetch. The endpoint 404s until the node has ingested a
   * frame, which is not an error the rest of the card should see, only a note in
   * this pane.
   */
  import { boardFrameUrl } from '$lib/api';

  interface Props {
    boardId: string;
    /** Cache-buster: the board's latest event seq, so a new frame is a new URL. */
    seq: number;
    open: boolean;
    onToggle: () => void;
  }

  const { boardId, seq, open, onToggle }: Props = $props();

  const frameUrl = $derived(boardFrameUrl(boardId, seq));

  let loaded = $state(false);
  // A fresh open, or a new frame while already open, starts from "no frame yet"
  // again — the note (or the stale image) would otherwise linger past its answer.
  $effect(() => {
    open;
    frameUrl;
    loaded = false;
  });
</script>

<div class="preview" class:open>
  <button
    class="preview-tab"
    aria-expanded={open}
    title="show the latest frame the node ingested"
    onclick={onToggle}
  >
    <span class="chev">›</span>preview
  </button>
  <div class="preview-pane">
    {#if open}
      <img
        class="frame"
        class:absent={!loaded}
        src={frameUrl}
        alt="latest ingested frame for board {boardId}"
        onload={() => (loaded = true)}
        onerror={() => (loaded = false)}
      />
    {/if}
    <div class="frame-note" class:on={!loaded}>no frame yet</div>
  </div>
</div>

<style>
  .preview {
    display: flex;
    align-items: flex-start;
  }
  .preview-tab {
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
  .preview-tab:hover {
    color: #e6e6e6;
    border-color: #3c434f;
  }
  .preview.open .preview-tab {
    color: #8fb6e8;
    border-color: #35507a;
  }
  .preview-tab .chev {
    transition: transform 200ms ease;
  }
  .preview.open .preview-tab .chev {
    transform: rotate(180deg);
  }
  /* Width is the only animated axis; the pane clips its contents while it grows. */
  .preview-pane {
    width: 0;
    overflow: hidden;
    transition: width 220ms ease;
    height: var(--board-size, 360px);
    position: relative;
  }
  .preview.open .preview-pane {
    width: var(--board-size, 360px);
    margin-left: 8px;
  }
  /* Explicit box, never a percentage through an auto-height ancestor. */
  .frame {
    width: var(--board-size, 360px);
    height: var(--board-size, 360px);
    object-fit: contain;
    display: block;
    background: #0f1216;
    border: 1px solid #272b33;
    border-radius: 6px;
  }
  .frame.absent {
    visibility: hidden;
  }
  .frame-note {
    position: absolute;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    color: #7d8590;
    text-align: center;
    border: 1px dashed #2f3540;
    border-radius: 6px;
    padding: 12px;
  }
  .frame-note.on {
    display: flex;
  }
</style>
