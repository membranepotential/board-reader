<script lang="ts">
  /**
   * Everything that is not a viewfinder control.
   *
   * Slid with a transform rather than `display: none`, so its contents keep layout
   * while it is closed and nothing inside it has to be re-measured on open.
   */
  import { MOVING_BELOW } from '$lib/protocol';
  import type { Lens } from './state';

  interface Props {
    open: boolean;
    boardId: string;
    tracking: boolean;
    rotateDisabled: boolean;
    lenses: readonly Lens[];
    lensId: string | null;
    sent: number;
    acked: number | null;
    dropped: number;
    fps: number;
    stability: number;
    bursting: boolean;
    /** The node's last reading for this board, verbatim. */
    placement: string;
    onClose: () => void;
    onBoardId: (id: string) => void;
    onToggleTracking: () => void;
    onResetCrop: () => void;
    /** Closes the sheet on the way out: its whole result is on the picture. */
    onRotate: () => void;
    onLens: (id: string, label: string) => void;
  }

  const {
    open,
    boardId,
    tracking,
    rotateDisabled,
    lenses,
    lensId,
    sent,
    acked,
    dropped,
    fps,
    stability,
    bursting,
    placement,
    onClose,
    onBoardId,
    onToggleTracking,
    onResetCrop,
    onRotate,
    onLens
  }: Props = $props();

  const moving = $derived(stability < MOVING_BELOW);
  const mode = $derived(
    bursting ? 'burst — a move is under way' : moving ? 'moving' : 'at rest'
  );
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="scrim" class:open onclick={onClose}></div>

<section class="sheet" class:open aria-label="settings">
  <div class="grab">
    <h2 class="title">settings</h2>
    <span class="spacer"></span>
    <button class="secondary" type="button" onclick={onClose}>close</button>
  </div>

  <h2>framing</h2>
  <!-- `rotate labels` and not `detect board`: a labelling is set once per board, so
       this is the right distance away, while detect turned out to be wanted often
       enough at a real board that it earned a pill back on the viewfinder. This one
       still appears out there too, warn-ringed, when a re-detection may have moved
       the labels — that case cannot wait for somebody to find a sheet. -->
  <div class="actions">
    <button class="secondary" type="button" disabled={rotateDisabled} onclick={onRotate}>
      rotate labels
    </button>
  </div>

  <h2>board</h2>
  <div class="row">
    <span class="board-id">
      board
      <input
        value={boardId}
        autocapitalize="off"
        autocorrect="off"
        spellcheck="false"
        onchange={(event) => onBoardId(event.currentTarget.value)}
      />
    </span>
    <button class="text-btn" type="button" onclick={onToggleTracking}>
      {tracking ? 'tracking on' : 'tracking off'}
    </button>
    <button class="text-btn" type="button" onclick={onResetCrop}>reset crop</button>
  </div>

  <!-- Hidden outright on a device with one camera: a select with a single option is
       furniture, not a control. -->
  {#if lenses.length > 1}
    <div class="lens">
      <label for="lens-select">lens</label>
      <select
        id="lens-select"
        value={lensId}
        onchange={(event) => {
          const option = event.currentTarget.selectedOptions[0];
          if (option !== undefined) onLens(option.value, option.textContent ?? '');
        }}
      >
        {#each lenses as lens (lens.id)}
          <option value={lens.id}>{lens.label}</option>
        {/each}
      </select>
    </div>
    <p class="hint">
      An iPhone lists each rear lens separately. The <b>ultra wide</b> one fits a board the main
      lens cannot from close up — at the cost of visible barrel distortion near the frame edges,
      where the quad will fit the board less exactly. Keep the board near the middle of the
      picture.
    </p>
  {/if}

  <h2>stream</h2>
  <div class="status">
    <div class="stat">
      <div class="k">frames</div>
      <div class="v">{sent} sent</div>
      <div class="k mono">
        {acked === null ? 'acked —' : `acked ${acked}`}{dropped ? ` · ${dropped} dropped` : ''}
      </div>
    </div>
    <div class="stat">
      <div class="k">rate</div>
      <div class="v">{fps.toFixed(1)} fps</div>
      <div class="k">{mode}</div>
    </div>
    <div class="stat">
      <div class="k">stability</div>
      <div class="v">{stability.toFixed(2)}</div>
      <div class="bar" class:moving><i style:width="{Math.round(stability * 100)}%"></i></div>
    </div>
  </div>

  <h2>what the reader sees</h2>
  <div class="readout">
    <div>placement <code class="pl">{placement}</code></div>
    <div class="legend"><i></i>low confidence</div>
  </div>

  <details>
    <summary>How framing works</summary>
    <p>
      Drag the four handles onto the playing surface's corners. <b>a8</b> is wherever
      <i>you</i> call a8 — geometry cannot tell which corner is a8, so use
      <b>rotate labels</b> if they sit wrong.
    </p>
    <p>
      The dashed box is what gets sent: it keeps headroom above the board so tall pieces are not
      decapitated at a real camera angle.
    </p>
    <p>
      Once set, the corners <b>follow the board</b> — move the phone and they move with it, labels
      and all. If the page loses the board it freezes the quad where it last fitted rather than
      sending a wrong crop, and after a while looks for the board again by itself.
    </p>
    <p>
      You rarely need any of this: the page looks for the board on its own as soon as the picture
      settles, and <b>detect board</b> asks it to look again.
    </p>
  </details>
</section>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 35;
    background: rgba(0, 0, 0, 0.5);
    opacity: 0;
    pointer-events: none;
    transition: opacity 160ms ease;
  }

  .scrim.open {
    opacity: 1;
    pointer-events: auto;
  }

  .sheet {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 40;
    max-height: 88dvh;
    overflow-y: auto;
    overscroll-behavior: contain;
    touch-action: pan-y;
    background: var(--panel);
    border-top: 1px solid var(--line);
    border-radius: 14px 14px 0 0;
    /* The sheet spans edge to edge, so in landscape it runs straight under the sensor
       housing and the first 44-59 px of every line is unreadable. Which side that is
       depends on which way the phone was turned, so both are budgeted. */
    padding: 10px max(16px, env(safe-area-inset-right, 0px))
      calc(24px + env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left, 0px));
    transform: translateY(101%);
    transition: transform 190ms ease;
    pointer-events: none;
  }

  .sheet.open {
    transform: none;
    pointer-events: auto;
  }

  .sheet h2 {
    margin: 18px 0 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }

  .sheet h2.title {
    margin: 8px 0;
  }

  .grab {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .grab .spacer {
    flex: 1;
  }

  .row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }

  .actions {
    display: flex;
    gap: 10px;
  }

  .actions button {
    flex: 1;
  }

  details {
    margin-top: 10px;
    font-size: 13px;
    color: var(--muted);
  }

  summary {
    color: var(--accent-soft);
    font-weight: 600;
    padding: 8px 0;
    min-height: 44px;
    display: flex;
    align-items: center;
  }

  details p {
    margin: 8px 0;
  }

  .hint {
    color: var(--muted);
    font-size: 13px;
    margin: 8px 0 0;
  }

  .hint b {
    color: var(--warn);
  }

  .status {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(112px, 1fr));
    gap: 10px;
    margin-top: 4px;
  }

  .stat {
    background: var(--panel-2);
    border-radius: 8px;
    padding: 8px 10px;
  }

  .stat .k {
    font-size: 11px;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .stat .v {
    font-size: 15px;
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
  }

  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .bar {
    height: 6px;
    border-radius: 3px;
    background: #2b2f37;
    overflow: hidden;
    margin-top: 7px;
  }

  .bar > i {
    display: block;
    height: 100%;
    background: var(--accent-soft);
    transition: width 120ms linear;
  }

  .bar.moving > i {
    background: var(--warn);
  }

  .board-id {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
    color: var(--muted);
  }

  .board-id input {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 8px;
    color: var(--fg);
    font: inherit;
    font-size: 14px;
    padding: 10px;
    min-height: 44px;
    width: 8.5em;
  }

  .lens {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--muted);
    margin-top: 10px;
  }

  .lens select {
    flex: 1;
    min-width: 0;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 8px;
    color: var(--fg);
    font: inherit;
    font-size: 14px;
    padding: 10px;
    min-height: 44px;
  }

  .readout {
    margin-top: 10px;
    font-size: 13px;
    color: var(--muted);
    display: grid;
    gap: 4px;
  }

  .readout .pl {
    word-break: break-all;
    color: #c8cdd4;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .legend {
    font-size: 12px;
    color: var(--dim);
  }

  .legend i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    vertical-align: -1px;
    margin-right: 4px;
    background: var(--warn);
  }

  /* `.row .text-btn`, not `.text-btn`: the page's `.camera button` base rule is one
     class plus one type, so a single class loses to it and these render as solid
     accent buttons with an underline through them — which is what they had been
     doing. */
  .row .text-btn {
    background: none;
    border: 0;
    color: var(--accent-soft);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    min-height: auto;
    padding: 6px 2px;
    text-decoration: underline;
    touch-action: manipulation;
    cursor: pointer;
  }
</style>
