<script lang="ts">
  /**
   * The viewfinder's controls, as a cluster of pills floating ON the picture.
   *
   * It used to be an opaque strip beside the picture, and the operator's verdict on
   * it was that the buttons were cramped and the strip ate the frame. So: no strip,
   * no border, four translucent targets over the picture itself, sized for a thumb.
   * A right-hand column in landscape with the primary button at the bottom where the
   * thumb rests; a bottom row in portrait. One DOM order serves both.
   *
   * **Four controls, not six.** `detect board` and `rotate labels` moved into the
   * sheet — the hint line's "look again" already covers a manual detect in the common
   * case. `rotate labels` comes BACK out here, warn-styled, for the one case that
   * cannot wait for somebody to find a sheet: an automatic re-detection may have put
   * a8 on a different physical corner (DESIGN decision #15), and until the operator
   * says otherwise every reading on the wire is turned.
   *
   * The nav itself is INERT and only the pills take pointers, so the gaps between
   * them pass touches through to the corner handles now underneath — the picture is
   * full-bleed, so there is something under them to reach.
   *
   * `--cluster-w` / `--cluster-h` come from the stage (`Viewfinder.svelte`), which is
   * also what the banner and the hint line dodge, so the reservation is stated once.
   */
  interface Props {
    positionShown: boolean;
    rotateDisabled: boolean;
    /**
     * A re-detection can land the labels on a different physical corner than the
     * operator chose — the one case where the page changes a quad the operator
     * authored, so it has to be impossible to miss which control puts it right. This
     * is also the only thing that puts `rotate labels` on screen at all.
     */
    rotateAttention: boolean;
    calibrateDisabled: boolean;
    /** The bank is additive, so the count says it is growing. */
    calibrateCount: number;
    /** Kept for a mouse; the hint line is what a phone can actually read. */
    calibrateTitle: string;
    streaming: boolean;
    onMore: () => void;
    onPosition: () => void;
    onRotate: () => void;
    onCalibrate: () => void;
    onStream: () => void;
  }

  const {
    positionShown,
    rotateDisabled,
    rotateAttention,
    calibrateDisabled,
    calibrateCount,
    calibrateTitle,
    streaming,
    onMore,
    onPosition,
    onRotate,
    onCalibrate,
    onStream
  }: Props = $props();
</script>

<nav class="cluster">
  <button class="pill" type="button" onclick={onMore}>more</button>
  <button class="pill" class:active={positionShown} type="button" onclick={onPosition}>
    position
  </button>
  {#if rotateAttention}
    <button
      class="pill attention"
      type="button"
      disabled={rotateDisabled}
      onclick={onRotate}
    >
      rotate labels
    </button>
  {/if}
  <button
    class="pill"
    type="button"
    disabled={calibrateDisabled}
    title={calibrateTitle}
    onclick={onCalibrate}
  >
    {calibrateCount ? `calibrate · ${calibrateCount}` : 'calibrate'}
  </button>
  <span class="spacer"></span>
  <button class="pill primary" class:on={streaming} type="button" onclick={onStream}>
    {streaming ? 'stop' : 'start'}
  </button>
</nav>

<style>
  .cluster {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: var(--cluster-w);
    z-index: 9;
    /* Inert: only the pills take pointers, so a corner handle in a gap is reachable. */
    pointer-events: none;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 10px;
    /* The notch is on the left OR the right in landscape depending on which way the
       phone was turned; this cluster only ever touches the right edge. */
    padding: calc(8px + env(safe-area-inset-top)) max(8px, env(safe-area-inset-right))
      max(8px, env(safe-area-inset-bottom)) 8px;
  }

  /* Every rule below is written `.cluster .pill…`, and that is not decoration: the
     page's own `.camera button` base rule (accent fill, 9 px corners, 15 px) is one
     class plus one type, so a bare `.pill` LOSES to it and the pills silently render
     as the old solid buttons. Two classes is the floor that wins. */
  .cluster .pill {
    pointer-events: auto;
    background: rgba(20, 23, 28, 0.78);
    -webkit-backdrop-filter: blur(6px);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 999px;
    color: var(--fg);
    font-size: 14px;
    font-weight: 600;
    line-height: 1.15;
    padding: 8px 10px;
    min-height: 48px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.45);
  }

  .cluster .pill.active {
    color: var(--accent-soft);
    border-color: rgba(106, 168, 255, 0.55);
    background: rgba(24, 42, 74, 0.82);
  }

  .cluster .pill.primary {
    background: rgba(42, 110, 240, 0.9);
    border-color: rgba(106, 168, 255, 0.5);
    font-size: 15px;
    font-weight: 700;
  }

  .cluster .pill.primary.on {
    background: rgba(150, 42, 42, 0.9);
    border-color: rgba(255, 143, 143, 0.5);
  }

  .cluster .pill.attention {
    color: var(--warn);
    border-color: var(--warn);
    box-shadow:
      0 0 0 2px rgba(255, 207, 51, 0.45),
      0 2px 12px rgba(0, 0, 0, 0.45);
  }

  /* Pushes the primary button to the bottom of the landscape column, under the thumb. */
  .cluster .spacer {
    flex: 1;
    min-height: 0;
  }

  @media (orientation: portrait) {
    .cluster {
      top: auto;
      left: 0;
      right: 0;
      bottom: 0;
      width: auto;
      height: var(--cluster-h);
      flex-direction: row;
      gap: 8px;
      padding: 8px max(8px, env(safe-area-inset-right)) max(8px, env(safe-area-inset-bottom))
        max(8px, env(safe-area-inset-left));
    }

    .cluster .pill {
      flex: 1;
      min-width: 0;
      padding: 8px 4px;
    }

    .cluster .spacer {
      display: none;
    }
  }
</style>
