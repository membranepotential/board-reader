<script lang="ts">
  /**
   * The six viewfinder controls. Everything else lives in the sheet.
   *
   * One DOM order serves both orientations: a right-hand column in landscape (a
   * flex spacer pushes the primary button to the bottom, where the thumb rests),
   * a 3x2 dock along the bottom in portrait.
   */
  interface Props {
    positionShown: boolean;
    detectDisabled: boolean;
    rotateDisabled: boolean;
    /**
     * A re-detection can land the labels on a different physical corner than the
     * operator chose — the one case where the page changes a quad the operator
     * authored, so it has to be impossible to miss which control puts it right.
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
    onDetect: () => void;
    onRotate: () => void;
    onCalibrate: () => void;
    onStream: () => void;
  }

  const {
    positionShown,
    detectDisabled,
    rotateDisabled,
    rotateAttention,
    calibrateDisabled,
    calibrateCount,
    calibrateTitle,
    streaming,
    onMore,
    onPosition,
    onDetect,
    onRotate,
    onCalibrate,
    onStream
  }: Props = $props();
</script>

<nav class="rail">
  <button class="secondary" type="button" onclick={onMore}>more</button>
  <button class:secondary={!positionShown} type="button" onclick={onPosition}>position</button>
  <button class="secondary" type="button" disabled={detectDisabled} onclick={onDetect}>
    detect board
  </button>
  <button
    class="secondary"
    class:attention={rotateAttention}
    type="button"
    disabled={rotateDisabled}
    onclick={onRotate}
  >
    rotate labels
  </button>
  <button
    class="secondary"
    type="button"
    disabled={calibrateDisabled}
    title={calibrateTitle}
    onclick={onCalibrate}
  >
    {calibrateCount ? `calibrate · ${calibrateCount}` : 'calibrate'}
  </button>
  <span class="spacer"></span>
  <button class:on={streaming} type="button" onclick={onStream}>
    {streaming ? 'stop' : 'start'}
  </button>
</nav>

<style>
  .rail {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: var(--rail);
    z-index: 9;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 8px 8px 6px;
    padding-right: calc(8px + env(safe-area-inset-right));
    padding-top: calc(8px + env(safe-area-inset-top));
    background: rgba(20, 23, 28, 0.78);
    border-left: 1px solid var(--line);
  }

  .rail button {
    font-size: 13px;
    padding: 6px 6px;
    width: 100%;
  }

  .rail .spacer {
    flex: 1;
    min-height: 0;
  }

  .rail button.attention {
    box-shadow: 0 0 0 2px var(--warn);
  }

  .rail button.on {
    background: #7a2a2a;
  }

  @media (orientation: portrait) {
    .rail {
      top: auto;
      left: 0;
      right: 0;
      bottom: 0;
      width: auto;
      height: var(--dock);
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      grid-auto-rows: 1fr;
      gap: 6px;
      border-left: 0;
      border-top: 1px solid var(--line);
      padding: 8px calc(8px + env(safe-area-inset-right)) calc(8px + env(safe-area-inset-bottom))
        calc(8px + env(safe-area-inset-left));
    }

    .rail .spacer {
      display: none;
    }
  }
</style>
