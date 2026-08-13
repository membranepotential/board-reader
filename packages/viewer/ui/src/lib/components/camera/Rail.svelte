<script lang="ts">
  /**
   * The viewfinder's controls: a cluster of pills in their own flex track beside the
   * picture — a right-hand column in landscape with the primary button at the bottom
   * where the thumb rests, a bottom row in portrait. One DOM order serves both, and
   * `Viewfinder.svelte`'s single `flex-direction` is what chooses.
   *
   * **It has no size of its own to get wrong.** The column is as wide as its widest
   * label and no wider (`flex: 0 0 auto` + `white-space: nowrap`), so nothing here
   * states a pill width, and nothing elsewhere has to be told one. The build this
   * replaced hard-coded 124 px in a `--cluster-w` that three other rules read; a
   * label, a font size or a phone's text-size setting that disagreed with that number
   * pushed the column off the right of the screen with nothing to catch it.
   *
   * Still no strip and no border — that part of the previous design was right, and
   * was the operator's original complaint about the build before it. The pills are
   * translucent targets on black; what changed is only that the black is the rail's
   * own track rather than the far edge of the picture.
   *
   * **Five controls, and a sixth that only appears when it matters.** `detect board`
   * is out here because a phone at a board needs it often enough that a sheet is too
   * far — it came back after the first phone test. `rotate labels` did NOT: a
   * labelling is set once per board, so the sheet is the right home for it. It still
   * appears here, warn-ringed, for the one case that cannot wait for somebody to find
   * a sheet — an automatic re-detection may have put a8 on a different physical
   * corner (DESIGN decision #15), and until the operator says otherwise every reading
   * on the wire is turned.
   */
  interface Props {
    positionShown: boolean;
    /** Something else already owns the camera and the quad — a detection, a calibration. */
    detectDisabled: boolean;
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

<nav class="cluster">
  <button class="pill" type="button" onclick={onMore}>more</button>
  <button class="pill" class:active={positionShown} type="button" onclick={onPosition}>
    position
  </button>
  <button class="pill" type="button" disabled={detectDisabled} onclick={onDetect}>
    detect board
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
  /* Its own flex track, sized by its content: `flex: 0 0 auto` in a column, so the
     width is the widest label plus padding and the frame gets the rest. No `env()`
     anywhere — the stage already took the insets.

     `overflow-y: auto` is the last-resort net rather than a feature: the `clamp()`
     rules below already fit six pills on the shortest landscape stage a phone
     presents, but if a future label wraps, or a phone's text-size setting inflates
     the pills, the column scrolls instead of putting `start` under the fold where
     `overflow: hidden` on the stage would simply swallow it. `pan-y` because the
     stage is `touch-action: none` and would otherwise eat the scroll. */
  .cluster {
    flex: 0 0 auto;
    z-index: 9;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: clamp(4px, 1.6vh, 10px);
    padding: 8px;
    overflow-y: auto;
    overscroll-behavior: contain;
    touch-action: pan-y;
    scrollbar-width: none;
  }

  /* Every rule below is written `.cluster .pill…`, and that is not decoration: the
     page's own `.camera button` base rule (accent fill, 9 px corners, 15 px) is one
     class plus one type, so a bare `.pill` LOSES to it and the pills silently render
     as the old solid buttons. Two classes is the floor that wins. */
  .cluster .pill {
    background: rgba(20, 23, 28, 0.78);
    -webkit-backdrop-filter: blur(6px);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 999px;
    color: var(--fg);
    font-size: 14px;
    font-weight: 600;
    line-height: 1.15;
    padding: 8px 12px;
    /* One line, always: it is what makes the column's intrinsic width equal to the
       widest label, and it means a long label widens the rail rather than growing it
       downwards into the space the primary button needs. */
    white-space: nowrap;
    /* 48 where there is room, never below the 44 px touch floor, and continuous in
       between — this is what the three per-device `max-height` breakpoints used to
       approximate. Worked through at the two stages that bite: 390 px of landscape
       gives 47 px pills and 6 px gaps (six pills = 340 px, fits), 320 px gives the
       44 px floor and 5 px gaps (six pills = 305 px, fits). */
    min-height: clamp(44px, 12vh, 48px);
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

  /* Pushes the primary button to the bottom of the landscape column, under the thumb.
     `flex: 1 1 0` so it is the FIRST thing to give when the column is tight: the
     pills keep their touch height and the gap to the primary closes instead. */
  .cluster .spacer {
    flex: 1 1 0;
    min-height: 0;
  }

  @media (orientation: portrait) {
    .cluster {
      flex-direction: row;
      /* THREE TO A ROW, always. Six across 390px gives 56px pills, and `position`
         and `calibrate · N` are both wider than that — and the two arrive together,
         since the contextual pill only appears once a calibration has already made
         the calibrate label longer. Rows of three give 119px, which every label fits
         on one line with room to spare. The row's HEIGHT is whatever the pills come
         to; nothing has to be told what that is. */
      flex-wrap: wrap;
      gap: 8px;
      overflow: visible;
    }

    .cluster .pill {
      /* Minus 18 rather than 16 (two gaps): the couple of spare pixels stop a
         sub-pixel rounding error from spilling the third pill onto its own row. */
      flex: 1 1 calc((100% - 18px) / 3);
      min-width: 0;
      padding: 8px 6px;
    }

    .cluster .spacer {
      display: none;
    }
  }
</style>
