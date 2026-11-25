<script lang="ts">
  /**
   * How a camera is framed, in one place: an Auto | Manual toggle on the left
   * (`config.auto`), and on the right what that mode offers.
   *
   *  - Auto: what Auto-frame is doing (`StateMessage.autoframe`) and "Re-detect
   *    frame" (`reframe`: the node finds the board again now). Switching to Auto
   *    sets `auto`, and the node frames the camera afresh.
   *  - Manual: "Crop", which puts the card in crop edit mode (the whole frame with
   *    the rectangle); while editing, Save crop / Cancel / Send whole frame. A crop
   *    saved here is a crop set by hand, so the camera stays Manual.
   */
  import type { StateMessage } from '#lib/protocol/generated.ts';
  import { AUTOFRAME_COPY } from '#lib/control/autoframe.ts';

  interface Props {
    camera: StateMessage | null;
    /** Nothing can be sent: no state yet, or the socket is down. */
    locked: boolean;
    /** The card is in crop edit mode. */
    cropping: boolean;
    /** A rectangle is drawn and not saved yet. */
    dirty: boolean;
    onauto: (on: boolean) => void;
    onreframe: () => void;
    oncrop: () => void;
    onsave: () => void;
    oncancel: () => void;
    onwhole: () => void;
  }

  const { camera, locked, cropping, dirty, onauto, onreframe, oncrop, onsave, oncancel, onwhole }: Props =
    $props();

  /** One name per card, so each card's two radios are one group. */
  const uid = $props.id();
  const group = `framing-${uid}`;
  const auto = $derived(camera?.config.auto ?? true);
  const connected = $derived(camera?.phone_connected ?? false);
  const wholeSent = $derived(camera?.config.crop === null);
</script>

<div class="framing">
  <div class="toggle" role="radiogroup" aria-label="framing">
    <label class:on={auto}>
      <input type="radio" name={group} checked={auto} disabled={locked} onchange={() => onauto(true)} />
      Auto
    </label>
    <label class:on={!auto}>
      <input type="radio" name={group} checked={!auto} disabled={locked} onchange={() => onauto(false)} />
      Manual
    </label>
  </div>

  <div class="mode">
    {#if auto}
      <p
        class="status"
        data-testid="autoframe"
        data-status={camera?.autoframe}
        class:bad={camera?.autoframe === 'no-board' || camera?.autoframe === 'not-visible'}
      >
        {camera === null ? '—' : AUTOFRAME_COPY[camera.autoframe]}
      </p>
      <div class="actions">
        <button
          type="button"
          disabled={locked || !connected}
          title={connected ? 'find the board again now' : 'the camera is not connected'}
          onclick={onreframe}>Re-detect frame</button
        >
      </div>
      {#if camera?.autoframe === 'framed' && !camera.watching}
        <p class="hint" data-testid="drift-off">
          The board is small in the picture, so a moved camera is not noticed: press Re-detect frame
          after moving it.
        </p>
      {/if}
    {:else if cropping}
      <div class="actions">
        <button type="button" class="primary" disabled={!dirty || locked} onclick={onsave}>Save crop</button>
        <button type="button" onclick={oncancel}>Cancel</button>
        <button type="button" disabled={locked || (wholeSent && !dirty)} onclick={onwhole}>Send whole frame</button>
        {#if dirty}<span class="dirty">unsaved</span>{/if}
      </div>
      <p class="hint">
        Drag the rectangle around the board with a small margin: the board should reach past both
        dashed guides (at least half the width). Landscape reads best.
      </p>
    {:else}
      <p class="status">{wholeSent ? 'the whole frame is sent' : 'the crop was set by hand'}</p>
      <div class="actions">
        <button type="button" disabled={locked} onclick={oncrop}>Crop</button>
      </div>
    {/if}
  </div>
</div>

<style>
  .framing {
    display: flex;
    align-items: flex-start;
    gap: 14px;
  }
  .toggle {
    display: flex;
    flex: none;
    border: 1px solid #2f3540;
    border-radius: 6px;
    overflow: hidden;
  }
  .toggle label {
    position: relative;
    font-size: 13px;
    padding: 5px 12px;
    background: #232833;
    color: #9aa3ad;
    cursor: pointer;
  }
  .toggle label + label {
    border-left: 1px solid #2f3540;
  }
  .toggle label.on {
    background: #23303f;
    color: #8fb6e8;
    font-weight: 600;
  }
  .toggle label:has(input:disabled) {
    cursor: default;
    opacity: 0.6;
  }
  .toggle label:has(input:focus-visible) {
    outline: 2px solid #ffcf33;
    outline-offset: -2px;
  }
  /* The radio itself is the label's job to show; keep it focusable, not visible. */
  .toggle input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
  }
  /* Status and buttons share the toggle's row; only a hint wraps to its own line. */
  .mode {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 12px;
    min-height: 30px;
  }
  p {
    margin: 0;
    font-size: 13px;
  }
  .status.bad {
    color: #ff8f8f;
  }
  .hint {
    flex-basis: 100%;
    font-size: 12px;
    color: #6f7782;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  button {
    font: inherit;
    font-size: 13px;
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid #2f3540;
    background: #232833;
    color: #e6e6e6;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  button.primary {
    background: #2a6ef0;
    border-color: #2a6ef0;
    color: #fff;
  }
  .dirty {
    font-size: 12px;
    color: #ffcf33;
  }
</style>
