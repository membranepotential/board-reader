<script lang="ts">
  /**
   * A camera's Auto-frame: what it is doing (`StateMessage.autoframe`), the on/off
   * switch (`config.auto`, sent as a config patch) and "Re-detect frame" (`reframe`:
   * the node finds the board again now and switches Auto-frame back on), which works
   * in and outside crop mode.
   */
  import type { CameraConfig, StateMessage } from '#lib/protocol/generated.ts';
  import { AUTOFRAME_COPY } from '#lib/control/autoframe.ts';

  interface Props {
    camera: StateMessage | null;
    locked: boolean;
    onsend: (patch: Partial<CameraConfig>) => void;
    onreframe: () => void;
  }

  const { camera, locked, onsend, onreframe }: Props = $props();

  const connected = $derived(camera?.phone_connected ?? false);
</script>

<div class="autoframe">
  <p
    class="status"
    data-testid="autoframe"
    data-status={camera?.autoframe}
    class:bad={camera?.autoframe === 'no-board' || camera?.autoframe === 'not-visible'}
  >
    {camera === null ? '—' : AUTOFRAME_COPY[camera.autoframe]}
  </p>
  {#if camera?.autoframe === 'framed' && !camera.watching}
    <p class="hint" data-testid="drift-off">
      The board is small in the picture, so a moved camera is not noticed: press Re-detect frame
      after moving it.
    </p>
  {/if}
  <div class="row">
    <label>
      <input
        type="checkbox"
        checked={camera?.config.auto ?? false}
        disabled={locked}
        onchange={(e) => onsend({ auto: e.currentTarget.checked })}
      />
      frame the board automatically
    </label>
    <button
      type="button"
      disabled={locked || !connected}
      title={connected ? 'find the board again now' : 'the camera is not connected'}
      onclick={onreframe}>Re-detect frame</button
    >
  </div>
</div>

<style>
  .autoframe {
    display: grid;
    gap: 8px;
  }
  p {
    margin: 0;
    font-size: 13px;
  }
  .status.bad {
    color: #ff8f8f;
  }
  .hint {
    font-size: 12px;
    color: #6f7782;
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
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
</style>
