<script lang="ts">
  /**
   * A camera's capture and board settings: lens, resolution, rules, and the stream
   * toggle. Each change goes up at once through `onsend` as a patch of the config;
   * the card builds the whole `configure` from it. A stream camera
   * (`CameraReport.kind`) has one picture at one size, so it offers no lens and no
   * resolution.
   */
  import { RESOLUTION_NAMES, RULES_NAMES, type Resolution, type Rules } from '#lib/protocol/index.ts';
  import type { CameraConfig, CameraReport } from '#lib/protocol/generated.ts';

  interface Props {
    config: CameraConfig | null;
    report: CameraReport | null;
    /** Nothing can be sent: no config yet, or the socket is down. */
    locked: boolean;
    onsend: (patch: Partial<CameraConfig>) => void;
  }

  const { config, report, locked, onsend }: Props = $props();

  /** How each setting of `rules` reads in the select. */
  const RULES_LABEL: Readonly<Record<Rules, string>> = {
    standard: 'standard chess',
    free: 'free (extra pieces, variants)'
  };
</script>

<div class="settings">
  {#if report?.kind !== 'stream'}
    <label>
      lens
      <select
        value={config?.device ?? ''}
        disabled={locked}
        onchange={(e) => onsend({ device: e.currentTarget.value || null })}
      >
        <option value="">phone's default rear camera</option>
        {#each report?.devices ?? [] as device (device.id)}
          <option value={device.label}>{device.label}</option>
        {/each}
        {#if config?.device && !report?.devices.some((d) => d.label === config.device)}
          <option value={config.device}>{config.device} (not reported)</option>
        {/if}
      </select>
    </label>
    <label>
      resolution
      <select
        value={config?.resolution}
        disabled={locked}
        onchange={(e) => onsend({ resolution: e.currentTarget.value as Resolution })}
      >
        {#each RESOLUTION_NAMES as name (name)}
          <option value={name}>{name}</option>
        {/each}
      </select>
    </label>
  {/if}
  <label class="wide">
    rules
    <select
      value={config?.rules}
      disabled={locked}
      onchange={(e) => onsend({ rules: e.currentTarget.value as Rules })}
    >
      {#each RULES_NAMES as name (name)}
        <option value={name}>{RULES_LABEL[name]}</option>
      {/each}
    </select>
    <span class="hint">
      {config?.rules === 'free'
        ? 'the settled board shows whatever the reads agree on'
        : 'a settle that would make the board impossible waits'}
    </span>
  </label>
  <button
    type="button"
    class="stream wide"
    class:on={config?.streaming}
    disabled={locked}
    onclick={() => onsend({ streaming: !config?.streaming })}
  >
    {config?.streaming ? 'Stop streaming' : 'Start streaming'}
  </button>
</div>

<style>
  .settings {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .wide {
    grid-column: 1 / -1;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
    color: #9aa3ad;
    min-width: 0;
  }
  select {
    font: inherit;
    font-size: 13px;
    background: #232833;
    color: #e6e6e6;
    border: 1px solid #2f3540;
    border-radius: 6px;
    padding: 4px 6px;
    min-width: 0;
  }
  .hint {
    font-size: 12px;
    color: #6f7782;
  }
  button {
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 12px;
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
  .stream.on {
    background: #3a2020;
    border-color: #5a2a2a;
    color: #ff9a9a;
  }
</style>
