<script lang="ts">
  /**
   * A camera's stats, as the node pushes them: the phone link, what the camera
   * delivers, the board direction, and the health tick (frames/s, the last frame's
   * age, the recognizer, refused frames).
   */
  import type { HealthMessage, StateMessage } from '#lib/protocol/generated.ts';
  import { formatAge } from '#lib/control/status.ts';
  import { whiteSideCopy } from '#lib/control/direction.ts';

  interface Props {
    camera: StateMessage | null;
    health: HealthMessage | null;
  }

  const { camera, health }: Props = $props();

  const report = $derived(camera?.report ?? null);
  const connected = $derived(camera?.phone_connected ?? false);
  const portrait = $derived(report?.frame != null && report.frame.height > report.frame.width);
  const direction = $derived(health?.direction ?? null);
</script>

<dl>
  <dt>connection</dt>
  <dd class:bad={camera !== null && !connected}>
    {camera === null ? '—' : connected ? 'connected' : 'not connected'}
  </dd>
  <dt>stream</dt>
  <dd>{report?.ingest ?? '—'}</dd>
  <dt>frames/s</dt>
  <dd>{health?.fps == null ? '—' : health.fps.toFixed(1)}</dd>
  <dt>last frame</dt>
  <dd>{health === null ? '—' : formatAge(health.frame_age_s)}</dd>
  {#if report?.kind !== 'stream'}
    <dt>lens open</dt>
    <dd>{report?.device ?? '—'}</dd>
  {/if}
  <dt>frame delivered</dt>
  <dd class:bad={portrait}>
    {report?.frame ? `${report.frame.width}×${report.frame.height}` : '—'}
    {#if portrait}(portrait){/if}
  </dd>
  <dt>board direction</dt>
  <dd class:bad={direction?.camera_moved}>
    {#if direction === null}
      —
    {:else}
      {whiteSideCopy(direction.white_side)}
      <span class="quiet">({Math.round(direction.degrees)}°)</span>
    {/if}
  </dd>
  <dt>health</dt>
  <dd class:bad={health?.fallback || (health?.refused ?? 0) > 0}>
    {#if health === null}
      —
    {:else}
      {health.recognizer}
      {#if health.fallback}<span class="quiet">({health.fallback})</span>{/if}
      · <span title="frames the node refused for this board">{health.refused} refused</span>
    {/if}
  </dd>
</dl>

<style>
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    margin: 0;
    font-size: 12px;
  }
  dt {
    color: #7d8590;
  }
  dd {
    margin: 0;
    color: #c8cdd4;
    font-variant-numeric: tabular-nums;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  dd.bad {
    color: #ff8f8f;
  }
  .quiet {
    color: #9aa3ad;
  }
</style>
