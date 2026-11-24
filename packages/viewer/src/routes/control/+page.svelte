<script lang="ts">
  /**
   * board-reader · camera control — the laptop side of the "dumb camera" (DESIGN D11).
   *
   * The phone (/camera) only films and streams. This page chooses what it films: the
   * crop rectangle on its full-frame preview, the lens, the resolution, and whether it
   * streams. The node holds that config per board and pushes it to the phone.
   *
   * One socket per board (`ControlLink`): the node pushes the camera state on connect
   * and on every change, the phone's preview, the latest received frame and a health
   * tick, so nothing here polls. The socket closes while the tab is hidden, which is
   * also what tells the phone to stop sending previews nobody sees.
   *
   * The crop is edited locally and only sent on Save, so a pushed state never yanks
   * a rectangle out from under the pointer. The lens, resolution and stream toggle
   * are sent the moment they change. Each `configure` is the whole config, built from
   * the last one this page knows — applied here at once, so a second change made
   * before the node's answer arrives builds on the first instead of undoing it.
   */
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import { RESOLUTION_NAMES, type Resolution } from '#lib/protocol/index.ts';
  import type {
    CameraConfig,
    ControlDown,
    CropRect,
    FrameMessage,
    HealthMessage,
    PreviewMessage,
    StateMessage
  } from '#lib/protocol/generated.ts';
  import { DEFAULT_BOARD_ID, boardIdFrom } from '#lib/components/camera/state.ts';
  import { ControlLink } from '#lib/control/link.ts';
  import { jpegSrc } from '#lib/control/messages.ts';
  import { WHOLE_FRAME } from '#lib/control/crop.ts';
  import { FRAMING_COPY, framingWarnings } from '#lib/control/framing.ts';
  import { formatAge } from '#lib/control/status.ts';
  import { CAMERA_MOVED_COPY, whiteSideCopy } from '#lib/control/direction.ts';
  import CropEditor from '#lib/components/control/CropEditor.svelte';

  /** `null` until mounted: the board comes from `?board=`, which prerendering cannot read. */
  let boardId = $state<string | null>(null);
  let camera = $state.raw<StateMessage | null>(null);
  let preview = $state.raw<PreviewMessage | null>(null);
  let received = $state.raw<FrameMessage | null>(null);
  let health = $state.raw<HealthMessage | null>(null);
  /** Whether this page's socket to the node is open. */
  let up = $state(false);
  /** The rectangle being edited and not yet saved; `null` shows the node's. */
  let edit = $state<CropRect | null>(null);
  let failure = $state<string | null>(null);
  let link: ControlLink | null = null;

  onMount(() => {
    boardId = boardIdFrom(location.search);
  });

  $effect(() => {
    const id = boardId;
    if (id === null) return;
    const next = new ControlLink({
      boardId: id,
      onMessage: receive,
      onLink: (isUp, detail) => {
        up = isUp;
        failure = isUp ? null : `the node is not reachable (${detail}) — reconnecting…`;
      }
    });
    link = next;
    if (document.visibilityState === 'visible') next.start();
    return () => {
      next.stop();
      if (link === next) link = null;
      up = false;
    };
  });

  function onVisibilityChange(): void {
    if (document.visibilityState === 'visible') link?.start();
    else link?.stop();
  }

  function receive(message: ControlDown): void {
    switch (message.type) {
      case 'state':
        camera = message;
        break;
      case 'preview':
        preview = message;
        break;
      case 'frame':
        received = message;
        break;
      case 'health':
        health = message;
        break;
      case 'refused':
        failure = `not saved: ${message.reason}`;
        break;
    }
  }

  function send(patch: Partial<CameraConfig>): boolean {
    if (camera === null || link === null) return false;
    const config = { ...camera.config, ...patch };
    if (!link.configure(config)) {
      failure = 'not saved: the node is not connected';
      return false;
    }
    camera = { ...camera, config };
    failure = null;
    return true;
  }

  function saveCrop(crop: CropRect | null): void {
    if (send({ crop })) edit = null;
  }

  function chooseBoard(raw: string): void {
    const next = raw.trim() || DEFAULT_BOARD_ID;
    if (next === boardId) return;
    boardId = next;
    camera = null;
    preview = null;
    received = null;
    health = null;
    edit = null;
    const url = new URL(location.href);
    url.searchParams.set('board', next);
    history.replaceState(history.state, '', url);
  }

  const config = $derived(camera?.config ?? null);
  const report = $derived(camera?.report ?? null);
  /** The camera's frame: what the phone says it delivers, else the preview's size. */
  const frame = $derived(report?.frame ?? preview ?? null);
  const shownCrop = $derived(edit ?? config?.crop ?? null);
  const warnings = $derived(framingWarnings(shownCrop, frame));
  const connected = $derived(camera?.phone_connected ?? false);
  const portrait = $derived(frame !== null && frame.height > frame.width);
  const locked = $derived(config === null || !up);
  const direction = $derived(health?.direction ?? null);
</script>

<svelte:document onvisibilitychange={onVisibilityChange} />

<header>
  <h1>board-reader · camera control</h1>
  <label class="board">
    board
    <input
      value={boardId ?? DEFAULT_BOARD_ID}
      size="10"
      onchange={(e) => chooseBoard(e.currentTarget.value)}
    />
  </label>
  <span class="chip" class:up={connected}>
    {camera === null ? 'connecting…' : connected ? 'phone connected' : 'phone not connected'}
  </span>
  <span class="spacer"></span>
  <a href={resolve('/')}>live positions</a>
</header>

{#if failure !== null}
  <p class="failure" role="alert">{failure}</p>
{/if}

<main>
  <section class="views">
    <figure>
      <figcaption>The phone's whole view — drag the rectangle to choose what it sends</figcaption>
      {#if camera !== null && frame !== null}
        <CropEditor
          src={preview === null ? null : jpegSrc(preview.jpeg)}
          {frame}
          rect={shownCrop ?? WHOLE_FRAME}
          whole={shownCrop === null}
          onchange={(next) => (edit = next)}
        />
        <div class="actions">
          <button
            type="button"
            class="primary"
            disabled={edit === null || !up}
            onclick={() => saveCrop(edit)}>Save crop</button
          >
          <button type="button" disabled={edit === null} onclick={() => (edit = null)}>Discard</button>
          <button
            type="button"
            disabled={!up || (config?.crop === null && edit === null)}
            onclick={() => saveCrop(null)}>Send whole frame</button
          >
          {#if edit !== null}<span class="dirty">unsaved</span>{/if}
        </div>
      {:else}
        <div class="placeholder">no camera yet — open /camera on the phone</div>
      {/if}
      <p class="hint">
        Draw the rectangle around the board with a small margin: the board should reach past
        both dashed guides (at least half the width). Landscape reads best.
      </p>
      {#each warnings as warning (warning.kind)}
        <p class="warn" data-warning={warning.kind}>{FRAMING_COPY[warning.kind]}</p>
      {/each}
      {#if direction?.camera_moved}
        <p class="moved" role="alert" data-warning="camera-moved">{CAMERA_MOVED_COPY}</p>
      {/if}
    </figure>

    <figure>
      <figcaption>What the node receives</figcaption>
      {#if received !== null}
        <img
          class="received"
          src={jpegSrc(received.jpeg)}
          alt="the latest frame the node received for board {boardId}"
        />
      {:else}
        <div class="placeholder">no frame received yet</div>
      {/if}
    </figure>
  </section>

  <section class="panel">
    <h2>Camera</h2>
    <label>
      lens
      <select
        value={config?.device ?? ''}
        disabled={locked}
        onchange={(e) => send({ device: e.currentTarget.value || null })}
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
        onchange={(e) => send({ resolution: e.currentTarget.value as Resolution })}
      >
        {#each RESOLUTION_NAMES as name (name)}
          <option value={name}>{name}</option>
        {/each}
      </select>
    </label>
    <button
      type="button"
      class="stream"
      class:on={config?.streaming}
      disabled={locked}
      onclick={() => send({ streaming: !config?.streaming })}
    >
      {config?.streaming ? 'Stop streaming' : 'Start streaming'}
    </button>

    <h2>Status</h2>
    <dl>
      <dt>phone link</dt>
      <dd class:bad={camera !== null && !connected}>
        {camera === null ? '—' : connected ? 'connected' : 'not connected'}
      </dd>
      <dt>lens open</dt>
      <dd>{report?.device ?? '—'}</dd>
      <dt>frame delivered</dt>
      <dd class:bad={portrait}>
        {report?.frame ? `${report.frame.width}×${report.frame.height}` : '—'}
        {#if portrait}(portrait){/if}
      </dd>
      <dt>phone stream</dt>
      <dd>{report?.ingest ?? '—'}</dd>
      <dt>board direction</dt>
      <dd class="direction" class:bad={direction?.camera_moved}>
        {#if direction === null}
          —
        {:else}
          {whiteSideCopy(direction.white_side)}
          <span class="degrees">({Math.round(direction.degrees)}°)</span>
        {/if}
      </dd>
      <dt>frames/s received</dt>
      <dd>{health?.fps == null ? '—' : health.fps.toFixed(1)}</dd>
      <dt>last frame</dt>
      <dd>{health === null ? '—' : formatAge(health.frame_age_s)}</dd>
      <dt>recognizer</dt>
      <dd class:bad={health?.fallback}>
        {health?.recognizer ?? '—'}
        {#if health?.fallback}<span class="fallback">({health.fallback})</span>{/if}
      </dd>
      <dt>frames refused (node)</dt>
      <dd class:bad={(health?.refused ?? 0) > 0}>{health?.refused ?? '—'}</dd>
    </dl>
  </section>
</main>

<style>
  header {
    padding: 14px 20px;
    border-bottom: 1px solid #272b33;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 14px;
  }
  h1 {
    font-size: 17px;
    margin: 0;
    font-weight: 600;
  }
  h2 {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9aa3ad;
    margin: 0;
  }
  .spacer {
    flex: 1;
  }
  a {
    color: #6aa8ff;
    font-size: 13px;
  }
  .board {
    font-size: 13px;
    color: #9aa3ad;
  }
  input,
  select {
    font: inherit;
    font-size: 13px;
    background: #232833;
    color: #e6e6e6;
    border: 1px solid #2f3540;
    border-radius: 6px;
    padding: 3px 6px;
  }
  .chip {
    font-size: 13px;
    padding: 2px 8px;
    border-radius: 10px;
    background: #3a2020;
    color: #ff9a9a;
  }
  .chip.up {
    background: #1f3a24;
    color: #97e0a6;
  }
  .failure {
    margin: 0;
    padding: 8px 20px;
    background: #3a2020;
    color: #ff9a9a;
    font-size: 13px;
  }

  main {
    display: flex;
    flex-wrap: wrap;
    gap: 22px;
    padding: 20px;
    align-items: flex-start;
  }
  .views {
    flex: 1 1 640px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 22px;
    min-width: 0;
  }
  figure {
    margin: 0;
    min-width: 0;
  }
  figcaption {
    font-size: 13px;
    color: #9aa3ad;
    margin-bottom: 8px;
  }
  .received {
    display: block;
    max-width: 100%;
    max-height: 70vh;
    background: #0f1216;
    border: 1px solid #272b33;
    border-radius: 6px;
  }
  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    border: 1px dashed #2f3540;
    border-radius: 6px;
    color: #7d8590;
    font-size: 13px;
    text-align: center;
    padding: 12px;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
  }
  button {
    font: inherit;
    font-size: 13px;
    padding: 6px 12px;
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
  .hint {
    color: #9aa3ad;
    font-size: 13px;
    margin: 10px 0 6px;
  }
  .warn {
    color: #ffcf33;
    font-size: 13px;
    margin: 6px 0;
  }
  .moved {
    margin: 10px 0;
    padding: 8px 10px;
    border: 1px solid #5a2a2a;
    border-radius: 6px;
    background: #3a2020;
    color: #ff9a9a;
    font-size: 13px;
    font-weight: 600;
  }

  .panel {
    flex: 0 1 300px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    background: #1b1f26;
    border: 1px solid #272b33;
    border-radius: 8px;
    padding: 14px;
  }
  .panel label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
    color: #9aa3ad;
  }
  .stream {
    font-weight: 600;
    padding: 10px;
  }
  .stream.on {
    background: #3a2020;
    border-color: #5a2a2a;
    color: #ff9a9a;
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    margin: 0;
    font-size: 13px;
  }
  dt {
    color: #7d8590;
  }
  dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
  }
  dd.bad {
    color: #ff8f8f;
  }
  .fallback,
  .degrees {
    color: #9aa3ad;
  }
</style>
