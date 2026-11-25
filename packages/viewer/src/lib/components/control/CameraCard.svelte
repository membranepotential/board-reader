<script lang="ts">
  /**
   * One camera on /control, as a card like the live page's: the picture on top, its
   * name, settings, Auto-frame and stats below, and Forget once it is disconnected.
   *
   * The card talks to its camera's board over that board's socket (`ControlLink`):
   * the node pushes the camera state on connect and on every change, the phone's
   * whole-frame preview, the latest received frame and a health tick, so nothing
   * here polls. The page keys cards by DEVICE id, and a camera's board id never
   * changes, so the socket is opened once per card. It closes while the tab is
   * hidden, which also tells the phone to stop sending previews nobody sees.
   *
   * The picture shows what the node receives (the cropped frame). Below it, Framing
   * (`Framing.svelte`) holds the Auto | Manual toggle; in Manual, Crop swaps in the
   * phone's whole frame with the crop rectangle (`CropEditor`). The rectangle is
   * edited locally and sent only on Save, so a pushed state never yanks it out from
   * under the pointer. Every other setting is sent the moment it changes. Each
   * `configure` is the whole config, built from the last one this card knows and
   * applied here at once, so a second change made before the node's answer arrives
   * builds on the first instead of undoing it.
   */
  import { onMount } from 'svelte';
  import type {
    CameraConfig,
    CameraEntry,
    ControlDown,
    CropRect,
    FrameMessage,
    HealthMessage,
    PreviewMessage,
    StateMessage
  } from '#lib/protocol/generated.ts';
  import { ControlLink } from '#lib/control/link.ts';
  import { jpegSrc } from '#lib/control/messages.ts';
  import { WHOLE_FRAME } from '#lib/control/crop.ts';
  import { FRAMING_COPY, framingWarnings } from '#lib/control/framing.ts';
  import { CAMERA_MOVED_COPY } from '#lib/control/direction.ts';
  import Card from '#lib/components/Card.svelte';
    import CameraSettings from './CameraSettings.svelte';
  import CameraStats from './CameraStats.svelte';
  import CropEditor from './CropEditor.svelte';
  import CroppedFrame from './CroppedFrame.svelte';
  import Framing from './Framing.svelte';
  import NameField from './NameField.svelte';

  interface Props {
    entry: CameraEntry;
    /** Rename this camera; `false` when the node is not connected. */
    onrename: (name: string | null) => boolean;
    /** Forget this (disconnected) camera; the page asks first. */
    onforget: () => void;
  }

  const { entry, onrename, onforget }: Props = $props();

  let camera = $state.raw<StateMessage | null>(null);
  let preview = $state.raw<PreviewMessage | null>(null);
  let received = $state.raw<FrameMessage | null>(null);
  let health = $state.raw<HealthMessage | null>(null);
  /** Whether this card's socket to the node is open. */
  let up = $state(false);
  let failure = $state<string | null>(null);
  /** Crop edit mode was entered: the whole frame with the rectangle instead of the cropped frame. */
  let cropping = $state(false);
  /** The rectangle being edited and not yet saved; `null` shows the node's. */
  let edit = $state<CropRect | null>(null);
  let link: ControlLink | null = null;

  onMount(() => {
    const next = new ControlLink({
      boardId: entry.board_id,
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
      link = null;
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
    if (send({ crop })) leaveCropMode();
  }

  function leaveCropMode(): void {
    edit = null;
    cropping = false;
  }

  /**
   * The Auto | Manual toggle. Auto sets `auto` and the node frames the camera afresh;
   * a crop being edited is dropped, since Auto chooses the crop.
   */
  function setAuto(on: boolean): void {
    if (on) leaveCropMode();
    send({ auto: on });
  }

  function reframe(): void {
    failure = link !== null && link.reframe() ? null : 'not sent: the node is not connected';
  }

  const label = $derived(entry.name ?? entry.board_id);
  const config = $derived(camera?.config ?? null);
  /** The camera's whole frame: what the phone says it delivers, else the preview's size. */
  const frame = $derived(camera?.report?.frame ?? preview ?? null);
  const shownCrop = $derived(edit ?? config?.crop ?? null);
  const warnings = $derived(framingWarnings(shownCrop, frame));
  const locked = $derived(config === null || !up);
  /** Crop edit mode only in Manual: if the node (or another page) switches to Auto, it ends. */
  const editing = $derived(cropping && config?.auto === false);
  const moved = $derived(health?.direction?.camera_moved ?? false);
</script>

<svelte:document onvisibilitychange={onVisibilityChange} />

<Card class="camera-card" data-testid="camera-card" data-device={entry.device_id} aria-label="camera {label}">
  {#snippet heading()}
    <span class="label">
      {label}
      {#if entry.name !== null}<span class="board-id">{entry.board_id}</span>{/if}
    </span>
    <span class="chip" class:up={entry.connected}>{entry.connected ? 'connected' : 'not connected'}</span>
  {/snippet}

  <div class="view">
    {#if editing && frame !== null}
      <CropEditor
        src={preview === null ? null : jpegSrc(preview.jpeg)}
        {frame}
        rect={shownCrop ?? WHOLE_FRAME}
        whole={shownCrop === null}
        onchange={(next) => (edit = next)}
      />
    {:else if editing}
      <div class="placeholder">no picture from this camera yet</div>
    {:else}
      <CroppedFrame
        frame={received}
        {preview}
        crop={config?.crop ?? null}
        streaming={config?.streaming ?? false}
        boardId={entry.board_id}
      />
    {/if}
  </div>

  <section class="framing" aria-label="Framing">
    <h3>Framing</h3>
    <Framing
      {camera}
      {locked}
      cropping={editing}
      dirty={edit !== null}
      onauto={setAuto}
      onreframe={reframe}
      oncrop={() => (cropping = true)}
      onsave={() => saveCrop(edit)}
      oncancel={leaveCropMode}
      onwhole={() => saveCrop(null)}
    />
  </section>
  {#each warnings as warning (warning.kind)}
    <p class="warn" data-warning={warning.kind}>{FRAMING_COPY[warning.kind]}</p>
  {/each}
  {#if moved}
    <p class="moved" role="alert" data-warning="camera-moved">{CAMERA_MOVED_COPY}</p>
  {/if}
  {#if failure !== null}
    <p class="failure" role="alert">{failure}</p>
  {/if}

  <section aria-label="settings">
    <NameField name={entry.name} boardId={entry.board_id} {onrename} />
    <CameraSettings {config} report={camera?.report ?? null} {locked} onsend={send} />
  </section>
  <section aria-label="stats">
    <h3>Stats</h3>
    <CameraStats {camera} {health} />
  </section>
  {#if !entry.connected}
    <button type="button" class="forget" title="drop this camera, its board id and its settings" onclick={onforget}
      >Forget</button
    >
  {/if}
</Card>

<style>
  .label {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .board-id {
    font-weight: 400;
    font-size: 12px;
    color: #6b7280;
    margin-left: 4px;
  }
  .chip {
    font-size: 12px;
    font-weight: 400;
    padding: 1px 7px;
    border-radius: 9px;
    white-space: nowrap;
    background: #3a2020;
    color: #ff9a9a;
  }
  .chip.up {
    background: #243;
    color: #9e9;
  }
  section {
    display: grid;
    gap: 10px;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #272b33;
  }
  h3 {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9aa3ad;
    margin: 0;
  }
  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 16 / 9;
    border: 1px dashed #2f3540;
    border-radius: 6px;
    color: #7d8590;
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
  .forget {
    margin-top: 14px;
    color: #ff9a9a;
    border-color: #5a2a2a;
  }
  p {
    margin: 8px 0 0;
    font-size: 13px;
  }
  .warn {
    color: #ffcf33;
  }
  .moved,
  .failure {
    padding: 8px 10px;
    border: 1px solid #5a2a2a;
    border-radius: 6px;
    background: #3a2020;
    color: #ff9a9a;
  }
  .moved {
    font-weight: 600;
  }
</style>
