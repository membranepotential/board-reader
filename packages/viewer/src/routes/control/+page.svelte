<script lang="ts">
  /**
   * board-reader · camera control — the laptop side of the "dumb camera" (DESIGN D11).
   *
   * The phone (/camera) only films and streams; this page chooses what it films. It
   * shows one card per camera the node knows (`CamerasLink`, `WS /cameras`), connected
   * or not, keyed by DEVICE id, so forgetting a camera and a new phone taking its board
   * id never leaves a card on the wrong camera. Each card (`CameraCard`) talks to its
   * own board over that board's socket.
   *
   * The top bar adds cameras: "Add camera" opens the QR code a phone scans to open
   * /camera, "Add stream" a form that asks the node to play a video or live stream as
   * a camera (`StreamsLink`, `WS /streams`, open only while the form is), above the
   * node's streams with a Remove button each (`StreamList`). The list socket closes
   * while the tab is hidden, like the cards' sockets.
   */
  import { onDestroy, onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import type {
    AddStreamMessage,
    CameraEntry,
    CamerasDown,
    StreamEntry,
    StreamsDown
  } from '#lib/protocol/generated.ts';
  import { CamerasLink } from '#lib/control/cameras.ts';
  import CameraCard from '#lib/components/control/CameraCard.svelte';
  import CameraQr from '#lib/components/control/CameraQr.svelte';
  import AddStreamForm from '#lib/components/control/AddStreamForm.svelte';
  import StreamList from '#lib/components/control/StreamList.svelte';
  import Modal from '#lib/components/control/Modal.svelte';
  import { StreamsLink, type AddStatus } from '#lib/components/control/streams-link.ts';

  /** Every camera the node knows; `null` until its first list. */
  let cameras = $state.raw<readonly CameraEntry[] | null>(null);
  /** Why the node refused the last rename or forget, or why the list is down. */
  let listFailure = $state<string | null>(null);
  let modal = $state<'camera' | 'stream' | null>(null);
  let streamsUp = $state(false);
  let adding = $state.raw<AddStatus>({ kind: 'idle' });
  /** The node's streams while the Add stream form is open; `null` until its first list. */
  let streamList = $state.raw<readonly StreamEntry[] | null>(null);
  /** Why the last removal did not go through. */
  let removeFailure = $state<string | null>(null);
  let list: CamerasLink | null = null;

  onMount(() => {
    const next = new CamerasLink({
      onMessage: receiveList,
      onLink: (isUp, detail) => {
        listFailure = isUp ? null : `the node is not reachable (${detail}) — reconnecting…`;
      }
    });
    list = next;
    if (document.visibilityState === 'visible') next.start();
    return () => {
      next.stop();
      list = null;
    };
  });

  /** The streams socket, open while the Add stream form is. */
  let streams: StreamsLink | null = null;

  function openStreamForm(): void {
    modal = 'stream';
    adding = { kind: 'idle' };
    streams?.stop();
    streams = new StreamsLink({
      onMessage: receiveStreams,
      onLink: (isUp, detail) => {
        streamsUp = isUp;
        if (!isUp && adding.kind === 'sending') adding = { kind: 'failed', reason: `the node is not reachable (${detail})` };
      }
    });
    streams.start();
  }

  function closeModal(): void {
    modal = null;
    adding = { kind: 'idle' };
    streams?.stop();
    streams = null;
    streamsUp = false;
    streamList = null;
    removeFailure = null;
  }

  onDestroy(() => streams?.stop());

  function onVisibilityChange(): void {
    if (document.visibilityState === 'visible') list?.start();
    else list?.stop();
  }

  function receiveList(message: CamerasDown): void {
    if (message.type === 'refused') listFailure = `the node refused: ${message.reason}`;
    else cameras = message.cameras;
  }

  function receiveStreams(message: StreamsDown): void {
    if (message.type === 'streams') streamList = message.streams;
    if (adding.kind !== 'sending') {
      if (message.type === 'refused') removeFailure = `the node refused: ${message.reason}`;
      return;
    }
    if (message.type === 'refused') {
      adding = { kind: 'failed', reason: `the node refused: ${message.reason}` };
    } else if (message.streams.some((stream) => stream.url === (adding.kind === 'sending' ? adding.url : null))) {
      closeModal();
    }
  }

  function addStream(message: AddStreamMessage): void {
    adding =
      streams !== null && streams.add(message)
        ? { kind: 'sending', url: message.url }
        : { kind: 'failed', reason: 'not added: the node is not connected' };
  }

  function removeStream(stream: StreamEntry): void {
    const camera = stream.board_id === null ? 'its camera' : `its camera ${stream.board_id}`;
    const asked =
      `Remove the stream ${stream.url}? The node stops playing it and forgets ${camera}, ` +
      'whose board id may go to the next new camera.';
    if (!window.confirm(asked)) return;
    removeFailure = streams !== null && streams.remove(stream.id) ? null : 'not removed: the node is not connected';
  }

  function rename(entry: CameraEntry, name: string | null): boolean {
    const sent = list !== null && list.rename(entry.device_id, name);
    listFailure = sent ? null : 'not renamed: the node is not connected';
    return sent;
  }

  function forget(entry: CameraEntry): void {
    const label = entry.name ?? entry.board_id;
    const asked =
      `Forget camera ${label}? Its board id ${entry.board_id} and its settings are dropped, ` +
      'and the id may go to the next new phone. If this phone comes back, it is a new camera.';
    if (!window.confirm(asked)) return;
    listFailure = list !== null && list.forget(entry.device_id) ? null : 'not forgotten: the node is not connected';
  }
</script>

<svelte:document onvisibilitychange={onVisibilityChange} />

<header>
  <h1>board-reader · camera control</h1>
  <button type="button" class="primary" onclick={() => (modal = 'camera')}>Add camera</button>
  <button type="button" onclick={openStreamForm}>Add stream</button>
  <a class="live" href={resolve('/')}>live positions</a>
</header>

{#if listFailure !== null}
  <p class="failure" role="alert">{listFailure}</p>
{/if}

<main>
  {#if cameras === null}
    <p class="hint">asking the node for its cameras…</p>
  {:else if cameras.length === 0}
    <p class="hint">no camera yet — press Add camera and scan the code with a phone, or Add stream</p>
  {:else}
    {#each cameras as entry (entry.device_id)}
      <CameraCard {entry} onrename={(name) => rename(entry, name)} onforget={() => forget(entry)} />
    {/each}
  {/if}
</main>

<Modal open={modal === 'camera'} title="Add a camera: scan this with a phone" onclose={closeModal}>
  <CameraQr />
</Modal>
<Modal open={modal === 'stream'} title="Add a video or live stream" onclose={closeModal}>
  <AddStreamForm ready={streamsUp} status={adding} onadd={addStream} />
  {#if streamList !== null}
    <section class="streams" aria-label="the node's streams">
      <h3>streams on the node</h3>
      {#if removeFailure !== null}
        <p class="warn" role="alert">{removeFailure}</p>
      {/if}
      <StreamList streams={streamList} onremove={removeStream} />
    </section>
  {/if}
</Modal>

<style>
  header {
    padding: 14px 20px;
    border-bottom: 1px solid #272b33;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 14px;
  }
  h1 {
    font-size: 17px;
    margin: 0 8px 0 0;
    font-weight: 600;
  }
  .live {
    margin-left: auto;
    font-size: 13px;
    color: #6aa8ff;
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
  button.primary {
    background: #2a6ef0;
    border-color: #2a6ef0;
    color: #fff;
  }
  .failure {
    margin: 0;
    padding: 8px 20px;
    background: #3a2020;
    color: #ff9a9a;
    font-size: 13px;
  }
  /* The live page's grid: cards side by side, as many as fit. */
  main {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr));
    gap: 22px;
    padding: 20px;
    align-items: start;
  }
  .hint {
    color: #9aa3ad;
    font-size: 13px;
    margin: 0 0 10px;
  }
  .streams {
    display: grid;
    gap: 8px;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #2f3540;
  }
  .streams h3 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: #9aa3ad;
  }
  .warn {
    margin: 0;
    font-size: 12px;
    color: #ffcf33;
  }
</style>
