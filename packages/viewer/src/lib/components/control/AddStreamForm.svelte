<script lang="ts">
  /**
   * The "Add stream" form: a video or live stream URL and, for a video, where to
   * start. It checks the URL with the generated `AddStreamMessage` validator, so it
   * refuses what the node would refuse, and hands the message to `onadd`. The page
   * sends it and says how it went (`status`).
   */
  import type { AddStreamMessage } from '#lib/protocol/generated.ts';
  import { addStreamMessage, type AddStatus } from './streams-link.ts';
  import { parseStartTime } from './start-time.ts';

  interface Props {
    /** The streams socket is open: a submit can go out. */
    ready: boolean;
    status: AddStatus;
    onadd: (message: AddStreamMessage) => void;
  }

  const { ready, status, onadd }: Props = $props();

  let url = $state('');
  let start = $state('');
  let problem = $state<string | null>(null);

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    const startTime = parseStartTime(start);
    if (!startTime.ok) {
      problem = startTime.reason;
      return;
    }
    const built = addStreamMessage(url, startTime.seconds);
    if (!built.ok) {
      problem = built.reason;
      return;
    }
    problem = null;
    onadd(built.message);
  }

  const shown = $derived(problem ?? (status.kind === 'failed' ? status.reason : null));
</script>

<form onsubmit={submit}>
  <label>
    video or live stream URL
    <input
      type="url"
      bind:value={url}
      placeholder="https://www.youtube.com/watch?v=…"
      required
      autocomplete="off"
    />
  </label>
  <label>
    start at (optional, a video only)
    <input bind:value={start} placeholder="0:00 or 1:23:45" autocomplete="off" inputmode="numeric" />
  </label>
  <p class="hint">The node plays it as a new camera; its card appears once the stream runs.</p>
  {#if !ready}
    <p class="hint">connecting to the node…</p>
  {/if}
  {#if shown !== null}
    <p class="warn" role="alert">{shown}</p>
  {/if}
  <div class="actions">
    <button type="submit" class="primary" disabled={!ready || status.kind === 'sending'}>
      {status.kind === 'sending' ? 'Adding…' : 'Add stream'}
    </button>
  </div>
</form>

<style>
  form {
    display: grid;
    gap: 12px;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
    color: #9aa3ad;
  }
  input {
    font: inherit;
    font-size: 13px;
    background: #232833;
    color: #e6e6e6;
    border: 1px solid #2f3540;
    border-radius: 6px;
    padding: 6px 8px;
  }
  p {
    margin: 0;
    font-size: 12px;
  }
  .hint {
    color: #6f7782;
  }
  .warn {
    color: #ffcf33;
  }
  .actions {
    display: flex;
    justify-content: flex-end;
  }
  button {
    font: inherit;
    font-size: 13px;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid #2a6ef0;
    background: #2a6ef0;
    color: #fff;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.6;
    cursor: default;
  }
</style>
