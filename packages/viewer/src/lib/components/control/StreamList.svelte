<script lang="ts">
  /**
   * The node's streams (`WS /streams`): each one's URL, start position and state,
   * and a Remove button. Removing a stream stops its process and forgets its
   * camera, so the caller confirms before it sends `StreamsLink.remove(stream.id)`.
   *
   *   <StreamList streams={message.streams} onremove={(stream) => …} />
   */
  import type { StreamEntry } from '#lib/protocol/generated.ts';
  import { formatStart, streamStatus } from '#lib/control/streams.ts';

  interface Props {
    /** The latest `streams` list the node pushed, in the order they were added. */
    streams: readonly StreamEntry[];
    /** The operator pressed this stream's Remove. */
    onremove: (stream: StreamEntry) => void;
  }

  const { streams, onremove }: Props = $props();
</script>

{#if streams.length === 0}
  <p class="hint">no stream yet</p>
{:else}
  <ul>
    {#each streams as stream (stream.id)}
      {@const status = streamStatus(stream)}
      <li>
        <div class="what">
          <span class="url" title={stream.url}>{stream.url}</span>
          {#if stream.start_s !== null}
            <span class="start">from {formatStart(stream.start_s)}</span>
          {/if}
          <span class="status {status.tone}">{status.text}</span>
        </div>
        <button type="button" onclick={() => onremove(stream)} aria-label="Remove {stream.url}">
          Remove
        </button>
      </li>
    {/each}
  </ul>
{/if}

<style>
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 8px;
  }
  li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    border: 1px solid #2f3540;
    border-radius: 6px;
    background: #232833;
  }
  .what {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
    flex: 1;
    font-size: 12px;
  }
  .url {
    color: #e6e6e6;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .start,
  .hint {
    color: #6f7782;
  }
  .hint {
    margin: 0;
    font-size: 12px;
  }
  .status.ok {
    color: #6fd08c;
  }
  .status.wait {
    color: #9aa3ad;
  }
  .status.bad {
    color: #ffcf33;
  }
  button {
    font: inherit;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #2f3540;
    background: #1b1f26;
    color: #e6e6e6;
    cursor: pointer;
  }
</style>
