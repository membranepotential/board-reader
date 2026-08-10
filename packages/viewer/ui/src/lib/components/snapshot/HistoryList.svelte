<script lang="ts">
  /**
   * Every read this session merged with every photo the node still retains, newest first.
   *
   * Selecting a session read restores the whole pipeline view — photo, corners, geometry,
   * crops, distributions — so two reads can be compared without re-shooting either.
   * Selecting a node-held photo loads it and reads it again for real, so the stages
   * always describe the photo actually on screen.
   */
  import { uploadPhotoUrl } from '$lib/api';
  import type { UploadRecord } from '$lib/protocol';
  import {
    readFlags,
    readHeadline,
    readMeta,
    uploadFlags,
    uploadTime,
    type HistoryRow
  } from './history';

  interface Props {
    rows: readonly HistoryRow[];
    note: string;
    activeId: number | null;
    clearable: boolean;
    onRestore: (id: number) => void;
    onLoadUpload: (upload: UploadRecord) => void;
    onClear: () => void;
  }

  const { rows, note, activeId, clearable, onRestore, onLoadUpload, onClear }: Props = $props();
</script>

<section class="hist">
  <div class="head">
    <h3>Recent positions</h3>
    <span class="spacer"></span>
    {#if clearable}
      <button class="text-btn" type="button" onclick={onClear}>Clear this session</button>
    {/if}
  </div>
  <p class="note">{note}</p>

  {#if rows.length === 0}
    <div class="empty">
      no reads yet — every position read this session, and every photo the inference node
      retains, lands here
    </div>
  {:else}
    <div class="list">
      {#each rows as row (row.key)}
        {#if row.kind === 'session'}
          {@const flags = readFlags(row.read, row.upload)}
          <button
            class="row"
            class:active={row.read.id === activeId}
            type="button"
            title="Restore this read — photo, corners, crops and distributions"
            onclick={() => onRestore(row.read.id)}
          >
            <img src={row.read.image.dataUrl} alt="" />
            <span class="body">
              <span class="fen">{readHeadline(row.read)}</span>
              <span class="meta">
                {readMeta(row.read)} · <span class="tag">this session</span>
                {#if flags.length}
                  · <span class="flag">{flags.join(', ')}</span>
                {/if}
              </span>
            </span>
          </button>
        {:else}
          {@const flags = uploadFlags(row.upload)}
          {@const at = uploadTime(row.upload)}
          <button
            class="row"
            type="button"
            title="Load this photo from the inference node and read it again"
            onclick={() => onLoadUpload(row.upload)}
          >
            <img loading="lazy" src={uploadPhotoUrl(row.upload.id, 160)} alt="" />
            <span class="body">
              <span class="fen">{row.upload.placement || 'no placement recorded'}</span>
              <span class="meta">
                {at ? at.toLocaleString() : '—'} ·
                <span class="tag">on the inference node</span>
                {#if flags.length}
                  · <span class="flag">{flags.join(', ')}</span>
                {/if}
              </span>
            </span>
          </button>
        {/if}
      {/each}
    </div>
  {/if}
</section>

<style>
  .hist {
    border-top: 1px solid var(--line);
    padding-top: 10px;
    margin-top: 18px;
  }

  .head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .spacer {
    flex: 1;
  }

  .note {
    font-size: 11.5px;
    color: var(--dim);
    margin: 0 0 8px;
  }

  .list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 320px;
    overflow-y: auto;
  }

  .row {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    text-align: left;
    background: none;
    border: 0;
    border-left: 2px solid transparent;
    padding: 6px 8px;
    color: var(--fg);
    font: inherit;
    min-height: 48px;
    touch-action: manipulation;
    cursor: pointer;
  }

  .row:hover {
    background: #1b1f26;
  }

  .row.active {
    border-left-color: var(--accent-soft);
    background: #1b1f26;
  }

  .row img {
    width: 56px;
    height: 42px;
    object-fit: cover;
    border-radius: 4px;
    flex: 0 0 auto;
    background: #0f1216;
  }

  .body {
    min-width: 0;
    flex: 1;
    display: block;
  }

  .fen {
    display: block;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    color: #c8cdd4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .meta {
    display: block;
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
  }

  /* Where a row comes from is a permanent property of the row, so it is spelled out on
     every one of them rather than implied by ordering. */
  .tag {
    color: var(--dim);
  }

  .flag {
    color: var(--warn);
  }

  .empty {
    font-size: 12.5px;
    color: var(--muted);
    padding: 4px 0;
  }

  .text-btn {
    background: none;
    border: 0;
    color: var(--accent-soft);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    padding: 4px 0;
    text-decoration: underline;
    touch-action: manipulation;
    cursor: pointer;
  }
</style>
