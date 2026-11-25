<script lang="ts">
  /** The body of the capture stage: what the browser did to the photo before posting it. */
  import KvTable from '#lib/components/snapshot/KvTable.svelte';
  import {
    MAX_SOURCE_PX,
    captureRows,
    type CaptureFacts,
    type SourceImage
  } from '#lib/components/snapshot/photo.ts';

  interface Props {
    image: SourceImage | null;
    facts: CaptureFacts | null;
  }

  const { image, facts }: Props = $props();

  const rows = $derived(image === null || facts === null ? [] : captureRows(image, facts));
</script>

<p class="note">
  The photo is downscaled to a {MAX_SOURCE_PX} px long side in the browser; that downscaled
  image is exactly what is POSTed to <code>/estimate</code>, so this stage describes the very
  pixels the server reads.
</p>
<KvTable {rows} />

<style>
  .note {
    font-size: 12.5px;
    color: var(--muted);
    margin: 8px 0;
  }
</style>
