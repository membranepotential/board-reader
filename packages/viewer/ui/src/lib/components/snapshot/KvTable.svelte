<script lang="ts">
  import type { KvRow } from './kv';

  const { rows }: { rows: readonly KvRow[] } = $props();
</script>

{#if rows.length}
  <table class="kv">
    <tbody>
      {#each rows as row (row.key)}
        <tr>
          <th scope="row">{row.key}</th>
          <td class={row.tone}>{row.value}</td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}

<style>
  .kv {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 10px 0;
  }

  th,
  td {
    text-align: left;
    padding: 4px 6px 4px 0;
    vertical-align: top;
  }

  th {
    color: var(--muted);
    font-weight: 500;
    width: 44%;
  }

  td {
    /* Values are quoted numbers and ids an operator copies out of a bug report. */
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12.5px;
    -webkit-user-select: text;
    user-select: text;
  }

  td.ok {
    color: var(--ok);
  }

  td.warn {
    color: var(--warn);
  }

  td.bad {
    color: var(--bad);
  }
</style>
