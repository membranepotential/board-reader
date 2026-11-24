<script lang="ts">
  /**
   * One pipeline stage: a plain expanding section, not a card.
   *
   * The summary always shows the stage's state and headline numbers; the body holds the
   * full evidence and stays collapsed until asked for. Every body exists on every run —
   * nothing here is gated behind an opt-in checkbox, because diagnostics are the point
   * of this page. State is text + colour; no emoji anywhere.
   */
  import type { Snippet } from 'svelte';
  import { CHIP_LABEL, type StageMeta, type StageStatus } from './pipeline';

  interface Props {
    stage: StageMeta;
    status: StageStatus;
    /** Bindable so a failed detection can open its own stage and show the way out. */
    open?: boolean;
    children: Snippet;
  }

  let { stage, status, open = $bindable(false), children }: Props = $props();
</script>

<section class="step" data-state={status.state}>
  <details bind:open>
    <summary>
      <span class="caret" aria-hidden="true">&#9656;</span>
      <span class="idx">{stage.index}</span>
      <span class="name">{stage.title}</span>
      <span class="chip">{CHIP_LABEL[status.state]}</span>
      <span class="sum">{status.summary}</span>
    </summary>
    <div class="body">{@render children()}</div>
  </details>
</section>

<style>
  .step {
    border-bottom: 1px solid var(--line);
  }

  .step:first-child {
    border-top: 1px solid var(--line);
  }

  summary {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    padding: 9px 2px;
    cursor: pointer;
    list-style: none;
    touch-action: manipulation;
  }

  summary::-webkit-details-marker {
    display: none;
  }

  summary:hover .name {
    color: #fff;
  }

  .caret {
    color: var(--dim);
    font-size: 10px;
    width: 10px;
    transition: transform 0.12s;
  }

  details[open] > summary .caret {
    transform: rotate(90deg);
  }

  .idx {
    color: var(--dim);
    font-size: 12px;
    min-width: 12px;
  }

  .name {
    font-weight: 600;
    font-size: 14px;
  }

  .chip {
    margin-left: auto;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--dim);
  }

  [data-state='run'] .chip {
    color: var(--accent-soft);
  }

  [data-state='ok'] .chip {
    color: var(--ok);
  }

  [data-state='warn'] .chip {
    color: var(--warn);
  }

  [data-state='fail'] .chip {
    color: var(--bad);
  }

  .sum {
    flex-basis: 100%;
    color: var(--muted);
    font-size: 12.5px;
    padding-left: 22px;
    word-break: break-word;
  }

  .body {
    padding: 0 2px 16px 22px;
  }

  .body > :global(:first-child) {
    margin-top: 4px;
  }
</style>
