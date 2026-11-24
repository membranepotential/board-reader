<script lang="ts">
  /**
   * Saving the labelling is a first-class action on the position, so it lives with the
   * position — not buried in a collapsed stage body, and never silently absent: when the
   * node retains no photos the button stays put and explains itself.
   */
  import type { LabelState } from './reading';

  interface Props {
    state: LabelState;
    /** How many squares the operator changed — shown so a label states its own cost. */
    corrections: number;
    submitting: boolean;
    onSave: () => void;
  }

  const { state, corrections, submitting, onSave }: Props = $props();

  const label = $derived.by(() => {
    switch (state) {
      case 'saved':
        return 'Labelling saved';
      case 'stale':
        return 'Update labelling';
      default:
        return 'Save labelling';
    }
  });

  const disabled = $derived(submitting || state === 'no-retention' || state === 'saved');
</script>

<div class="row">
  <button type="button" {disabled} onclick={onSave}>{label}</button>
</div>

{#if state === 'no-retention'}
  <p class="note">
    This inference node is not retaining photos, so a labelling would have no image to
    belong to and no eval could ever replay it. Restart the node with
    <code>BR_SAVE_UPLOADS</code> set to keep labellings.
  </p>
{:else}
  <p class="note">
    Fix any wrong squares first, then save — the board as shown is kept as the true
    position for this photo{corrections ? ` (${corrections} correction(s) so far)` : ''}.
    Labellings are only ever used to <em>score</em> the reader, never to train it.
  </p>
{/if}

<style>
  .row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin: 10px 0;
  }

  button {
    background: var(--accent);
    color: #fff;
    border: 0;
    border-radius: 9px;
    padding: 10px 16px;
    font: inherit;
    font-size: 15px;
    font-weight: 600;
    min-height: 48px;
    touch-action: manipulation;
    cursor: pointer;
  }

  button:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .note {
    font-size: 12.5px;
    color: var(--muted);
    margin: 8px 0;
  }
</style>
