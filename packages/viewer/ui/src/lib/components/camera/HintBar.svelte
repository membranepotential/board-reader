<script lang="ts">
  /**
   * One sentence, chosen by priority (`hintFor`): the single most useful thing to
   * say right now. It exists partly because `title` is unreachable on a touch
   * screen — a disabled control's reason has to be readable somewhere the operator
   * can actually see it.
   *
   * The hand-written page had to guard this against rebuilding at 5 Hz, which moved
   * the recovery button out from under a finger. Nothing guards it here and nothing
   * needs to: the `{#if}` block owns the button's identity, so it survives every
   * rewording of the sentence and is only ever created when the offer itself
   * appears.
   */
  import type { Hint } from './state';

  interface Props {
    hint: Hint | null;
    onLookAgain: () => void;
  }

  const { hint, onLookAgain }: Props = $props();
</script>

{#if hint !== null}
  <div class="hint-bar">
    <span>
      {hint.text}
      {#if hint.action === 'look-again'}
        <button type="button" onclick={onLookAgain}>look again</button>
      {/if}
    </span>
  </div>
{/if}

<style>
  /* Inert, so a corner handle underneath it still takes the touch.

     Centred in whatever the button cluster leaves: `--cluster-w` keeps it clear of
     the landscape column on the right, `--cluster-h` of the portrait row below.

     The home-indicator inset is budgeted ONCE, and which rule budgets it depends on
     the orientation: in landscape the cluster reserves no height, so this bar clears
     the indicator itself; in portrait `--cluster-h` already contains it, and adding
     it again here would float the sentence a second inset above the buttons. */
  .hint-bar {
    position: absolute;
    z-index: 7;
    pointer-events: none;
    left: 0;
    right: var(--cluster-w);
    bottom: max(10px, env(safe-area-inset-bottom));
    display: flex;
    justify-content: center;
    padding: 0 12px 0 calc(12px + env(safe-area-inset-left));
  }

  @media (orientation: portrait) {
    .hint-bar {
      bottom: calc(var(--cluster-h) + 10px);
    }
  }

  .hint-bar > span {
    background: rgba(20, 23, 28, 0.85);
    color: var(--fg);
    font-size: 13px;
    border-radius: 999px;
    padding: 8px 14px;
    text-align: center;
    max-width: 100%;
  }

  /* The recovery link must NOT be inert: after automatic detection gives up, "look
     again" is the only way back other than dragging a handle. It is a real touch
     target, 44px like the handles — a past bug left this one escape hatch at 31px,
     which is unreachable on a phone. */
  .hint-bar button {
    pointer-events: auto;
    min-height: 44px;
    padding: 0 10px;
    background: none;
    border: 0;
    color: var(--accent-soft);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    text-decoration: underline;
    touch-action: manipulation;
    cursor: pointer;
  }
</style>
