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
   *
   * It positions NOTHING — see the stylesheet. The bottom chrome stack in the page
   * places it, the banner above it, and the chips beside it, so those three cannot
   * drift apart or overlap.
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
  /* Inert, so a corner handle underneath it still takes the touch. It positions
     nothing: the page's bottom chrome stack owns where the messages sit and what they
     dodge, and this is a block in it. */
  .hint-bar {
    pointer-events: none;
    display: flex;
    justify-content: center;
    max-width: 100%;
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
