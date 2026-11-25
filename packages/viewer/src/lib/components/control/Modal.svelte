<script lang="ts">
  /**
   * A modal dialog on the native `<dialog>`: `showModal()` gives the focus trap, the
   * inert page behind it and Escape for free. `open` is the caller's; Escape, the
   * Close button and a click on the backdrop all ask to close through `onclose`.
   */
  import type { Snippet } from 'svelte';

  interface Props {
    open: boolean;
    title: string;
    onclose: () => void;
    children: Snippet;
  }

  const { open, title, onclose, children }: Props = $props();

  let dialog: HTMLDialogElement | undefined = $state();

  $effect(() => {
    if (dialog === undefined) return;
    if (open && !dialog.open) dialog.showModal();
    else if (!open && dialog.open) dialog.close();
  });

  /** A click on the dialog element itself, not on its content, is on the backdrop. */
  function onclick(event: MouseEvent): void {
    if (event.target === dialog) onclose();
  }
</script>

<dialog bind:this={dialog} aria-label={title} {onclick} onclose={() => onclose()}>
  <div class="content">
    <header>
      <h2>{title}</h2>
      <button type="button" class="close" aria-label="Close" onclick={() => onclose()}>×</button>
    </header>
    {#if open}{@render children()}{/if}
  </div>
</dialog>

<style>
  dialog {
    padding: 0;
    border: 1px solid #2f3540;
    border-radius: 10px;
    background: #1b1f26;
    color: #e6e6e6;
    width: min(440px, calc(100vw - 32px));
  }
  dialog::backdrop {
    background: rgb(0 0 0 / 0.6);
  }
  .content {
    padding: 16px;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  h2 {
    font-size: 15px;
    margin: 0;
    font-weight: 600;
  }
  .close {
    font: inherit;
    font-size: 18px;
    line-height: 1;
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid #2f3540;
    background: #232833;
    color: #e6e6e6;
    cursor: pointer;
  }
</style>
