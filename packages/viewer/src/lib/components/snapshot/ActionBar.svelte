<script lang="ts">
  /**
   * The run control. On a phone it is fixed to the bottom of the screen so it stays
   * reachable without scrolling out of the pipeline; it publishes its real height
   * (`height`, bindable) because the page pads for it and the inspector sheet docks
   * directly above it.
   */
  interface Props {
    disabled: boolean;
    busy: string | null;
    onRead: () => void;
    height?: number;
  }

  let { disabled, busy, onRead, height = $bindable(64) }: Props = $props();

  let bar: HTMLDivElement | undefined = $state();

  $effect(() => {
    const el = bar;
    if (!el) return;
    const measure = () => (height = Math.round(el.getBoundingClientRect().height));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  });
</script>

<div class="action-bar" bind:this={bar}>
  <button type="button" {disabled} onclick={onRead}>Read position</button>
  {#if busy !== null}<span class="busy">{busy}</span>{/if}
</div>

<style>
  .action-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 12px 0;
  }

  button {
    flex: 1 1 auto;
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

  .busy {
    font-size: 13px;
    color: var(--muted);
    white-space: nowrap;
  }

  @media (max-width: 759px) {
    .action-bar {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 30;
      margin: 0;
      padding: 8px 16px calc(8px + env(safe-area-inset-bottom));
      background: var(--panel);
      border-top: 1px solid var(--line);
      box-shadow: 0 -6px 20px rgba(0, 0, 0, 0.4);
    }
  }
</style>
