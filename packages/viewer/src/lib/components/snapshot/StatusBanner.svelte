<script lang="ts">
  /** A non-blocking message near the board. Text only — state is carried by colour. */
  interface Props {
    kind: 'info' | 'warn';
    text: string;
  }

  const { kind, text }: Props = $props();
</script>

<div class="banner {kind}">{text}</div>

<style>
  .banner {
    font-size: 13px;
    font-weight: 600;
    padding: 9px 12px;
    border-radius: 8px;
    margin-bottom: 10px;
  }

  .info {
    background: #1f3a24;
    color: var(--ok);
  }

  .warn {
    background: #2a2410;
    color: var(--warn);
    border: 1px solid #4a3f10;
  }
</style>
