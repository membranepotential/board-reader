<script lang="ts">
  /**
   * What the node just said, over the picture. Inert throughout, so it can never
   * intercept a corner handle underneath it.
   */
  import type { BannerMessage } from './state';

  interface Props {
    banner: BannerMessage | null;
  }

  const { banner }: Props = $props();
</script>

{#if banner !== null}
  <div class="banner {banner.kind}">{banner.text}</div>
{/if}

<style>
  .banner {
    position: absolute;
    z-index: 7;
    pointer-events: none;
    top: calc(44px + env(safe-area-inset-top));
    left: calc((100% - var(--rail)) / 2);
    transform: translateX(-50%);
    width: min(460px, calc(100% - var(--rail) - 24px));
    font-size: 13px;
    padding: 9px 12px;
    border-radius: 8px;
    white-space: pre-wrap;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.55);
  }

  .banner.ok {
    background: #1f3a24;
    color: var(--ok);
  }

  .banner.err {
    background: #3a2020;
    color: var(--bad);
  }

  .banner.warn {
    background: #3a2f18;
    color: var(--warn);
  }
</style>
