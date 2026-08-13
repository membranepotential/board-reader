<script lang="ts">
  /**
   * What the node just said, over the picture. Inert throughout, so it can never
   * intercept a corner handle underneath it.
   *
   * It positions NOTHING: it is a block in the page's bottom chrome stack, which owns
   * where the messages sit and what they dodge. It used to float top-centre, and the
   * operator's verdict was that the top of the picture — the part being aimed — must
   * stay clean.
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
    pointer-events: none;
    max-width: min(460px, 100%);
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
