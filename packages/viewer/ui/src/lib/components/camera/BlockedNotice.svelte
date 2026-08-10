<script lang="ts">
  /**
   * The explainer, shown when the camera is unreachable. With no camera there is no
   * viewfinder, so this takes the whole screen and the stage is not rendered at all.
   *
   * Every branch is real markup rather than an interpolated HTML string: the old
   * page put the browser's own `err.message` through `innerHTML`, which is both a
   * needless injection surface and the reason the copy could not be reviewed as
   * markup.
   */
  import type { BlockedReason } from './state';

  interface Props {
    reason: BlockedReason;
    /** The port this page was served on, so the two recipes below are copy-pasteable. */
    port: string;
  }

  const { reason, port }: Props = $props();

  const REFUSALS: Readonly<Record<string, string>> = {
    NotAllowedError:
      "The camera permission was denied. Allow it for this site in the browser's site " +
      'settings and reload.',
    NotFoundError: 'No camera was found on this device.',
    NotReadableError: 'The camera is in use by another app. Close it and reload.',
    OverconstrainedError:
      'No camera matched the request. This page asks for the rear camera; a device with ' +
      'only a front camera will land here.'
  };

  const title = $derived(
    reason.kind === 'insecure'
      ? 'Camera blocked: insecure context'
      : reason.kind === 'unsupported'
        ? 'Camera unsupported'
        : 'Camera unavailable'
  );
</script>

<section class="blocked">
  <h2>{title}</h2>

  {#if reason.kind === 'insecure'}
    <p>
      This page is served over <code>http</code>, and browsers only hand out the camera in a
      <b>secure context</b>. Nothing is wrong with your permissions — the camera was never
      offered.
    </p>
    <p>Two ways in:</p>
    <ol>
      <li>
        <b>Serve the viewer over https.</b> Restart it with
        <code>board-reader-viewer --host 0.0.0.0 --ssl</code>, then open the
        <code>https://…/camera</code> address it prints. Your phone warns once about the
        self-signed certificate; accept it and the camera works from then on.
      </li>
      <li>
        <b>Android over USB.</b> Plug the phone in and run
        <code>adb reverse tcp:{port} tcp:{port}</code> on this machine, then open
        <code>http://localhost:{port}/camera</code> on the phone — <code>localhost</code> counts
        as secure whatever the transport.
      </li>
    </ol>
  {:else if reason.kind === 'unsupported'}
    <p>This browser has no <code>getUserMedia</code>.</p>
  {:else}
    <p>{REFUSALS[reason.name] ?? 'The browser refused to start the camera.'}</p>
    <p class="detail">{reason.name}: {reason.detail}</p>
  {/if}
</section>

<style>
  .blocked {
    position: fixed;
    inset: 0;
    z-index: 50;
    overflow-y: auto;
    background: var(--bg);
    padding: calc(18px + env(safe-area-inset-top)) calc(16px + env(safe-area-inset-right))
      calc(28px + env(safe-area-inset-bottom)) calc(16px + env(safe-area-inset-left));
  }

  .blocked h2 {
    margin: 0;
    font-weight: 600;
    color: var(--warn);
    font-size: 16px;
  }

  .blocked p {
    margin: 8px 0;
    font-size: 14px;
  }

  .blocked code {
    background: #14171c;
    padding: 2px 6px;
    border-radius: 5px;
    color: var(--accent-soft);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .blocked ol {
    margin: 8px 0;
    padding-left: 22px;
  }

  .blocked li {
    margin: 7px 0;
    font-size: 14px;
  }

  .detail {
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
</style>
