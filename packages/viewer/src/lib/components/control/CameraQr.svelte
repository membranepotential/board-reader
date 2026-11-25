<script lang="ts">
  /**
   * A QR code that opens `/camera` on a phone: `<CameraQr />`, no props.
   *
   * It asks the viewer for the addresses a phone can open (`GET /api/camera-links`,
   * best first) once on mount, shows the first as a QR code and as text with a copy
   * button, and lets the operator switch to another one. The links carry no
   * `?board=`: the node assigns the phone its board.
   */
  import { onMount } from 'svelte';
  import { encode } from 'uqr';
  import { cameraLinksSchema, type CameraLink, type LinkSource } from './camera-links.ts';

  const ROUTE = '/api/camera-links';
  const TIMEOUT_MS = 5_000;

  /** How each kind of link reads in the switcher. */
  const VIA_LABEL: Readonly<Record<LinkSource, string>> = {
    public: 'public URL',
    lan: 'local network',
    hostname: 'hostname',
    loopback: 'this machine only'
  };

  type Links =
    | { readonly kind: 'loading' }
    | { readonly kind: 'failed'; readonly message: string }
    | { readonly kind: 'ready'; readonly links: readonly CameraLink[] };

  let answer = $state.raw<Links>({ kind: 'loading' });
  let chosen = $state(0);
  let copied = $state<'copied' | 'failed' | null>(null);

  const links = $derived(answer.kind === 'ready' ? answer.links : []);
  const current = $derived<CameraLink | undefined>(links[chosen]);
  const onlyLoopback = $derived(links.length > 0 && links.every((l) => l.via === 'loopback'));
  /** The dark modules as one SVG path, one unit per module (uqr adds the quiet zone). */
  const qr = $derived.by(() => {
    if (current === undefined) return undefined;
    const { size, data } = encode(current.url, { ecc: 'M', border: 4 });
    const d = data
      .flatMap((row, y) => row.map((dark, x) => (dark ? `M${x},${y}h1v1h-1z` : '')))
      .join('');
    return { size, d };
  });

  async function load(): Promise<Links> {
    try {
      const response = await fetch(ROUTE, { signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (!response.ok) return { kind: 'failed', message: `${response.status} ${response.statusText}` };
      const parsed = cameraLinksSchema.safeParse(await response.json());
      return parsed.success
        ? { kind: 'ready', links: parsed.data.links }
        : { kind: 'failed', message: 'unreadable answer from the viewer' };
    } catch (cause) {
      return { kind: 'failed', message: String(cause) };
    }
  }

  onMount(() => {
    void load().then((loaded) => (answer = loaded));
  });

  function choose(index: number): void {
    chosen = index;
    copied = null;
  }

  async function copy(url: string): Promise<void> {
    try {
      // `navigator.clipboard` is missing outside a secure context (plain http on the LAN).
      await navigator.clipboard.writeText(url);
      copied = 'copied';
    } catch {
      copied = 'failed';
    }
  }
</script>

<section class="camera-qr" aria-label="Open the camera on a phone">
  {#if answer.kind === 'loading'}
    <p class="hint">Looking up this machine's addresses…</p>
  {:else if answer.kind === 'failed'}
    <p class="warn">Could not get the camera addresses: {answer.message}</p>
  {:else if current === undefined || qr === undefined}
    <p class="warn">
      No address a phone can open: the viewer serves no https port. Start it with
      <code>--tls-port</code> (<code>make dev</code> does), or set <code>BR_PUBLIC_URL</code> to a
      tunnel.
    </p>
  {:else}
    <svg
      class="qr"
      viewBox="0 0 {qr.size} {qr.size}"
      shape-rendering="crispEdges"
      role="img"
      aria-label="QR code for {current.url}"
    >
      <rect width={qr.size} height={qr.size} fill="#fff" />
      <path d={qr.d} fill="#000" />
    </svg>
    <div class="url">
      <code data-testid="camera-url">{current.url}</code>
      <button type="button" onclick={() => copy(current.url)}>Copy</button>
      {#if copied === 'copied'}
        <span class="hint" role="status">copied</span>
      {:else if copied === 'failed'}
        <span class="warn" role="status">copy failed: select the address instead</span>
      {/if}
    </div>
    {#if links.length > 1}
      <fieldset>
        <legend>address</legend>
        {#each links as link, index (link.url)}
          <label>
            <input type="radio" name="camera-link" checked={index === chosen} onchange={() => choose(index)} />
            {link.url} <span class="via">({VIA_LABEL[link.via]})</span>
          </label>
        {/each}
      </fieldset>
    {/if}
    {#if onlyLoopback}
      <p class="warn">
        Only this machine's own address is known, which a phone cannot open. Connect this machine to
        the phone's network, or set <code>BR_PUBLIC_URL</code> to a tunnel.
      </p>
    {:else if current.via !== 'public'}
      <p class="hint">The phone must be on the same network. Accept the certificate warning once.</p>
    {/if}
  {/if}
</section>

<style>
  .camera-qr {
    font-size: 13px;
    color: #e6e6e6;
  }
  .qr {
    display: block;
    width: 220px;
    max-width: 100%;
    height: auto;
    border-radius: 6px;
  }
  .url {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
  }
  code {
    overflow-wrap: anywhere;
    user-select: all;
  }
  button {
    font: inherit;
    font-size: 13px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #2f3540;
    background: #232833;
    color: #e6e6e6;
    cursor: pointer;
  }
  fieldset {
    border: 1px solid #2f3540;
    border-radius: 6px;
    margin: 10px 0 0;
    padding: 6px 10px;
  }
  legend {
    color: #9aa3ad;
  }
  label {
    display: block;
    padding: 2px 0;
    overflow-wrap: anywhere;
  }
  .via,
  .hint {
    color: #9aa3ad;
  }
  .warn {
    color: #ffcf33;
  }
  p {
    margin: 8px 0 0;
  }
</style>
