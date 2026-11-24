/**
 * Client-side trial log — the phone's half of the story a live trial leaves behind.
 *
 * A failed phone session used to be undiagnosable from the dev machine: the node
 * logs what *arrived*, but "the camera froze", "the socket never connected" and
 * "every frame was refused" all look identical from there — silence. This logger
 * notes the page's own lifecycle (socket status, camera track events, stall
 * recovery, detect outcomes), batches the notes, and posts them to the
 * viewer's `/api/client-log` sink, which appends them beside the server's own
 * records. Same-origin and relative like every other request the page makes.
 *
 * Delivery is best-effort by design: the log must never compete with the frame
 * stream for the uplink (entries are a few hundred bytes on a multi-second
 * timer), and a lost batch is a lost batch — `sendBeacon` on `pagehide` is the
 * one concession, because the most interesting entries are the ones just before
 * the page dies. Every note is mirrored to the console for desktop debugging.
 */

export interface TrialNote {
  /** Epoch ms, stamped at `note()` time. */
  readonly t: number;
  readonly kind: string;
  readonly detail?: Record<string, unknown>;
}

export const FLUSH_MS = 3000;
/** Oldest notes are dropped past this — a runaway loop must not hoard memory. */
export const MAX_BUFFERED_NOTES = 500;

export interface TrialLogOptions {
  readonly url?: string;
  readonly flushMs?: number;
  /** Injected in tests; `fetch` with `keepalive` in a browser. */
  readonly postFn?: (url: string, body: string) => void;
  readonly now?: () => number;
}

export class TrialLog {
  private buffer: TrialNote[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;
  private readonly url: string;
  private readonly flushMs: number;
  private readonly post: (url: string, body: string) => void;
  private readonly now: () => number;

  constructor(options: TrialLogOptions = {}) {
    this.url = options.url ?? '/api/client-log';
    this.flushMs = options.flushMs ?? FLUSH_MS;
    this.now = options.now ?? (() => Date.now());
    this.post =
      options.postFn ??
      ((url, body) => {
        void fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
          keepalive: true
        }).catch(() => {
          /* best-effort: a lost batch must never surface as a page error */
        });
      });
  }

  note(kind: string, detail?: Record<string, unknown>): void {
    const entry: TrialNote = { t: this.now(), kind, ...(detail === undefined ? {} : { detail }) };
    this.buffer.push(entry);
    if (this.buffer.length > MAX_BUFFERED_NOTES) this.buffer.shift();
    // eslint-disable-next-line no-console
    console.log(`[trial] ${kind}`, detail ?? '');
  }

  /** How many notes await the next flush — for tests and the settings sheet. */
  get pending(): number {
    return this.buffer.length;
  }

  start(): void {
    if (this.timer !== null) return;
    this.timer = setInterval(() => this.flush(), this.flushMs);
  }

  stop(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    this.flush();
  }

  flush(): void {
    if (this.buffer.length === 0) return;
    const body = JSON.stringify({ notes: this.buffer });
    this.buffer = [];
    this.post(this.url, body);
  }

  /**
   * `pagehide` delivery: a normal fetch started here is cancelled with the page,
   * so this one batch rides `sendBeacon`, which outlives it.
   */
  flushBeacon(): void {
    if (this.buffer.length === 0) return;
    const body = JSON.stringify({ notes: this.buffer });
    this.buffer = [];
    if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
      navigator.sendBeacon(this.url, new Blob([body], { type: 'application/json' }));
    } else {
      this.post(this.url, body);
    }
  }
}
