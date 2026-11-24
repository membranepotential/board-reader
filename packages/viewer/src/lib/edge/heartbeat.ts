/**
 * Streaming heartbeat — the phone's own account of what it was doing.
 *
 * The trial log is event-driven: it notes what *happens* (socket up, track muted,
 * stall recovered). The 08-13 trial showed the gap that leaves — 14 notes for 748
 * frames, and no way to tell afterwards whether a quiet stretch was a phone
 * sending happily or a phone sending nothing. A heartbeat closes it: one note
 * every {@link HEARTBEAT_MS}, carrying the counters and the rates between beats.
 *
 * It samples rather than counts, so nothing in the capture path has to report to
 * it, and it stays as best-effort as the log it writes into: a missed beat is a
 * missed beat, never an error the operator sees mid-trial.
 */

export interface HeartbeatCounters {
  /** Cumulative frames handed to the socket. */
  readonly sent: number;
  /** Cumulative frames the page decided not to send (behind, unchanged, refused). */
  readonly dropped: number;
  /** Highest sequence the gateway has acked, or `null` before the first ack. */
  readonly acked: number | null;
  readonly connected: boolean;
  /** True while the socket's buffer says the uplink is not keeping up. */
  readonly behind: boolean;
}

/** What one beat writes: the totals, plus what changed since the last beat. */
export interface HeartbeatNote extends Record<string, unknown> {
  readonly sent: number;
  readonly dropped: number;
  readonly acked: number | null;
  readonly connected: boolean;
  readonly behind: boolean;
  /** Frames sent per second since the previous beat. */
  readonly sentPerSec: number;
  /** Frames dropped per second since the previous beat. */
  readonly droppedPerSec: number;
  /** Sequences sent but not yet acked — the uplink's backlog. */
  readonly unacked: number;
  /** Seconds since the previous beat; the window every rate here is over. */
  readonly since: number;
}

export const HEARTBEAT_MS = 10_000;

export interface HeartbeatOptions {
  readonly intervalMs?: number;
  /** Monotonic-ish clock in ms. Injected in tests. */
  readonly now?: () => number;
}

/** Somewhere to write a beat — {@link TrialLog} satisfies this. */
export interface NoteSink {
  note(kind: string, detail?: Record<string, unknown>): void;
}

export class Heartbeat {
  private timer: ReturnType<typeof setInterval> | null = null;
  private last: HeartbeatCounters | null = null;
  private lastAt = 0;
  private readonly intervalMs: number;
  private readonly now: () => number;

  constructor(
    private readonly sink: NoteSink,
    private readonly sample: () => HeartbeatCounters,
    options: HeartbeatOptions = {}
  ) {
    this.intervalMs = options.intervalMs ?? HEARTBEAT_MS;
    this.now = options.now ?? (() => Date.now());
  }

  /** Begin beating. The first beat lands one interval from now, not immediately. */
  start(): void {
    if (this.timer !== null) return;
    this.last = this.sample();
    this.lastAt = this.now();
    this.timer = setInterval(() => this.beat(), this.intervalMs);
  }

  /** Stop beating, with one last beat so the final stretch is on the record. */
  stop(): void {
    if (this.timer === null) return;
    clearInterval(this.timer);
    this.timer = null;
    this.beat();
    this.last = null;
  }

  get running(): boolean {
    return this.timer !== null;
  }

  /** One beat. Public so the page can force one at a moment that matters. */
  beat(): void {
    const counters = this.sample();
    const at = this.now();
    const since = this.last === null ? 0 : Math.max(0, (at - this.lastAt) / 1000);
    const rate = (delta: number): number =>
      since > 0 ? Math.round((delta / since) * 100) / 100 : 0;
    const note: HeartbeatNote = {
      sent: counters.sent,
      dropped: counters.dropped,
      acked: counters.acked,
      connected: counters.connected,
      behind: counters.behind,
      sentPerSec: rate(counters.sent - (this.last?.sent ?? counters.sent)),
      droppedPerSec: rate(counters.dropped - (this.last?.dropped ?? counters.dropped)),
      unacked: counters.acked === null ? counters.sent : Math.max(0, counters.sent - counters.acked - 1),
      since: Math.round(since * 100) / 100
    };
    this.last = counters;
    this.lastAt = at;
    this.sink.note('heartbeat', note);
  }
}
