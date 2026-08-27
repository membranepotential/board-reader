import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HEARTBEAT_MS, Heartbeat, type HeartbeatCounters } from './heartbeat';

describe('Heartbeat', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  function harness(initial: Partial<HeartbeatCounters> = {}) {
    const notes: Array<{ kind: string; detail: Record<string, unknown> | undefined }> = [];
    let clock = 0;
    let counters: HeartbeatCounters = {
      sent: 0,
      dropped: 0,
      acked: null,
      connected: true,
      behind: false,
      ...initial
    };
    const beat = new Heartbeat(
      { note: (kind, detail) => notes.push({ kind, detail }) },
      () => counters,
      { now: () => clock }
    );
    const advance = (ms: number, next: Partial<HeartbeatCounters> = {}) => {
      clock += ms;
      counters = { ...counters, ...next };
      vi.advanceTimersByTime(ms);
    };
    return { notes, beat, advance };
  }

  it('reports the rates between beats, not just the totals', () => {
    const { notes, beat, advance } = harness();
    beat.start();
    expect(notes).toHaveLength(0); // the first beat is an interval away, not immediate

    advance(HEARTBEAT_MS, { sent: 75, dropped: 5, acked: 70 });

    expect(notes).toHaveLength(1);
    expect(notes[0]?.kind).toBe('heartbeat');
    expect(notes[0]?.detail).toMatchObject({
      sent: 75,
      dropped: 5,
      acked: 70,
      sentPerSec: 7.5,
      droppedPerSec: 0.5,
      unacked: 4,
      since: 10,
      connected: true,
      behind: false
    });
  });

  it('rates are per-interval, so a phone that stops sending says so', () => {
    const { notes, beat, advance } = harness();
    beat.start();
    advance(HEARTBEAT_MS, { sent: 75 });
    advance(HEARTBEAT_MS); // nothing sent in the second window

    expect(notes[1]?.detail).toMatchObject({ sent: 75, sentPerSec: 0 });
  });

  it('beats once more on stop, so the last stretch is on the record', () => {
    const { notes, beat, advance } = harness();
    beat.start();
    advance(HEARTBEAT_MS, { sent: 75 });
    advance(HEARTBEAT_MS / 2, { sent: 110 });
    beat.stop();

    expect(notes).toHaveLength(2);
    expect(notes[1]?.detail).toMatchObject({ sent: 110, since: 5, sentPerSec: 7 });
    expect(beat.running).toBe(false);

    advance(HEARTBEAT_MS * 2); // stopped means stopped
    expect(notes).toHaveLength(2);
  });

  it('counts every sent frame as unacked until the first ack', () => {
    const { notes, beat, advance } = harness();
    beat.start();
    advance(HEARTBEAT_MS, { sent: 12, acked: null });

    expect(notes[0]?.detail).toMatchObject({ unacked: 12 });
  });

  it('starting twice keeps one timer, not two', () => {
    const { notes, beat, advance } = harness();
    beat.start();
    beat.start();
    advance(HEARTBEAT_MS, { sent: 10 });

    expect(notes).toHaveLength(1);
  });

  it('stopping without starting is a no-op, not a stray note', () => {
    const { notes, beat } = harness();
    beat.stop();
    expect(notes).toHaveLength(0);
  });
});
