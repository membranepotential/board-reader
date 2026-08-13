import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FLUSH_MS, MAX_BUFFERED_NOTES, TrialLog } from './triallog';

describe('TrialLog', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(console, 'log').mockImplementation(() => {});
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  function harness() {
    const posts: Array<{ url: string; body: unknown }> = [];
    const log = new TrialLog({
      postFn: (url, body) => posts.push({ url, body: JSON.parse(body) }),
      now: () => 42
    });
    return { posts, log };
  }

  it('batches notes and posts them on the flush timer, stamped at note time', () => {
    const { posts, log } = harness();
    log.start();
    log.note('camera-stall-reopen', { attempt: 1 });
    log.note('pagehide');
    expect(posts).toHaveLength(0); // nothing sent per note — never competes with frames

    vi.advanceTimersByTime(FLUSH_MS);
    expect(posts).toHaveLength(1);
    expect(posts[0]?.url).toBe('/api/client-log');
    expect(posts[0]?.body).toEqual({
      notes: [
        { t: 42, kind: 'camera-stall-reopen', detail: { attempt: 1 } },
        { t: 42, kind: 'pagehide' }
      ]
    });

    // an empty interval posts nothing at all
    vi.advanceTimersByTime(FLUSH_MS);
    expect(posts).toHaveLength(1);
    log.stop();
  });

  it('stop() delivers whatever is still buffered', () => {
    const { posts, log } = harness();
    log.start();
    log.note('streaming-on');
    log.stop();
    expect(posts).toHaveLength(1);
  });

  it('drops the oldest notes past the buffer cap', () => {
    const { posts, log } = harness();
    for (let i = 0; i < MAX_BUFFERED_NOTES + 10; i++) log.note('tick', { i });
    expect(log.pending).toBe(MAX_BUFFERED_NOTES);
    log.flush();
    const notes = (posts[0]?.body as { notes: Array<{ detail: { i: number } }> }).notes;
    expect(notes[0]?.detail.i).toBe(10); // the oldest ten fell off
  });

  it('flushBeacon falls back to the post function when sendBeacon is absent', () => {
    const { posts, log } = harness();
    log.note('pagehide');
    log.flushBeacon(); // node test env: no navigator.sendBeacon
    expect(posts).toHaveLength(1);
  });
});
