import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { pollWhileVisible, type Visibility } from './poll';

class FakeDocument extends EventTarget implements Visibility {
  visibilityState: DocumentVisibilityState = 'visible';

  set(state: DocumentVisibilityState): void {
    this.visibilityState = state;
    this.dispatchEvent(new Event('visibilitychange'));
  }
}

describe('pollWhileVisible', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('polls at once, then every period while visible', async () => {
    const tick = vi.fn(async () => {});
    const stop = pollWhileVisible(tick, 1000, new FakeDocument());
    expect(tick).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(3000);
    expect(tick).toHaveBeenCalledTimes(4);
    stop();
    await vi.advanceTimersByTimeAsync(3000);
    expect(tick).toHaveBeenCalledTimes(4);
  });

  it('stops while hidden and polls immediately when shown again', async () => {
    const doc = new FakeDocument();
    const tick = vi.fn(async () => {});
    const stop = pollWhileVisible(tick, 1000, doc);
    await vi.advanceTimersByTimeAsync(0);
    doc.set('hidden');
    await vi.advanceTimersByTimeAsync(10_000);
    expect(tick).toHaveBeenCalledTimes(1);
    doc.set('visible');
    expect(tick).toHaveBeenCalledTimes(2);
    stop();
  });

  it('never has two polls in flight against a slow node', async () => {
    let inFlight = 0;
    let most = 0;
    const tick = vi.fn(async () => {
      inFlight++;
      most = Math.max(most, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 2500));
      inFlight--;
    });
    const doc = new FakeDocument();
    const stop = pollWhileVisible(tick, 1000, doc);
    doc.set('visible'); // a visibility event mid-poll must not start a second one
    await vi.advanceTimersByTimeAsync(10_000);
    expect(most).toBe(1);
    // One 2.5 s poll plus a 1 s wait per cycle: 3 cycles start in 10 s.
    expect(tick).toHaveBeenCalledTimes(3);
    stop();
  });
});
