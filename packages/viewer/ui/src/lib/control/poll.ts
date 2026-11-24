/**
 * Run `tick` every `ms` while the tab is visible, and not at all while it is hidden.
 *
 * Each tick is awaited before the next is scheduled, so a slow node never has two
 * polls in flight. Becoming visible again polls at once rather than waiting out a
 * period, so a tab brought back shows fresh state immediately.
 *
 * `doc` is the document's visibility, passed in so tests can drive it.
 */

export interface Visibility extends EventTarget {
  readonly visibilityState: DocumentVisibilityState;
}

export function pollWhileVisible(tick: () => Promise<void>, ms: number, doc: Visibility): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let running = false;
  let stopped = false;

  const schedule = (): void => {
    clearTimeout(timer);
    timer = undefined;
    if (!stopped && doc.visibilityState === 'visible') timer = setTimeout(run, ms);
  };

  const run = async (): Promise<void> => {
    if (running || stopped || doc.visibilityState !== 'visible') return;
    running = true;
    try {
      await tick();
    } finally {
      running = false;
      schedule();
    }
  };

  const onVisibility = (): void => {
    if (doc.visibilityState === 'visible') void run();
    else schedule();
  };

  doc.addEventListener('visibilitychange', onVisibility);
  void run();
  return () => {
    stopped = true;
    clearTimeout(timer);
    doc.removeEventListener('visibilitychange', onVisibility);
  };
}
