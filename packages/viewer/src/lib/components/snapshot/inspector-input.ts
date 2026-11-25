import type { BoardState } from '#lib/components/snapshot/board-state.svelte.ts';
import { classOfKey } from '#lib/components/snapshot/reading.ts';

const ARROWS: Readonly<Record<string, readonly [number, number]>> = {
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
  ArrowUp: [-1, 0],
  ArrowDown: [1, 0]
};

/**
 * Document-level input for the square inspector; returns the cleanup. Call it from an
 * `$effect`, because it touches `document`.
 *
 * Clicking outside the board and the inspector closes it. The check uses the event's
 * dispatch-time `composedPath()`, never a live `element.contains(event.target)`: an inner
 * handler may replace the clicked node, and a detached target makes `contains()` answer
 * false, which used to close the panel on every direct pick. Containers that must not
 * close it mark themselves with `data-keeps-inspector`.
 *
 * The keyboard is the diagnostic fast path on a desktop: arrows walk the board, Escape
 * closes, and a FEN letter (case carries colour) assigns that class to the square.
 */
export function listenInspector(board: BoardState): () => void {
  const onDocumentClick = (event: MouseEvent): void => {
    if (board.selected === null) return;
    const kept = event
      .composedPath()
      .some((node) => node instanceof HTMLElement && node.dataset['keepsInspector'] !== undefined);
    if (!kept) board.deselect();
  };

  const onKeyDown = (event: KeyboardEvent): void => {
    const at = board.selected;
    if (at === null || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target instanceof HTMLInputElement) return;
    if (event.key === 'Escape') {
      board.deselect();
      return;
    }
    const step = ARROWS[event.key];
    if (step !== undefined) {
      event.preventDefault();
      board.selectSquare(at.r + step[0], at.c + step[1]);
      return;
    }
    const cls = classOfKey(event.key);
    if (cls !== null) {
      event.preventDefault();
      board.assign(cls);
    }
  };

  document.addEventListener('click', onDocumentClick);
  document.addEventListener('keydown', onKeyDown);
  return () => {
    document.removeEventListener('click', onDocumentClick);
    document.removeEventListener('keydown', onKeyDown);
  };
}
