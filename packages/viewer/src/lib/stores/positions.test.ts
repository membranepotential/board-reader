import { describe, expect, it, vi } from 'vitest';
import { PositionStream } from './positions.svelte';
import { START_PLACEMENT } from '../protocol';
import { positionEvent } from '../protocol/fixtures';
import { FLASH_MS, STALE_MS } from './liveness';
import { MOTION_STALE_MS } from './positions-motion';

function motion(over: Record<string, unknown> = {}): Record<string, unknown> {
  return { type: 'motion', board_id: 'phone', moving: false, stability: 1, ts: 1000, ...over };
}

/** A stream driven by an explicit clock, so nothing here waits on real time. */
function streamAt(clock: { now: number }): PositionStream {
  return new PositionStream({ clock: () => clock.now });
}

describe('PositionStream', () => {
  it('keys boards by board_id and keeps them in a stable order', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent({ board_id: 'b' }));
    stream.ingest(positionEvent({ board_id: 'a' }));
    expect(stream.boards.map((b) => b.id)).toEqual(['a', 'b']);
  });

  it('drops a malformed frame instead of rendering a broken board', () => {
    const stream = streamAt({ now: 0 });
    expect(stream.ingest('{"nope": true}')).toBeNull();
    expect(stream.boards).toHaveLength(0);
  });

  it('refuses a placement that is not a board (isPlacement), rather than drawing a guess', () => {
    const stream = streamAt({ now: 0 });
    expect(stream.ingest(positionEvent({ placement: 'rxbqkbnr/8/8/8/8/8/8/8' }))).toBeNull();
    expect(stream.boards).toHaveLength(0);
  });

  it('marks only the squares that changed, diffed against the previous grid', () => {
    const clock = { now: 1000 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent());
    expect(Object.keys(stream.board('phone')!.changedAt)).toEqual([]);
    clock.now = 1100;
    stream.ingest(positionEvent({ seq: 2, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    expect(Object.keys(stream.board('phone')!.changedAt).sort()).toEqual(['e2', 'e4']);
  });

  it('keeps no history beyond the previous grid', () => {
    // The store's whole memory is one grid per board (DESIGN D1) — a third event
    // must not resurrect the first one's squares.
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent());
    clock.now = 100;
    stream.ingest(positionEvent({ seq: 2, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    clock.now = 200;
    stream.ingest(positionEvent({ seq: 3, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    // e2/e4 are still fading (they are inside FLASH_MS) but nothing NEW changed.
    const board = stream.board('phone')!;
    expect(Object.keys(board.changedAt).sort()).toEqual(['e2', 'e4']);
    expect(board.changedAt['e4']).toBe(100);
  });

  it('forgets a ring once it has finished fading', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent());
    clock.now = 100;
    stream.ingest(positionEvent({ seq: 2, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    clock.now = 100 + FLASH_MS + 1;
    stream.ingest(positionEvent({ seq: 3, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    expect(Object.keys(stream.board('phone')!.changedAt)).toEqual([]);
  });

  it('reads motion from its own messages: none until one arrives, then moving or still', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent({ stability: 0.1 })); // the READ frame's stability is not motion
    const board = stream.board('phone')!;
    expect(stream.liveness(board).motion).toEqual({ kind: 'none' });

    stream.ingestMotion(motion({ moving: true, stability: 0.3 }));
    expect(stream.liveness(board).motion).toEqual({ kind: 'moving', stability: 0.3 });
    stream.ingestMotion(motion({ moving: false, stability: 0.98 }));
    expect(stream.liveness(board).motion).toEqual({ kind: 'still', stability: 0.98 });
  });

  it('a motion message that stops arriving goes stale instead of claiming "moving"', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent());
    stream.ingestMotion(motion({ moving: true }));
    clock.now = MOTION_STALE_MS + 1;
    stream.ingest(positionEvent({ seq: 2 })); // reads may keep coming; motion is judged on its own
    expect(stream.liveness(stream.board('phone')!).motion).toEqual({ kind: 'stale' });
  });

  it('keeps the two message kinds apart', () => {
    const stream = streamAt({ now: 0 });
    expect(stream.ingest(motion())).toBeNull(); // a motion message is not a board
    expect(stream.boards).toHaveLength(0);
    expect(stream.ingestMotion(positionEvent())).toBeNull(); // nor is a position event motion
  });

  it.each([
    ['a stability outside [0, 1]', { stability: 1.7 }],
    ['a moving flag that is not a boolean', { moving: 'yes' }],
    ['an empty board id', { board_id: '' }],
    ['no timestamp', { ts: null }]
  ])('refuses a motion update with %s, as the node would', (_label, over) => {
    const stream = streamAt({ now: 0 });
    stream.ingest(positionEvent());
    expect(stream.ingestMotion(motion(over))).toBeNull();
    expect(stream.liveness(stream.board('phone')!).motion).toEqual({ kind: 'none' });
  });

  it('reads a 1 fps keepalive as 1.0 ev/s and goes stale when it stops', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    for (let i = 0; i <= 3; i++) {
      clock.now = i * 1000;
      stream.ingest(positionEvent({ seq: i }));
    }
    const board = stream.board('phone')!;
    expect(stream.liveness(board).eventsPerSecond).toBeCloseTo(1.0, 9);
    expect(stream.liveness(board).stale).toBe(false);
  });

  it('connects, reports live, and reconnects with backoff after a close', () => {
    vi.useFakeTimers();
    const sockets: Array<Partial<WebSocket> & { fire: (name: 'open' | 'close') => void }> = [];
    const stream = new PositionStream({
      url: 'ws://test/ws/viewer',
      clock: () => 0,
      reconnectMinMs: 100,
      reconnectMaxMs: 400,
      socketFactory: () => {
        const socket = {
          onopen: null as ((e: Event) => void) | null,
          onclose: null as ((e: CloseEvent) => void) | null,
          onerror: null,
          onmessage: null,
          close: () => {},
          fire(name: 'open' | 'close') {
            if (name === 'open') this.onopen?.({} as Event);
            else this.onclose?.({} as CloseEvent);
          }
        };
        sockets.push(socket as unknown as (typeof sockets)[number]);
        return socket as unknown as WebSocket;
      }
    });
    stream.start();
    expect(stream.connection).toBe('connecting');
    sockets[0]!.fire('open');
    expect(stream.connection).toBe('live');

    sockets[0]!.fire('close');
    expect(stream.connection).toBe('disconnected');
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(2);
    sockets[1]!.fire('close');
    vi.advanceTimersByTime(199);
    expect(sockets).toHaveLength(2); // 200 ms this time, not 100
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(3);

    stream.stop();
    vi.advanceTimersByTime(10_000);
    expect(sockets).toHaveLength(3);
    vi.useRealTimers();
  });

  it('routes both message kinds off one socket', () => {
    let socket: { onmessage: ((m: MessageEvent) => void) | null } = { onmessage: null };
    const stream = new PositionStream({
      url: 'ws://test/ws/viewer',
      clock: () => 0,
      socketFactory: () => {
        socket = { onmessage: null };
        return Object.assign(socket, { close: () => {} }) as unknown as WebSocket;
      }
    });
    stream.start();
    socket.onmessage?.({ data: JSON.stringify(positionEvent()) } as MessageEvent);
    socket.onmessage?.({ data: JSON.stringify(motion({ moving: true })) } as MessageEvent);
    socket.onmessage?.({ data: 'not json' } as MessageEvent);
    const board = stream.board('phone')!;
    expect(stream.boards).toHaveLength(1);
    expect(stream.liveness(board).motion.kind).toBe('moving');
    stream.stop();
  });

  it('reports staleness against the ticker, not against event.utc', () => {
    const clock = { now: 1000 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent());
    const board = stream.board('phone')!;
    expect(stream.liveness(board).stale).toBe(false);
    // Nothing arrives for longer than STALE_MS; the store's own clock is what says so.
    clock.now = 1000 + STALE_MS + 1;
    stream.ingest(positionEvent({ board_id: 'other' })); // advances the shared clock
    expect(stream.liveness(board).stale).toBe(true);
  });
});

describe('the settled board on the stream', () => {
  const SETTLED = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR';

  it('draws the event placement, the settled board, and exposes its pending squares', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(positionEvent({ placement: SETTLED, pending: ['e2', 'e4'] }));
    const board = stream.board('phone')!;
    expect(board.grid[3]).toEqual(['', '', '', '', 'p', '', '', '']); // rank 5, e5
    expect(board.pending).toEqual(['e2', 'e4']);
  });

  it('diffs change rings against the settled grid, so a quiet debounce is quiet', () => {
    const clock = { now: 1000 };
    const stream = streamAt(clock);
    stream.ingest(positionEvent({ placement: SETTLED, pending: ['e2', 'e4'] }));
    clock.now = 2000;
    stream.ingest(positionEvent({ seq: 2, placement: SETTLED, pending: ['e2', 'e4'] }));
    expect(Object.keys(stream.board('phone')!.changedAt)).toEqual([]);
  });

  it('refuses an event whose pending squares are not squares', () => {
    const stream = streamAt({ now: 0 });
    expect(stream.ingest(positionEvent({ pending: ['e4', 'z9'] }))).toBeNull();
    expect(stream.boards).toHaveLength(0);
  });

  it('an event whose read agrees with the settled board has no pending squares', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(positionEvent());
    expect(stream.board('phone')!.pending).toEqual([]);
  });
});
