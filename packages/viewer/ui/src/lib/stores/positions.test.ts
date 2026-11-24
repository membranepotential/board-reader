import { describe, expect, it, vi } from 'vitest';
import { PositionStream } from './positions.svelte';
import { START_PLACEMENT } from '../protocol';
import { FLASH_MS, STALE_MS } from './liveness';
import { MOTION_STALE_MS } from './positions-motion';

function event(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    board_id: 'phone',
    seq: 1,
    utc: '2026-08-07T10:00:00.000Z',
    placement: START_PLACEMENT,
    confidence: 0.9,
    stability: 0.95,
    clock: { running: 'none' },
    status: 'ok',
    ...over
  };
}

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
    stream.ingest(event({ board_id: 'b' }));
    stream.ingest(event({ board_id: 'a' }));
    expect(stream.boards.map((b) => b.id)).toEqual(['a', 'b']);
  });

  it('drops a malformed frame instead of rendering a broken board', () => {
    const stream = streamAt({ now: 0 });
    expect(stream.ingest('{"nope": true}')).toBeNull();
    expect(stream.boards).toHaveLength(0);
  });

  it('sanitizes the placement once, at ingest', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(event({ placement: 'rxbqkbnr/8/8/8/8/8/8/8' }));
    expect(stream.board('phone')?.grid[0]).toEqual(['r', '', 'b', 'q', 'k', 'b', 'n', 'r']);
  });

  it('marks only the squares that changed, diffed against the previous grid', () => {
    const clock = { now: 1000 };
    const stream = streamAt(clock);
    stream.ingest(event());
    expect(Object.keys(stream.board('phone')!.changedAt)).toEqual([]);
    clock.now = 1100;
    stream.ingest(event({ seq: 2, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    expect(Object.keys(stream.board('phone')!.changedAt).sort()).toEqual(['e2', 'e4']);
  });

  it('keeps no history beyond the previous grid', () => {
    // The store's whole memory is one grid per board (DESIGN D1) — a third event
    // must not resurrect the first one's squares.
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(event());
    clock.now = 100;
    stream.ingest(event({ seq: 2, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    clock.now = 200;
    stream.ingest(event({ seq: 3, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    // e2/e4 are still fading (they are inside FLASH_MS) but nothing NEW changed.
    const board = stream.board('phone')!;
    expect(Object.keys(board.changedAt).sort()).toEqual(['e2', 'e4']);
    expect(board.changedAt['e4']).toBe(100);
  });

  it('forgets a ring once it has finished fading', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(event());
    clock.now = 100;
    stream.ingest(event({ seq: 2, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    clock.now = 100 + FLASH_MS + 1;
    stream.ingest(event({ seq: 3, placement: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' }));
    expect(Object.keys(stream.board('phone')!.changedAt)).toEqual([]);
  });

  it('reads motion from its own messages: none until one arrives, then moving or still', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    stream.ingest(event({ stability: 0.1 })); // the READ frame's stability is not motion
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
    stream.ingest(event());
    stream.ingestMotion(motion({ moving: true }));
    clock.now = MOTION_STALE_MS + 1;
    stream.ingest(event({ seq: 2 })); // reads may keep coming; motion is judged on its own
    expect(stream.liveness(stream.board('phone')!).motion).toEqual({ kind: 'stale' });
  });

  it('keeps the two message kinds apart', () => {
    const stream = streamAt({ now: 0 });
    expect(stream.ingest(motion())).toBeNull(); // a motion message is not a board
    expect(stream.boards).toHaveLength(0);
    expect(stream.ingestMotion(event())).toBeNull(); // nor is a position event motion
  });

  it('reads a 1 fps keepalive as 1.0 ev/s and goes stale when it stops', () => {
    const clock = { now: 0 };
    const stream = streamAt(clock);
    for (let i = 0; i <= 3; i++) {
      clock.now = i * 1000;
      stream.ingest(event({ seq: i }));
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
    socket.onmessage?.({ data: JSON.stringify(event()) } as MessageEvent);
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
    stream.ingest(event());
    const board = stream.board('phone')!;
    expect(stream.liveness(board).stale).toBe(false);
    // Nothing arrives for longer than STALE_MS; the store's own clock is what says so.
    clock.now = 1000 + STALE_MS + 1;
    stream.ingest(event({ board_id: 'other' })); // advances the shared clock
    expect(stream.liveness(board).stale).toBe(true);
  });
});

describe('prior repair on the stream', () => {
  const TWO_KINGS = 'rnbkkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
  const REPAIRED = 'rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
  const prior = {
    placement: REPAIRED,
    flips: [{ square: 'e8', from: 'k', to: 'q', cost: 0.74, constraint: 'kings' }],
    residual: {}
  };

  it('draws the repaired board, never the impossible raw one', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(event({ placement: TWO_KINGS, prior }));
    const board = stream.board('phone')!;
    expect(board.grid[0]).toEqual(['r', 'n', 'b', 'k', 'q', 'b', 'n', 'r']);
    expect(board.event.placement).toBe(TWO_KINGS); // the raw read stays readable
    expect(board.flipped).toEqual(['e8']);
  });

  it('diffs change rings against the repaired grid, so a stable repair is quiet', () => {
    const clock = { now: 1000 };
    const stream = streamAt(clock);
    stream.ingest(event({ placement: TWO_KINGS, prior }));
    clock.now = 2000;
    stream.ingest(event({ seq: 2, placement: TWO_KINGS, prior }));
    expect(Object.keys(stream.board('phone')!.changedAt)).toEqual([]);
  });

  it('ignores flip squares that are not squares', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(
      event({
        placement: TWO_KINGS,
        prior: { ...prior, flips: [...prior.flips, { square: 'z9', from: 'k', to: 'q', cost: 1, constraint: 'kings' }] }
      })
    );
    expect(stream.board('phone')!.flipped).toEqual(['e8']);
  });

  it('an event without a prior draws its own placement and flips nothing', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(event());
    expect(stream.board('phone')!.flipped).toEqual([]);
  });
});

describe('settled board on the stream', () => {
  const SETTLED = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR';
  const settled = { placement: SETTLED, pending: ['e2', 'e4'] };

  it('draws the settled board ahead of the raw one, and exposes its pending squares', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(event({ placement: START_PLACEMENT, settled }));
    const board = stream.board('phone')!;
    expect(board.grid[3]).toEqual(['', '', '', '', 'p', '', '', '']); // rank 5, e5
    expect(board.event.placement).toBe(START_PLACEMENT); // the raw read stays readable
    expect(board.pending).toEqual(['e2', 'e4']);
  });

  it('prefers settled over prior when an event carries both', () => {
    const stream = streamAt({ now: 0 });
    const prior = { placement: START_PLACEMENT, flips: [], residual: {} };
    stream.ingest(event({ placement: START_PLACEMENT, prior, settled }));
    expect(stream.board('phone')!.grid[3]).toEqual(['', '', '', '', 'p', '', '', '']);
  });

  it('diffs change rings against the settled grid, so a quiet debounce is quiet', () => {
    const clock = { now: 1000 };
    const stream = streamAt(clock);
    stream.ingest(event({ placement: START_PLACEMENT, settled }));
    clock.now = 2000;
    stream.ingest(event({ seq: 2, placement: START_PLACEMENT, settled }));
    expect(Object.keys(stream.board('phone')!.changedAt)).toEqual([]);
  });

  it('ignores pending squares that are not squares', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(
      event({ settled: { placement: START_PLACEMENT, pending: ['e4', 'z9'] } })
    );
    expect(stream.board('phone')!.pending).toEqual(['e4']);
  });

  it('an event without a settled board draws its prior (or raw) placement and exposes no pending squares', () => {
    const stream = streamAt({ now: 0 });
    stream.ingest(event());
    expect(stream.board('phone')!.pending).toEqual([]);
  });
});
