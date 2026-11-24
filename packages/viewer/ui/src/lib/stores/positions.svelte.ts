/**
 * The live position stream, keyed by board — `/ws/viewer` as Svelte 5 state.
 *
 * **This is not a state machine over positions.** DESIGN D1 says the CV layer is a
 * stateless per-frame estimator: it streams positions, never moves, legality or
 * game state. The UI must not quietly become the thing D1 refuses to build, so the
 * only history kept is the PREVIOUS grid — and only because a changed square has
 * to be diffed against something to be drawn. There is no move list, no history
 * buffer, no inference from one event to the next; a board's whole state is its
 * latest event plus how fresh that event is.
 *
 * Reconnects with exponential backoff, because the viewer relay closes the socket
 * with the node's own reason when the node is the thing that is down, and a page
 * left at the board must come back by itself.
 */

import { VIEWER_SOCKET, wsUrl } from '../api';
import { changedSquares, fenToGrid, type BoardGrid } from '../board/fen';
import { isSquareKey, parseWire, type SquareKey } from '../protocol';
import { MotionUpdate, PositionEvent, ViewerMessage } from '../protocol/generated';
import {
  changeRings,
  eventsPerSecond,
  isStale,
  pruneChanges,
  withinWindow,
  type ChangeRing
} from './liveness';
import { motionReadout, type MotionReadout, type MotionSample } from './positions-motion';

export type ConnectionState = 'connecting' | 'live' | 'disconnected';

/**
 * The squares the prior decoder relabelled. The validator already refused any
 * event whose square is not one; the guard only narrows `string` to `SquareKey`.
 */
function flippedSquares(event: PositionEvent): SquareKey[] {
  return (event.prior?.flips ?? []).map((flip) => flip.square).filter(isSquareKey);
}

/** The settled board's pending squares, narrowed the same way. */
function pendingSquares(event: PositionEvent): SquareKey[] {
  return (event.settled?.pending ?? []).filter(isSquareKey);
}

/** Everything one card knows about its board. Replaced wholesale on every event. */
export interface BoardState {
  readonly id: string;
  readonly event: PositionEvent;
  /**
   * The sanitized 8×8 — what the board is drawn from, never the raw string.
   * `settled.placement` when the event carries one (a debounced board, held
   * still against flicker); else the node's prior-repaired placement, since the
   * raw argmax can spell an impossible board (two kings) that must never be what
   * the operator stares at; else the raw placement itself.
   */
  readonly grid: BoardGrid;
  /** Squares the prior decoder relabelled to make the board possible. */
  readonly flipped: readonly SquareKey[];
  /** Squares the settled board hasn't adopted yet — a move in flight, or a
   *  misread still being debounced. Empty when the event carries no `settled`. */
  readonly pending: readonly SquareKey[];
  /** Square → when it last changed, for the fading ring. The only history kept. */
  readonly changedAt: Readonly<Partial<Record<SquareKey, number>>>;
  /** Arrival, not `event.utc`: this is stream freshness, not sensor time. */
  readonly lastAt: number;
  /** Arrival stamps still inside the rate window. */
  readonly stamps: readonly number[];
}

/** The derived readout a card shows, recomputed against the shared clock. */
export interface BoardLiveness {
  readonly ageMs: number;
  readonly stale: boolean;
  readonly eventsPerSecond: number;
  readonly rings: readonly ChangeRing[];
  /** Is the camera seeing motion right now — from the per-frame motion messages. */
  readonly motion: MotionReadout;
}

export interface PositionStreamOptions {
  /** Relative by default and derived from the page — nothing here may name an origin. */
  readonly url?: string;
  readonly socketFactory?: (url: string) => WebSocket;
  readonly reconnectMinMs?: number;
  readonly reconnectMaxMs?: number;
  /** Injected in tests; `performance.now()` in a browser. */
  readonly clock?: () => number;
}

/** Backoff bounds: quick enough to feel instant on a blip, slow enough that a node
 *  that is down for a game is not hammered once a second for an hour. */
export const RECONNECT_MIN_MS = 1500;
export const RECONNECT_MAX_MS = 8000;

/**
 * One `/ws/viewer` subscription and the boards it has seen.
 *
 * `start()`/`stop()` rather than connecting in the constructor: every route here is
 * prerendered, so module scope runs where there is no socket and no clock.
 */
export class PositionStream {
  #boards = $state<Record<string, BoardState>>({});
  /** Per board, the last per-frame motion sample — kept apart from the reads. */
  #motion = $state<Record<string, MotionSample>>({});
  #connection = $state<ConnectionState>('connecting');
  /** Advanced by the ticker; every age, rate and ring fade is derived from it. */
  #now = $state(0);

  #socket: WebSocket | null = null;
  #running = false;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #reconnectMs: number;
  #ticker: ReturnType<typeof setInterval> | null = null;

  readonly #clock: () => number;
  readonly #factory: (url: string) => WebSocket;

  constructor(private readonly options: PositionStreamOptions = {}) {
    this.#clock = options.clock ?? (() => performance.now());
    this.#factory = options.socketFactory ?? ((url) => new WebSocket(url));
    this.#reconnectMs = options.reconnectMinMs ?? RECONNECT_MIN_MS;
    this.#now = this.#clock();
  }

  get connection(): ConnectionState {
    return this.#connection;
  }

  /** Boards in a stable order, so cards never reshuffle as events arrive. */
  get boards(): readonly BoardState[] {
    return Object.values(this.#boards).sort((a, b) => a.id.localeCompare(b.id));
  }

  board(id: string): BoardState | undefined {
    return this.#boards[id];
  }

  /** Freshness of one board against the shared clock — reactive, no timers per card. */
  liveness(board: BoardState): BoardLiveness {
    const now = this.#now;
    const ageMs = now - board.lastAt;
    return {
      ageMs,
      stale: isStale(ageMs),
      eventsPerSecond: eventsPerSecond(withinWindow(board.stamps, now)),
      rings: changeRings(board.changedAt, now),
      motion: motionReadout(this.#motion[board.id], now)
    };
  }

  start(tickMs = 200): void {
    if (this.#running) return;
    this.#running = true;
    this.#reconnectMs = this.options.reconnectMinMs ?? RECONNECT_MIN_MS;
    this.#now = this.#clock();
    this.#ticker = setInterval(() => {
      this.#now = this.#clock();
    }, tickMs);
    this.#connect();
  }

  stop(): void {
    this.#running = false;
    if (this.#ticker !== null) clearInterval(this.#ticker);
    this.#ticker = null;
    if (this.#reconnectTimer !== null) clearTimeout(this.#reconnectTimer);
    this.#reconnectTimer = null;
    const socket = this.#socket;
    this.#socket = null;
    socket?.close();
    this.#connection = 'disconnected';
  }

  /**
   * Fold one position event in (decoded JSON or its text). Public so a page (or a
   * test) can replay events without a socket; anything that is not a valid event
   * returns `null` and changes nothing.
   */
  ingest(raw: unknown): PositionEvent | null {
    const parsed = parseWire(PositionEvent, raw);
    return parsed.ok ? this.#foldEvent(parsed.value) : null;
  }

  /**
   * Fold one per-frame motion update in (see `./positions-motion`). Anything else —
   * a position event, garbage — returns `null` and changes nothing.
   */
  ingestMotion(raw: unknown): MotionUpdate | null {
    const parsed = parseWire(MotionUpdate, raw);
    return parsed.ok ? this.#foldMotion(parsed.value) : null;
  }

  #foldEvent(event: PositionEvent): PositionEvent {
    const now = this.#clock();
    this.#now = now;

    const previous = this.#boards[event.board_id];
    const grid = fenToGrid(event.settled?.placement ?? event.prior?.placement ?? event.placement);
    // Diffed against the previous GRID, which is this store's whole memory of the
    // past — see the module note on DESIGN D1.
    const changed = changedSquares(previous?.grid ?? null, grid);
    const changedAt: Partial<Record<SquareKey, number>> = pruneChanges(
      previous?.changedAt ?? {},
      now
    );
    for (const square of changed) changedAt[square] = now;

    this.#boards[event.board_id] = {
      id: event.board_id,
      event,
      grid,
      flipped: flippedSquares(event),
      pending: pendingSquares(event),
      changedAt,
      lastAt: now,
      stamps: [...withinWindow(previous?.stamps ?? [], now), now]
    };
    return event;
  }

  #foldMotion(message: MotionUpdate): MotionUpdate {
    const now = this.#clock();
    this.#now = now;
    const { moving, stability } = message;
    this.#motion[message.board_id] = { moving, stability, at: now };
    return message;
  }

  #connect(): void {
    const socket = this.#factory(this.options.url ?? wsUrl(VIEWER_SOCKET));
    this.#socket = socket;
    this.#connection = 'connecting';

    // Every handler stands down if a newer socket has taken over: a close arriving
    // after stop()/start() must neither null the live socket nor spawn a duplicate.
    socket.onopen = () => {
      if (socket !== this.#socket) return;
      this.#reconnectMs = this.options.reconnectMinMs ?? RECONNECT_MIN_MS;
      this.#connection = 'live';
    };
    socket.onmessage = (message: MessageEvent) => {
      if (socket !== this.#socket) return;
      if (typeof message.data !== 'string') return;
      // One stream, two message kinds: a motion update names its `type`, a
      // position event has none.
      const parsed = parseWire(ViewerMessage, message.data);
      if (!parsed.ok) return;
      if ('type' in parsed.value) this.#foldMotion(parsed.value);
      else this.#foldEvent(parsed.value);
    };
    socket.onerror = () => socket.close();
    socket.onclose = () => {
      if (socket !== this.#socket) return;
      this.#socket = null;
      this.#connection = 'disconnected';
      if (!this.#running) return;
      const wait = this.#reconnectMs;
      this.#reconnectMs = Math.min(
        this.#reconnectMs * 2,
        this.options.reconnectMaxMs ?? RECONNECT_MAX_MS
      );
      this.#reconnectTimer = setTimeout(() => {
        this.#reconnectTimer = null;
        if (this.#running && this.#socket === null) this.#connect();
      }, wait);
    };
  }
}
