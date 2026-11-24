/**
 * A WebSocket kept open while started: it reconnects by itself, with backoff.
 *
 * The camera control channel's two clients (`edge/cameralink.ts` on the phone,
 * `control/link.ts` on the laptop) hold one of these each. Every handler closes
 * over ITS socket and stands down when a newer one has taken over, because a stale
 * socket's late `close` would otherwise null the live connection and spawn a
 * duplicate (a stop→start inside a close handshake, a reconnect timer firing late).
 */

export const OPEN = 1;

/**
 * Just enough of a `WebSocket` for the clients, so a test can be a plain object.
 * The DOM event types are used as-is: a real `WebSocket` must satisfy this without
 * a cast, because the point is that production runs the code the tests run.
 */
export interface WebSocketLike {
  readonly readyState: number;
  readonly bufferedAmount: number;
  send(data: string | Blob | ArrayBufferView | ArrayBufferLike): void;
  close(code?: number, reason?: string): void;
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
}

export type SocketFactory = (url: string) => WebSocketLike;

const defaultSocketFactory: SocketFactory = (url) => new WebSocket(url);

/** Quick enough to feel instant on a blip, slow enough not to hammer a node that is down. */
export const RECONNECT_MIN_MS = 500;
export const RECONNECT_MAX_MS = 8000;

export interface SocketOptions {
  readonly url: string;
  readonly socketFactory?: SocketFactory;
  readonly reconnectMinMs?: number;
  readonly reconnectMaxMs?: number;
  readonly setTimeoutFn?: (fn: () => void, ms: number) => number;
  readonly clearTimeoutFn?: (handle: number) => void;
}

export interface SocketHandlers {
  /** A connection opened (the first, or a reconnect). */
  readonly onOpen: () => void;
  /** A TEXT message arrived; the control channel's sockets speak nothing else down. */
  readonly onText: (text: string) => void;
  /** A connection was lost while started; a reconnect is already scheduled. */
  readonly onLost: (reason: string) => void;
}

export class ReconnectingSocket {
  private socket: WebSocketLike | null = null;
  private running = false;
  private reconnectMs: number;
  private reconnectTimer: number | null = null;

  private readonly factory: SocketFactory;
  private readonly minReconnect: number;
  private readonly maxReconnect: number;
  private readonly schedule: (fn: () => void, ms: number) => number;
  private readonly unschedule: (handle: number) => void;

  constructor(
    private readonly options: SocketOptions,
    private readonly handlers: SocketHandlers
  ) {
    this.factory = options.socketFactory ?? defaultSocketFactory;
    this.minReconnect = options.reconnectMinMs ?? RECONNECT_MIN_MS;
    this.maxReconnect = options.reconnectMaxMs ?? RECONNECT_MAX_MS;
    this.reconnectMs = this.minReconnect;
    // `setTimeout` answers a number in a browser and a Timeout object under node;
    // the handle is opaque to us either way, so it is carried as a number.
    this.schedule = options.setTimeoutFn ?? ((fn, ms) => setTimeout(fn, ms) as unknown as number);
    this.unschedule = options.clearTimeoutFn ?? ((handle) => clearTimeout(handle));
  }

  get open(): boolean {
    return this.socket !== null && this.socket.readyState === OPEN;
  }

  get bufferedAmount(): number {
    return this.socket?.bufferedAmount ?? 0;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.reconnectMs = this.minReconnect;
    this.connect();
  }

  stop(): void {
    this.running = false;
    if (this.reconnectTimer !== null) {
      this.unschedule(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const socket = this.socket;
    this.socket = null;
    socket?.close();
  }

  /** `false` when there is no open socket to send on: nothing is queued. */
  send(data: string | Blob): boolean {
    const socket = this.socket;
    if (socket === null || socket.readyState !== OPEN) return false;
    socket.send(data);
    return true;
  }

  private connect(): void {
    const socket = this.factory(this.options.url);
    this.socket = socket;

    socket.onopen = () => {
      if (socket !== this.socket) return;
      this.reconnectMs = this.minReconnect;
      this.handlers.onOpen();
    };

    socket.onmessage = (event) => {
      if (socket !== this.socket) return;
      if (typeof event.data === 'string') this.handlers.onText(event.data);
    };

    socket.onerror = () => socket.close();

    socket.onclose = (event) => {
      if (socket !== this.socket) return;
      this.socket = null;
      if (!this.running) return;
      // The relay closes with 1011 and the upstream's own error when the node is
      // the thing that is down — worth reading rather than hiding behind "closed".
      this.handlers.onLost(event.reason || 'connection lost');
      this.reconnectTimer = this.schedule(() => {
        this.reconnectTimer = null;
        if (this.running && this.socket === null) this.connect();
      }, this.reconnectMs);
      this.reconnectMs = Math.min(this.reconnectMs * 2, this.maxReconnect);
    };
  }
}
