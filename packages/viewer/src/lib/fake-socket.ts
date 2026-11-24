/**
 * A `WebSocketLike` for unit tests: records what was sent, and lets a test open,
 * deliver to and close it. Not imported by any page.
 */

import type { WebSocketLike } from './socket';

export class FakeSocket implements WebSocketLike {
  readyState = 0;
  bufferedAmount = 0;
  closed = false;
  readonly sent: Array<string | Blob | ArrayBufferView | ArrayBufferLike> = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(readonly url: string) {}

  send(data: string | Blob | ArrayBufferView | ArrayBufferLike): void {
    this.sent.push(data);
  }

  /** The TEXT messages sent, parsed. */
  get texts(): unknown[] {
    return this.sent.filter((data) => typeof data === 'string').map((data) => JSON.parse(data) as unknown);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.({} as Event);
  }

  deliver(message: unknown): void {
    this.onmessage?.({ data: JSON.stringify(message) } as MessageEvent);
  }

  /** The far end hung up (or the client closed it). */
  close(code = 1000, reason = ''): void {
    if (this.closed) return;
    this.closed = true;
    this.readyState = 3;
    this.onclose?.({ code, reason } as CloseEvent);
  }
}

/** A socket factory that keeps every socket it made, and a manual timer queue. */
export function fakeNetwork(): {
  sockets: FakeSocket[];
  timers: Array<() => void>;
  options: {
    socketFactory: (url: string) => FakeSocket;
    setTimeoutFn: (fn: () => void, ms: number) => number;
    clearTimeoutFn: (handle: number) => void;
  };
} {
  const sockets: FakeSocket[] = [];
  const timers: Array<() => void> = [];
  return {
    sockets,
    timers,
    options: {
      socketFactory: (url) => {
        const socket = new FakeSocket(url);
        sockets.push(socket);
        return socket;
      },
      setTimeoutFn: (fn) => timers.push(fn),
      clearTimeoutFn: () => {}
    }
  };
}
