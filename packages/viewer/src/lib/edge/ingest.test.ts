import { describe, expect, it, vi } from 'vitest';
import { IngestClient } from './ingest';
import type { WebSocketLike } from '../socket';
import { FrameHeader } from '../protocol/generated';

/** A `WebSocket` that records what was sent, in order and by frame kind. */
class FakeSocket implements WebSocketLike {
  readyState = 1;
  bufferedAmount = 0;
  closed = false;
  readonly frames: Array<{ kind: 'text' | 'binary'; data: unknown }> = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  send(data: string | Blob | ArrayBufferView | ArrayBufferLike): void {
    this.frames.push({ kind: typeof data === 'string' ? 'text' : 'binary', data });
  }

  close(): void {
    this.closed = true;
    this.onclose?.({ code: 1000, reason: '' } as CloseEvent);
  }

  deliver(text: string): void {
    this.onmessage?.({ data: text } as MessageEvent);
  }

  open(): void {
    this.onopen?.({} as Event);
  }
}

function connected(): { client: IngestClient; sockets: FakeSocket[] } {
  const sockets: FakeSocket[] = [];
  const client = new IngestClient({
    url: 'ws://test/ws/ingest',
    socketFactory: () => {
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket;
    }
  });
  client.start();
  sockets[0]!.open();
  return { client, sockets };
}

const frame = (extra: Partial<Parameters<IngestClient['sendFrame']>[0]> = {}) => ({
  boardId: 'phone',
  deviceId: 'kitchen-phone-id',
  captureTs: 1_754_560_800.5,
  stability: 0.9,
  burstId: null,
  jpeg: new Blob(['jpeg-bytes']),
  ...extra
});

describe('the two-frame ingest wire', () => {
  it('sends the JSON header as TEXT and its JPEG as BINARY, in that order', () => {
    const { client, sockets } = connected();
    expect(client.sendFrame(frame())).toBe('sent');
    expect(sockets[0]!.frames.map((f) => f.kind)).toEqual(['text', 'binary']);
  });

  it('keeps the pair adjacent across consecutive frames', () => {
    const { client, sockets } = connected();
    client.sendFrame(frame());
    client.sendFrame(frame());
    expect(sockets[0]!.frames.map((f) => f.kind)).toEqual(['text', 'binary', 'text', 'binary']);
  });

  it('writes the header shape board_reader_core.wire reads', () => {
    const { client, sockets } = connected();
    client.sendFrame(frame({ burstId: 'phone:b0' }));
    const header = JSON.parse(sockets[0]!.frames[0]!.data as string) as Record<string, unknown>;
    expect(header).toEqual({
      type: 'frame',
      board_id: 'phone',
      seq: 0,
      capture_ts: 1_754_560_800.5,
      stability: 0.9,
      sharpness: null,
      clock: 'none',
      burst_id: 'phone:b0',
      device_id: 'kitchen-phone-id'
    });
    // Every key the schema states, so the node's strict `FrameHeader` accepts it.
    expect(FrameHeader.safeParse(header).success).toBe(true);
  });

  it('never puts corners on the wire: the recognizer finds the board itself', () => {
    const { client, sockets } = connected();
    client.sendFrame(frame());
    const header = JSON.parse(sockets[0]!.frames[0]!.data as string) as Record<string, unknown>;
    expect('corners' in header).toBe(false);
  });

  it('numbers frames monotonically and only for frames that went out', () => {
    const { client, sockets } = connected();
    client.sendFrame(frame());
    sockets[0]!.bufferedAmount = 10 * 1024 * 1024;
    expect(client.sendFrame(frame())).toBe('behind');
    sockets[0]!.bufferedAmount = 0;
    client.sendFrame(frame());
    const seqs = sockets[0]!.frames
      .filter((f) => f.kind === 'text')
      .map((f) => (JSON.parse(f.data as string) as { seq: number }).seq);
    expect(seqs).toEqual([0, 1]);
  });
});

describe('backlog and drops', () => {
  it('drops rather than queues when the link is behind (DESIGN D3/D4)', () => {
    const { client, sockets } = connected();
    sockets[0]!.bufferedAmount = 1024 * 1024;
    expect(client.sendFrame(frame())).toBe('behind');
    expect(sockets[0]!.frames).toHaveLength(0);
    expect(client.dropped).toBe(1);
  });

  it('says so rather than half-sending when the socket is not open', () => {
    const { client, sockets } = connected();
    sockets[0]!.readyState = 3;
    expect(client.sendFrame(frame())).toBe('not_connected');
    expect(sockets[0]!.frames).toHaveLength(0);
  });

  it('counts a frame the caller dropped because its encoder was busy', () => {
    const { client } = connected();
    client.noteDropped();
    expect(client.dropped).toBe(1);
  });

  // `behind` is what lets the caller skip the JPEG encode for a frame `sendFrame` is
  // going to refuse anyway. If it ever stopped agreeing with `sendFrame`, the camera
  // page would either encode frames it discards (the regression this restores) or skip
  // frames the link could have carried.
  it('reports the link as behind on exactly the condition sendFrame refuses on', () => {
    const { client, sockets } = connected();
    expect(client.behind).toBe(false);

    sockets[0]!.bufferedAmount = 1024 * 1024;
    expect(client.behind).toBe(true);
    expect(client.sendFrame(frame())).toBe('behind');

    sockets[0]!.bufferedAmount = 0;
    expect(client.behind).toBe(false);
    expect(client.sendFrame(frame())).toBe('sent');
  });

  // The original page filtered acks by board (`camera.html`: `d.board_id === boardId()`).
  // Without it, switching board mid-stream lets a late ack for the old board undo the
  // `forgetAcks()` the switch just performed.
  it('ignores an ack for a board it has switched away from', () => {
    const { client, sockets } = connected();
    client.sendFrame(frame());
    sockets[0]!.deliver(JSON.stringify({ ack: 0, board_id: 'phone' }));
    expect(client.acked).toBe(0);

    client.forgetAcks();
    expect(client.acked).toBeNull();

    // In flight when the operator switched boards — names the board we no longer stream.
    sockets[0]!.deliver(JSON.stringify({ ack: 1, board_id: 'phone' }));
    expect(client.acked).toBeNull();

    // The new board's own acks land normally.
    client.sendFrame(frame({ boardId: 'table-2' }));
    sockets[0]!.deliver(JSON.stringify({ ack: 2, board_id: 'table-2' }));
    expect(client.acked).toBe(2);
  });

  it('is not behind when there is no open socket to be behind on', () => {
    const { client, sockets } = connected();
    sockets[0]!.bufferedAmount = 1024 * 1024;
    sockets[0]!.readyState = 3;
    // Otherwise a closed socket's stale buffer would read as backpressure and the page
    // would skip encoding forever instead of reporting `not_connected`.
    expect(client.behind).toBe(false);
    expect(client.sendFrame(frame())).toBe('not_connected');
  });
});

describe('ack accounting', () => {
  it('records the latest ack', () => {
    const acks: number[] = [];
    const sockets: FakeSocket[] = [];
    const client = new IngestClient({
      url: 'ws://test',
      socketFactory: () => {
        const s = new FakeSocket();
        sockets.push(s);
        return s;
      },
      onAck: (seq) => acks.push(seq)
    });
    client.start();
    sockets[0]!.open();
    // Send first: the node only acks frames it received, and the client matches an ack
    // against the board it is streaming, so an ack out of nowhere is not a real event.
    client.sendFrame(frame());
    sockets[0]!.deliver('{"ack": 4, "board_id": "phone"}');
    expect(client.acked).toBe(4);
    expect(acks).toEqual([4]);
  });

  it('surfaces a per-frame refusal without dropping the socket', () => {
    const refusals: string[] = [];
    const sockets: FakeSocket[] = [];
    const client = new IngestClient({
      url: 'ws://test',
      socketFactory: () => {
        const s = new FakeSocket();
        sockets.push(s);
        return s;
      },
      onFrameRefused: (reason) => refusals.push(reason)
    });
    client.start();
    sockets[0]!.open();
    sockets[0]!.deliver('{"error": "bad frame: corners must be 4 (x, y) pairs"}');
    expect(refusals).toEqual(['bad frame: corners must be 4 (x, y) pairs']);
    expect(sockets[0]!.closed).toBe(false);
    expect(client.acked).toBeNull();
  });

  it('ignores anything that is not an ack or a refusal', () => {
    const { client, sockets } = connected();
    sockets[0]!.deliver('hello');
    expect(client.acked).toBeNull();
  });
});

describe('reconnect', () => {
  it('backs off exponentially, capped', () => {
    const waits: number[] = [];
    const pending: Array<() => void> = [];
    const sockets: FakeSocket[] = [];
    const client = new IngestClient({
      url: 'ws://test',
      socketFactory: () => {
        const s = new FakeSocket();
        sockets.push(s);
        return s;
      },
      reconnectMinMs: 100,
      reconnectMaxMs: 400,
      setTimeoutFn: (fn, ms) => {
        waits.push(ms);
        pending.push(fn);
        return waits.length;
      },
      clearTimeoutFn: () => {}
    });
    client.start();
    for (let i = 0; i < 4; i++) {
      sockets[sockets.length - 1]!.close();
      pending.pop()?.();
    }
    expect(waits).toEqual([100, 200, 400, 400]);
  });

  it('resets the backoff once a connection succeeds', () => {
    const waits: number[] = [];
    const pending: Array<() => void> = [];
    const sockets: FakeSocket[] = [];
    const client = new IngestClient({
      url: 'ws://test',
      socketFactory: () => {
        const s = new FakeSocket();
        sockets.push(s);
        return s;
      },
      reconnectMinMs: 100,
      setTimeoutFn: (fn, ms) => {
        waits.push(ms);
        pending.push(fn);
        return waits.length;
      },
      clearTimeoutFn: () => {}
    });
    client.start();
    sockets[0]!.close();
    pending.pop()?.();
    sockets[1]!.open();
    sockets[1]!.close();
    expect(waits).toEqual([100, 100]);
  });

  it('does not reconnect after stop()', () => {
    const sockets: FakeSocket[] = [];
    const schedule = vi.fn(() => 1);
    const client = new IngestClient({
      url: 'ws://test',
      socketFactory: () => {
        const s = new FakeSocket();
        sockets.push(s);
        return s;
      },
      setTimeoutFn: schedule,
      clearTimeoutFn: () => {}
    });
    client.start();
    client.stop();
    expect(schedule).not.toHaveBeenCalled();
    expect(sockets).toHaveLength(1);
  });

  it('lets a stale socket close without touching the live one', () => {
    // stop() -> start() inside a close handshake: the old socket's close handler must
    // neither null the new socket nor spawn a duplicate connection.
    const sockets: FakeSocket[] = [];
    const schedule = vi.fn(() => 1);
    const client = new IngestClient({
      url: 'ws://test',
      socketFactory: () => {
        const s = new FakeSocket();
        sockets.push(s);
        return s;
      },
      setTimeoutFn: schedule,
      clearTimeoutFn: () => {}
    });
    client.start();
    const stale = sockets[0]!;
    client.stop();
    client.start();
    sockets[1]!.open();
    stale.onclose?.({ code: 1006, reason: 'late' } as CloseEvent);
    expect(schedule).not.toHaveBeenCalled();
    expect(client.connected).toBe(true);
    expect(sockets).toHaveLength(2);
  });

  it('reports the upstream reason the relay closed with', () => {
    const statuses: Array<[string, string]> = [];
    const sockets: FakeSocket[] = [];
    const client = new IngestClient({
      url: 'ws://test',
      socketFactory: () => {
        const s = new FakeSocket();
        sockets.push(s);
        return s;
      },
      onStatus: (status, detail) => statuses.push([status, detail]),
      setTimeoutFn: () => 1,
      clearTimeoutFn: () => {}
    });
    client.start();
    sockets[0]!.onclose?.({
      code: 1011,
      reason: 'ws://localhost:8000/ingest: connection refused'
    } as CloseEvent);
    expect(statuses.at(-1)).toEqual([
      'reconnecting',
      'reconnecting — ws://localhost:8000/ingest: connection refused'
    ]);
  });
});
