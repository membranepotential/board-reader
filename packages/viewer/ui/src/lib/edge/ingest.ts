/**
 * The `/ws/ingest` client — the browser edge's end of the L1→L2 wire.
 *
 * **The framing is two frames per image and must stay that way**: a JSON header as
 * a TEXT frame, immediately followed by its JPEG as a BINARY frame
 * (`board_reader_core.wire`, and `proxy.py::_client_to_upstream`, which forwards
 * whichever kind arrived precisely so the pair survives the relay). Collapsing them
 * into one frame, reordering them, or letting anything else be sent between them
 * corrupts the stream from that point on — the node reads header/JPEG in lockstep
 * (`server.ingest`). That is why `sendFrame` is the only way to send: the two
 * `send` calls sit next to each other with no `await` between them, so nothing can
 * interleave.
 *
 * Acks mean "received", not "estimated" — the node's pre-filter may drop an
 * acknowledged frame for a fresher one (DESIGN D3/D4).
 */

import { FRAME_TYPE, parseIngestReply, type IngestFrameHeader, type IngestState } from '../protocol';
import { INGEST_SOCKET, wsUrl } from '../api';
import {
  OPEN,
  RECONNECT_MAX_MS,
  RECONNECT_MIN_MS,
  type SocketFactory,
  type WebSocketLike
} from '../socket';

/** ~3 frames' worth: above this the link is behind and another frame only makes the reading staler. */
export const MAX_BUFFERED = 192 * 1024;

/** What the operator is told about the link, in the words the page shows. */

export type SendOutcome =
  | 'sent'
  /** The link is behind; dropping is the correct answer at both rates (DESIGN D3/D4). */
  | 'behind'
  | 'not_connected';

export interface OutgoingFrame {
  readonly boardId: string;
  /** Epoch SECONDS at capture (DESIGN D5). */
  readonly captureTs: number;
  readonly stability: number;
  /**
   * Focus score of the sent crop (tenengrad), unitless — the node compares it
   * only between neighbouring frames, to spend its ~1 s read on a crisp one.
   * Omitted by an edge that does not measure it.
   */
  readonly sharpness?: number;
  readonly burstId: string | null;
  readonly jpeg: Blob;
}

export interface IngestClientOptions {
  /**
   * The socket address. Relative by default and derived from the page (`api.wsUrl`)
   * — nothing here may name an origin. Overridden only by tests, which have no page.
   */
  readonly url?: string;
  readonly socketFactory?: SocketFactory;
  readonly maxBuffered?: number;
  readonly reconnectMinMs?: number;
  readonly reconnectMaxMs?: number;
  readonly onStatus?: (status: IngestState, detail: string) => void;
  readonly onAck?: (seq: number, boardId: string) => void;
  /** The node refusing ONE frame (a truncated JPEG, a malformed header) — not a disconnect. */
  readonly onFrameRefused?: (reason: string) => void;
  readonly setTimeoutFn?: (fn: () => void, ms: number) => number;
  readonly clearTimeoutFn?: (handle: number) => void;
}

/**
 * One live ingest connection, its sequence numbers and its ack accounting.
 *
 * Every handler closes over ITS socket and stands down if a newer one has taken
 * over — a stop→start inside a close handshake, or a pending reconnect timer firing
 * late, must neither null the live socket nor spawn a duplicate connection.
 */
export class IngestClient {
  private socket: WebSocketLike | null = null;
  private streaming = false;
  private reconnectMs: number;
  private reconnectTimer: number | null = null;

  /** Monotonic per client, as the wire wants (`wire.Frame.seq`). */
  private seq = 0;
  private sentCount = 0;
  private droppedCount = 0;
  private ackedSeq: number | null = null;

  /**
   * The board the most recent frame was sent for, so an ack can be matched to it.
   *
   * Acks arrive asynchronously and name their board. Without this, switching board
   * mid-stream lets an ack for the *previous* board land after `forgetAcks()` and report
   * progress for a board this client is no longer streaming.
   */
  private streamingBoardId: string | null = null;

  private readonly factory: SocketFactory;
  private readonly maxBuffered: number;
  private readonly minReconnect: number;
  private readonly maxReconnect: number;
  private readonly schedule: (fn: () => void, ms: number) => number;
  private readonly unschedule: (handle: number) => void;

  constructor(private readonly options: IngestClientOptions = {}) {
    this.factory = options.socketFactory ?? ((url) => new WebSocket(url));
    this.maxBuffered = options.maxBuffered ?? MAX_BUFFERED;
    this.minReconnect = options.reconnectMinMs ?? RECONNECT_MIN_MS;
    this.maxReconnect = options.reconnectMaxMs ?? RECONNECT_MAX_MS;
    this.reconnectMs = this.minReconnect;
    // `setTimeout` answers a number in a browser and a Timeout object under node;
    // the handle is opaque to us either way, so it is carried as a number.
    this.schedule = options.setTimeoutFn ?? ((fn, ms) => setTimeout(fn, ms) as unknown as number);
    this.unschedule = options.clearTimeoutFn ?? ((handle) => clearTimeout(handle));
  }

  get connected(): boolean {
    return this.socket !== null && this.socket.readyState === OPEN;
  }

  get sent(): number {
    return this.sentCount;
  }

  get dropped(): number {
    return this.droppedCount;
  }

  get acked(): number | null {
    return this.ackedSeq;
  }

  /**
   * Whether the link is already behind — the same condition ``sendFrame`` refuses on.
   *
   * Exposed so a caller can skip the **JPEG encode** as well as the send. `sendFrame`
   * can only check once it has been handed a blob, by which point a full canvas readback
   * and compression have already been paid for — per frame, at up to the burst rate, on
   * a phone whose uplink is the bottleneck. That is exactly when its battery matters
   * most, and the frame is going to be dropped either way.
   *
   * An optimisation, never the guard: `sendFrame` re-checks, because the buffer can fill
   * during the encode this skips.
   */
  get behind(): boolean {
    const socket = this.socket;
    return (
      socket !== null &&
      socket.readyState === OPEN &&
      socket.bufferedAmount > this.maxBuffered
    );
  }

  /** A frame the CALLER dropped (its encoder was still busy) — counted, not sent. */
  noteDropped(): void {
    this.droppedCount++;
  }

  /**
   * Acks are per board; switching board makes the old board's ack meaningless.
   *
   * Clearing the count is only half of it — see `streamingBoardId`. An ack for the old
   * board can still be in flight over the socket when this is called, and accepting it
   * would re-populate the counter the switch just cleared.
   */
  forgetAcks(): void {
    this.ackedSeq = null;
    this.streamingBoardId = null;
  }

  start(): void {
    if (this.streaming) return;
    this.streaming = true;
    this.droppedCount = 0;
    this.reconnectMs = this.minReconnect;
    this.status('connecting', 'connecting…');
    this.connect();
  }

  stop(): void {
    this.streaming = false;
    if (this.reconnectTimer !== null) {
      this.unschedule(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const socket = this.socket;
    this.socket = null;
    socket?.close();
    this.status('idle', 'idle');
  }

  /**
   * Header (text) then JPEG (binary), adjacent and in that order. Nothing may be
   * sent between them and nothing may be awaited between them — see the module note.
   */
  sendFrame(frame: OutgoingFrame): SendOutcome {
    const socket = this.socket;
    if (socket === null || socket.readyState !== OPEN) return 'not_connected';
    if (socket.bufferedAmount > this.maxBuffered) {
      this.droppedCount++;
      return 'behind';
    }
    this.streamingBoardId = frame.boardId;
    const header: IngestFrameHeader = {
      type: FRAME_TYPE,
      board_id: frame.boardId,
      seq: this.seq++,
      capture_ts: frame.captureTs,
      stability: frame.stability,
      burst_id: frame.burstId,
      ...(frame.sharpness === undefined ? {} : { sharpness: frame.sharpness })
    };
    socket.send(JSON.stringify(header));
    socket.send(frame.jpeg);
    this.sentCount++;
    return 'sent';
  }

  private status(status: IngestState, detail: string): void {
    this.options.onStatus?.(status, detail);
  }

  private connect(): void {
    const socket = this.factory(this.options.url ?? wsUrl(INGEST_SOCKET));
    this.socket = socket;

    socket.onopen = () => {
      if (socket !== this.socket) return;
      this.reconnectMs = this.minReconnect;
      this.status('streaming', 'streaming');
    };

    socket.onmessage = (event) => {
      if (socket !== this.socket) return;
      if (typeof event.data !== 'string') return;
      const reply = parseIngestReply(event.data);
      if (reply === null) return; // the node speaks only acks here; anything else is not ours to mind
      if (reply.kind === 'ack') {
        // Only the board we are actually streaming. An ack naming a board we have
        // switched away from is in-flight history: counting it would undo the
        // `forgetAcks()` that the switch performed, and report progress for a board
        // this client no longer sends.
        if (reply.board_id !== this.streamingBoardId) return;
        this.ackedSeq = reply.seq;
        this.options.onAck?.(reply.seq, reply.board_id);
      } else {
        // One bad frame is refused in the ack slot and the socket stays up — a
        // browser edge that reconnected on its own bug would loop forever.
        this.options.onFrameRefused?.(reply.error);
      }
    };

    socket.onerror = () => socket.close();

    socket.onclose = (event) => {
      if (socket !== this.socket) return;
      this.socket = null;
      if (!this.streaming) return;
      // The relay closes with 1011 and the upstream's own error when the inference
      // node is the thing that is down — worth reading rather than hiding behind
      // "closed".
      const reason = event.reason ? `reconnecting — ${event.reason}` : 'reconnecting…';
      this.status('reconnecting', reason);
      const wait = this.reconnectMs;
      this.reconnectTimer = this.schedule(() => {
        this.reconnectTimer = null;
        if (this.streaming && this.socket === null) this.connect();
      }, wait);
      this.reconnectMs = Math.min(this.reconnectMs * 2, this.maxReconnect);
    };
  }
}
