/**
 * The control page's end of the stream list (`WS /streams` on the node, relayed
 * under `/ws/streams`): it sends `add_stream` (a video or live stream URL, and a
 * video's start position) and `remove_stream`, and hears the `streams` list or a
 * refusal. A stream the
 * node runs registers as a camera, so its card comes through the camera list.
 */

import { wsUrl } from '#lib/api.ts';
import { parseWire } from '#lib/protocol/index.ts';
import { AddStreamMessage, StreamsDown } from '#lib/protocol/generated.ts';
import { ReconnectingSocket } from '#lib/socket.ts';
import { removeStreamMessage } from '#lib/control/streams.ts';

export const STREAMS_SOCKET = '/ws/streams';

/** Where the page's last `add_stream` stands. */
export type AddStatus =
  | { readonly kind: 'idle' }
  | { readonly kind: 'sending'; readonly url: string }
  | { readonly kind: 'failed'; readonly reason: string };

export interface StreamsLinkOptions {
  readonly onMessage: (message: StreamsDown) => void;
  /** The socket opened (`up`) or was lost, with the reason to show. */
  readonly onLink: (up: boolean, detail: string) => void;
}

/** An `add_stream` the node would take, or why not, checked against the same validator. */
export function addStreamMessage(
  url: string,
  startS: number | null
): { readonly ok: true; readonly message: AddStreamMessage } | { readonly ok: false; readonly reason: string } {
  const parsed = AddStreamMessage.safeParse({ type: 'add_stream', url: url.trim(), start_s: startS });
  return parsed.success
    ? { ok: true, message: parsed.data }
    : { ok: false, reason: 'enter an http(s) address without spaces' };
}

export class StreamsLink {
  private readonly socket: ReconnectingSocket;

  constructor(options: StreamsLinkOptions) {
    this.socket = new ReconnectingSocket(
      { url: wsUrl(STREAMS_SOCKET) },
      {
        onOpen: () => options.onLink(true, 'connected'),
        onLost: (reason) => options.onLink(false, reason),
        onText: (text) => {
          const parsed = parseWire(StreamsDown, text);
          if (parsed.ok) options.onMessage(parsed.value);
        }
      }
    );
  }

  start(): void {
    this.socket.start();
  }

  stop(): void {
    this.socket.stop();
  }

  /** `false` when the node is not connected: nothing was sent. */
  add(message: AddStreamMessage): boolean {
    return this.socket.send(JSON.stringify(message));
  }

  /** Stop a stream and forget its camera. `false` when the node is not connected. */
  remove(streamId: string): boolean {
    return this.socket.send(removeStreamMessage(streamId));
  }
}
