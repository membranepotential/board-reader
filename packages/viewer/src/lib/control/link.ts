/**
 * The control page's end of the camera control channel: one socket per board.
 *
 * The node pushes everything the page shows — the camera's state on connect and on
 * every change, the phone's preview, the latest received frame and a health tick —
 * so the page polls nothing. A config goes up as one `configure` message carrying
 * the whole config; the node answers every page watching the board with the new
 * state, or this page alone with a refusal.
 */

import { controlSocket, wsUrl } from '#lib/api.ts';
import { parseWire } from '#lib/protocol/index.ts';
import { ControlDown, type CameraConfig } from '#lib/protocol/generated.ts';
import { ReconnectingSocket, type SocketOptions } from '#lib/socket.ts';
import { configureMessage } from './messages';

export interface ControlLinkOptions extends Partial<SocketOptions> {
  readonly boardId: string;
  readonly onMessage: (message: ControlDown) => void;
  /** The socket opened (`up`) or was lost, with the reason to show. */
  readonly onLink: (up: boolean, detail: string) => void;
}

export class ControlLink {
  private readonly socket: ReconnectingSocket;

  constructor(options: ControlLinkOptions) {
    const { boardId, onMessage, onLink, ...socket } = options;
    this.socket = new ReconnectingSocket(
      { ...socket, url: socket.url ?? wsUrl(controlSocket(boardId)) },
      {
        onOpen: () => onLink(true, 'connected'),
        onLost: (reason) => onLink(false, reason),
        onText: (text) => {
          const parsed = parseWire(ControlDown, text);
          // A message this page cannot read is the node speaking a newer contract;
          // dropping it keeps the last good state on screen.
          if (parsed.ok) onMessage(parsed.value);
        }
      }
    );
  }

  get up(): boolean {
    return this.socket.open;
  }

  start(): void {
    this.socket.start();
  }

  stop(): void {
    this.socket.stop();
  }

  /** `false` when the node is not connected: nothing was sent. */
  configure(config: CameraConfig): boolean {
    return this.socket.send(configureMessage(config));
  }
}
