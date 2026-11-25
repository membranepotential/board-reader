/**
 * The control page's list of every known camera: one socket, `WS /cameras`.
 *
 * The node pushes the whole list on connect, on every change (a phone connects or
 * leaves, reports, is renamed) and about once a second for the frame ages, so the
 * page polls nothing. A rename goes up as one `rename` message; the node answers
 * with the new list, or with a refusal (a name another camera holds, one that is no
 * board id).
 */

import { CAMERAS_SOCKET, wsUrl } from '#lib/api.ts';
import { parseWire } from '#lib/protocol/index.ts';
import { CamerasDown } from '#lib/protocol/generated.ts';
import { ReconnectingSocket, type SocketOptions } from '#lib/socket.ts';
import { renameMessage } from './messages';

export interface CamerasLinkOptions extends Partial<SocketOptions> {
  readonly onMessage: (message: CamerasDown) => void;
  /** The socket opened (`up`) or was lost, with the reason to show. */
  readonly onLink: (up: boolean, detail: string) => void;
}

export class CamerasLink {
  private readonly socket: ReconnectingSocket;

  constructor(options: CamerasLinkOptions) {
    const { onMessage, onLink, ...socket } = options;
    this.socket = new ReconnectingSocket(
      { ...socket, url: socket.url ?? wsUrl(CAMERAS_SOCKET) },
      {
        onOpen: () => onLink(true, 'connected'),
        onLost: (reason) => onLink(false, reason),
        onText: (text) => {
          const parsed = parseWire(CamerasDown, text);
          if (parsed.ok) onMessage(parsed.value);
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
  rename(deviceId: string, boardId: string): boolean {
    return this.socket.send(renameMessage(deviceId, boardId));
  }
}
