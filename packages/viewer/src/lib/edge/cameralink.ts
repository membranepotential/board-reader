/**
 * The phone's end of the camera control channel: one socket, by its device id.
 *
 * The node says which board id the phone streams as and what it is called
 * (`assigned`) — on connect, and again with the new name whenever the control UI
 * renames it; the board id never changes — pushes the config the control UI set — on
 * connect and on every change — and says whether anyone is watching (`previews`).
 * The phone sends its `CameraReport` on connect and whenever it changes, never on a
 * timer, and a whole-frame preview JPEG only while previews are wanted.
 *
 * A device has one socket: when another page of the same phone connects, the node
 * says `replaced` to this one, which then stops for good rather than reconnect and
 * take the camera back, which would make two pages swap it forever.
 *
 * The frames themselves stay on `/ws/ingest` (`ingest.ts`): its two-message framing
 * and its recording are left exactly as every edge and every replay knows them.
 *
 * A lost link changes nothing the phone is doing: it keeps the last config it heard
 * and keeps streaming on `/ws/ingest`, and the node pushes the config again when the
 * link comes back.
 */

import { cameraSocket, wsUrl } from '#lib/api.ts';
import { reportMessage } from '#lib/control/messages.ts';
import { parseWire } from '#lib/protocol/index.ts';
import { PhoneDown, type CameraConfig, type CameraReport } from '#lib/protocol/generated.ts';
import { ReconnectingSocket, type SocketOptions } from '#lib/socket.ts';

export interface CameraLinkOptions extends Partial<SocketOptions> {
  readonly deviceId: string;
  /** The board id the node says this phone streams as, and its name if it has one. */
  readonly onAssigned: (boardId: string, name: string | null) => void;
  /** Another page of this phone took over; this link has stopped. */
  readonly onReplaced: () => void;
  readonly onConfig: (config: CameraConfig) => void;
  readonly onPreviews: (on: boolean) => void;
  /** The link opened (`up`) or was lost, with the reason to show. */
  readonly onLink: (up: boolean, detail: string) => void;
  readonly onRefused?: (reason: string) => void;
}

export class CameraLink {
  private readonly socket: ReconnectingSocket;
  private latest: CameraReport | null = null;
  /** The report text last sent on the CURRENT socket; reset on every open. */
  private sent: string | null = null;
  private previews = false;

  constructor(private readonly options: CameraLinkOptions) {
    const { deviceId, onAssigned, onReplaced, onConfig, onPreviews, onLink, onRefused, ...socket } = options;
    this.socket = new ReconnectingSocket(
      { ...socket, url: socket.url ?? wsUrl(cameraSocket(deviceId)) },
      {
        onOpen: () => {
          this.sent = null;
          this.flush();
          onLink(true, 'connected');
        },
        onLost: (reason) => {
          // The node tells a fresh socket again; until then nobody is known to watch.
          this.setPreviews(false);
          onLink(false, reason);
        },
        onText: (text) => {
          const parsed = parseWire(PhoneDown, text);
          if (!parsed.ok) return; // a newer node's message; not ours to mind
          const message = parsed.value;
          switch (message.type) {
            case 'assigned':
              onAssigned(message.board_id, message.name ?? null);
              break;
            case 'config':
              onConfig(message.config);
              break;
            case 'previews':
              this.setPreviews(message.on);
              break;
            case 'replaced':
              this.stop();
              onReplaced();
              break;
            case 'refused':
              onRefused?.(message.reason);
              break;
          }
        }
      }
    );
  }

  /** Whether a control page is watching, so a preview is worth sending. */
  get previewsWanted(): boolean {
    return this.previews;
  }

  start(): void {
    this.socket.start();
  }

  stop(): void {
    this.socket.stop();
    this.setPreviews(false);
  }

  /** What the camera is now. Sent at once if it changed; on the next open otherwise. */
  report(report: CameraReport): void {
    this.latest = report;
    this.flush();
  }

  /**
   * A whole-frame preview, sent only while wanted and while the last one has left:
   * a preview still in the send buffer means the next would only arrive staler.
   */
  sendPreview(jpeg: Blob): boolean {
    if (!this.previews || this.socket.bufferedAmount > 0) return false;
    return this.socket.send(jpeg);
  }

  private flush(): void {
    if (this.latest === null) return;
    const text = reportMessage(this.latest);
    if (text === this.sent) return;
    if (this.socket.send(text)) this.sent = text;
  }

  private setPreviews(on: boolean): void {
    if (on === this.previews) return;
    this.previews = on;
    this.options.onPreviews(on);
  }
}
