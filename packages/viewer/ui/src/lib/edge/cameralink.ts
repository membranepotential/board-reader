/**
 * The phone's end of the camera control channel: one socket for its board.
 *
 * The node pushes the config the control UI set — on connect and on every change —
 * and says whether anyone is watching (`previews`). The phone sends its
 * `CameraReport` on connect and whenever it changes, never on a timer, and a
 * whole-frame preview JPEG only while previews are wanted.
 *
 * The frames themselves stay on `/ws/ingest` (`ingest.ts`): its two-message framing
 * and its recording are left exactly as every edge and every replay knows them.
 *
 * A lost link changes nothing the phone is doing: it keeps the last config it heard
 * and keeps streaming on `/ws/ingest`, and the node pushes the config again when the
 * link comes back.
 */

import { cameraSocket, wsUrl } from '$lib/api';
import { parsePhoneDown, reportMessage } from '$lib/control/messages';
import type { CameraConfig, CameraReport } from '$lib/protocol';
import { ReconnectingSocket, type SocketOptions } from '$lib/socket';

export interface CameraLinkOptions extends Partial<SocketOptions> {
  readonly boardId: string;
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
    const { boardId, onConfig, onPreviews, onLink, onRefused, ...socket } = options;
    this.socket = new ReconnectingSocket(
      { ...socket, url: socket.url ?? wsUrl(cameraSocket(boardId)) },
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
          const parsed = parsePhoneDown(text);
          if (!parsed.ok) return; // a newer node's message; not ours to mind
          const message = parsed.value;
          if (message.type === 'config') onConfig(message.config);
          else if (message.type === 'previews') this.setPreviews(message.on);
          else onRefused?.(message.reason);
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
