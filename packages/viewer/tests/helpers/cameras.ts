/**
 * A stand-in inference node for several phones and a control page at once.
 *
 * It plays `control.camera.registry.CameraRegistry`'s rules on the sockets the pages open,
 * from the test process, so two browser contexts (two phones, each with its own
 * `localStorage`, so its own device id) and a control page meet one node:
 *
 *  - `/ws/cameras/<device>`: the device streams under its device id as its board id;
 *    a new device is named the lowest `board-N` no other camera is called; a second
 *    socket of one device replaces the first.
 *  - `/ws/cameras`: the list, pushed on every change; `rename` sets a camera's name
 *    (never another camera's name or board id; `null` a default name again),
 *    `forget` drops a disconnected one.
 *  - `/ws/boards/<id>/control`: the board's state; `configure` reaches its phone.
 *  - `/ws/ingest`: every frame header, kept to read which board a frame named.
 *
 * `restart()` drops every socket and keeps only what the node writes to its config
 * file (the names and the configs), the way a restarted node comes back. The file itself is the Python suite's to test (`test_persistence.py`).
 */

import type { BrowserContext, WebSocketRoute } from '@playwright/test';

export interface Config {
  crop: { x: number; y: number; w: number; h: number } | null;
  device: string | null;
  resolution: '720p' | '1080p' | '2160p';
  streaming: boolean;
  rules: 'standard' | 'free';
  auto: boolean;
}

const DEFAULT_CONFIG: Config = {
  crop: null,
  device: null,
  resolution: '1080p',
  streaming: false,
  rules: 'standard',
  auto: true
};

const PHONE_SOCKET = /\/ws\/cameras\/[^/?]+(\?.*)?$/;
const LIST_SOCKET = /\/ws\/cameras$/;
const CONTROL_SOCKET = /\/ws\/boards\/([^/]+)\/control$/;

export class FakeCameraNode {
  /** What survives a restart: each known camera's name (by device id), each board's config. */
  readonly names = new Map<string, string>();
  readonly configs = new Map<string, Config>();
  /** Every `/ws/ingest` header, from every phone. */
  readonly headers: Record<string, unknown>[] = [];

  readonly #phones = new Map<string, WebSocketRoute>();
  readonly #reports = new Map<string, unknown>();
  readonly #lists = new Set<WebSocketRoute>();
  readonly #controls = new Set<{ readonly board: string; readonly ws: WebSocketRoute }>();

  /** Make `context`'s pages talk to this node. */
  async attach(context: BrowserContext): Promise<void> {
    await context.routeWebSocket(PHONE_SOCKET, (ws) => this.#phone(ws));
    await context.routeWebSocket(LIST_SOCKET, (ws) => this.#list(ws));
    await context.routeWebSocket(CONTROL_SOCKET, (ws) => this.#control(ws));
    await context.routeWebSocket('**/ws/ingest', (ws) => {
      ws.onMessage((message) => {
        if (typeof message === 'string') this.headers.push(JSON.parse(message) as Record<string, unknown>);
      });
    });
    await context.route('**/api/client-log', (route) => route.fulfill({ status: 204 }));
    await context.route('**/api/camera-links', (route) =>
      route.fulfill({ json: { links: [{ url: 'https://192.168.1.7:8443/camera', via: 'lan' }] } })
    );
  }

  /** The control UI set this board's config. */
  configure(board: string, patch: Partial<Config>): void {
    const config = { ...this.#config(board), ...patch };
    this.configs.set(board, config);
    this.#phoneOn(board)?.send(JSON.stringify({ type: 'config', config }));
    this.#tellControls(board);
  }

  /**
   * The node's about-once-a-second list push (the frame ages): the same cameras
   * again, so a page must not treat it as an answer to anything it asked.
   */
  tick(): void {
    this.#tellLists();
  }

  /** Stop and start: every socket drops, the pages reconnect to what was saved. */
  restart(): void {
    const sockets = [...this.#phones.values(), ...this.#lists, ...[...this.#controls].map((c) => c.ws)];
    this.#phones.clear();
    this.#lists.clear();
    this.#controls.clear();
    this.#reports.clear();
    for (const ws of sockets) void ws.close({ code: 1012, reason: 'node restarting' });
  }

  #config(board: string): Config {
    return this.configs.get(board) ?? DEFAULT_CONFIG;
  }

  #phoneOn(board: string): WebSocketRoute | undefined {
    return this.#phones.get(board);
  }

  /** The device ids and names no camera but `device` may be called. */
  #taken(device: string): Set<string> {
    return new Set([...this.names].filter(([d]) => d !== device).flat());
  }

  #defaultName(device: string): string {
    const taken = this.#taken(device);
    for (let n = 1; ; n++) if (!taken.has(`board-${n}`)) return `board-${n}`;
  }

  /** `null` when `device` may be called `name`, else the node's reason. */
  #refusal(device: string, name: string): string | null {
    const other = [...this.names].find(([d, n]) => d !== device && (d === name || n === name));
    return other === undefined ? null : `${name} is taken by camera ${other[1]}`;
  }

  #assigned(device: string): string {
    return JSON.stringify({ type: 'assigned', board_id: device, name: this.names.get(device) });
  }

  #phone(ws: WebSocketRoute): void {
    const device = decodeURIComponent(new URL(ws.url()).pathname.split('/').at(-1)!);
    const previous = this.#phones.get(device);
    if (previous !== undefined) {
      previous.send(JSON.stringify({ type: 'replaced' }));
      void previous.close({ code: 1000, reason: 'replaced' });
    }
    this.#phones.set(device, ws);
    ws.onClose(() => {
      if (this.#phones.get(device) !== ws) return;
      this.#phones.delete(device);
      this.#tellLists();
    });
    ws.onMessage((message) => {
      if (typeof message !== 'string' || this.#phones.get(device) !== ws) return;
      const parsed = JSON.parse(message) as { type: string; report: unknown };
      if (parsed.type === 'report') this.#reports.set(device, parsed.report);
      this.#tellLists();
    });

    if (!this.names.has(device)) this.names.set(device, this.#defaultName(device));
    ws.send(this.#assigned(device));
    ws.send(JSON.stringify({ type: 'config', config: this.#config(device) }));
    ws.send(JSON.stringify({ type: 'previews', on: false }));
    this.#tellLists();
  }

  #rename(ws: WebSocketRoute, device: string, name: string | null): void {
    const wanted = name ?? this.#defaultName(device);
    const refusal = this.#refusal(device, wanted);
    if (refusal !== null) {
      ws.send(JSON.stringify({ type: 'refused', reason: refusal }));
      return;
    }
    this.names.set(device, wanted);
    this.#phones.get(device)?.send(this.#assigned(device));
    this.#tellLists();
  }

  #forget(ws: WebSocketRoute, device: string): void {
    if (this.#phones.has(device)) {
      const reason = `camera ${this.names.get(device)} is connected: close its /camera page, then forget it`;
      ws.send(JSON.stringify({ type: 'refused', reason }));
      return;
    }
    this.configs.delete(device);
    this.names.delete(device);
    this.#reports.delete(device);
    this.#tellLists();
  }

  #list(ws: WebSocketRoute): void {
    this.#lists.add(ws);
    ws.onClose(() => this.#lists.delete(ws));
    ws.onMessage((message) => {
      const asked = JSON.parse(message as string) as
        | { type: 'rename'; device_id: string; name: string | null }
        | { type: 'forget'; device_id: string };
      if (asked.type === 'rename') this.#rename(ws, asked.device_id, asked.name);
      else this.#forget(ws, asked.device_id);
    });
    ws.send(this.#cameras());
  }

  #control(ws: WebSocketRoute): void {
    const board = decodeURIComponent(CONTROL_SOCKET.exec(new URL(ws.url()).pathname)![1]!);
    const control = { board, ws };
    this.#controls.add(control);
    ws.onClose(() => this.#controls.delete(control));
    ws.onMessage((message) => {
      const parsed = JSON.parse(message as string) as { type: string; config: Config };
      if (parsed.type === 'configure') this.configure(board, parsed.config);
    });
    ws.send(this.#state(board));
  }

  #cameras(): string {
    const cameras = [...this.names]
      .sort(([, a], [, b]) => a.localeCompare(b, 'en', { numeric: true }))
      .map(([device, name]) => ({
        device_id: device,
        name,
        connected: this.#phones.has(device),
        report: this.#reports.get(device) ?? null,
        frame_age_s: null
      }));
    return JSON.stringify({ type: 'cameras', cameras });
  }

  #state(board: string): string {
    return JSON.stringify({
      type: 'state',
      config: this.#config(board),
      report: this.#reports.get(board) ?? null,
      phone_connected: this.#phoneOn(board) !== undefined,
      autoframe: this.#config(board).auto ? 'idle' : 'manual',
      watching: false
    });
  }

  #tellLists(): void {
    const cameras = this.#cameras();
    for (const ws of this.#lists) ws.send(cameras);
  }

  #tellControls(board: string): void {
    for (const control of this.#controls) if (control.board === board) control.ws.send(this.#state(board));
  }
}
