/** The `/ws/*` upgrade relay, between a real `ws` client and a local `ws` stand-in node. */

import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { type RawData, WebSocket, WebSocketServer } from 'ws';

import { relayUpgrade, UPSTREAM_UNREACHABLE, wsOrigin } from './ws.ts';

const errors: string[] = [];
let node: WebSocketServer;
let viewer: http.Server;
let paths: string[];

/** A viewer whose upgrades go to `origin`. */
async function startViewer(origin: string): Promise<string> {
  viewer = http.createServer().on('upgrade', relayUpgrade(origin, (line) => errors.push(line)));
  await new Promise<void>((resolve) => viewer.listen(0, '127.0.0.1', resolve));
  return `ws://127.0.0.1:${(viewer.address() as AddressInfo).port}`;
}

beforeEach(async () => {
  paths = [];
  // Every socket echoes each message, prefixed `echo:`, with its frame type kept.
  node = new WebSocketServer({ port: 0, host: '127.0.0.1' });
  node.on('connection', (socket, req) => {
    paths.push(req.url ?? '');
    socket.on('message', (data: RawData, isBinary) =>
      socket.send(Buffer.concat([Buffer.from('echo:'), data as Buffer]), { binary: isBinary })
    );
  });
  await new Promise((resolve) => node.once('listening', resolve));
});
afterEach(async () => {
  for (const client of node.clients) client.terminate();
  await new Promise((resolve) => node.close(resolve));
  viewer.closeAllConnections();
  await new Promise((resolve) => viewer.close(resolve));
  errors.length = 0;
});

const nodeOrigin = (): string => `ws://127.0.0.1:${(node.address() as AddressInfo).port}`;

function open(url: string): Promise<WebSocket> {
  const ws = new WebSocket(url);
  return new Promise((resolve, reject) => ws.once('open', () => resolve(ws)).once('error', reject));
}

/** The next `n` messages: text as strings, binary as Buffers. */
function received(ws: WebSocket, n: number): Promise<Array<string | Buffer>> {
  const got: Array<string | Buffer> = [];
  return new Promise((resolve) =>
    ws.on('message', (data: Buffer, isBinary) => {
      got.push(isBinary ? data : data.toString());
      if (got.length === n) resolve(got);
    })
  );
}

const closed = (ws: WebSocket): Promise<[number, string]> =>
  new Promise((resolve) => ws.once('close', (code, reason) => resolve([code, reason.toString()])));

const echo = (bytes: Buffer): Buffer => Buffer.concat([Buffer.from('echo:'), bytes]);

describe('the /ws relay', () => {
  it.each(['/ws/ingest', '/ws/viewer', '/ws/boards/b%201/camera', '/ws/boards/b/control?x=1'])(
    'carries text and binary both ways, frame types kept, to the same path on the node: %s',
    async (path) => {
      const ws = await open((await startViewer(nodeOrigin())) + path);
      const replies = received(ws, 2);
      ws.send('{"type":"frame","seq":1}');
      ws.send(Buffer.from([0xff, 0xd8, 0xff, 0xe0]));
      expect(await replies).toEqual(['echo:{"type":"frame","seq":1}', echo(Buffer.from([0xff, 0xd8, 0xff, 0xe0]))]);
      expect(paths).toEqual([path.slice('/ws'.length)]);
      ws.close();
    }
  );

  it('keeps the order across frame types, so a header stays with its JPEG', async () => {
    const ws = await open(`${await startViewer(nodeOrigin())}/ws/ingest`);
    const replies = received(ws, 6);
    for (let seq = 0; seq < 3; seq++) {
      ws.send(`{"seq":${seq}}`);
      ws.send(Buffer.alloc(4, seq));
    }
    expect(await replies).toEqual([0, 1, 2].flatMap((seq) => [`echo:{"seq":${seq}}`, echo(Buffer.alloc(4, seq))]));
    ws.close();
  });

  it('passes a close from the node through to the page', async () => {
    node.on('connection', (socket) => socket.close(4001, 'board gone'));
    const ws = await open(`${await startViewer(nodeOrigin())}/ws/boards/b/camera`);
    expect(await closed(ws)).toEqual([4001, 'board gone']);
  });

  it("closes with 1011 and the node's address when the node is down", async () => {
    const ws = await open(`${await startViewer('ws://127.0.0.1:1')}/ws/ingest`);
    const [code, reason] = await closed(ws);
    expect(code).toBe(UPSTREAM_UNREACHABLE);
    expect(reason).toContain('ws://127.0.0.1:1/ingest');
    expect(errors).toHaveLength(1);
  });

  it('refuses a socket outside /ws', async () => {
    await expect(open(`${await startViewer(nodeOrigin())}/camera`)).rejects.toThrow('404');
  });

  it("takes only the origin of the node's /viewer URL", () => {
    expect(wsOrigin({ BR_INFERENCE_WS: 'wss://jetson.lan:8443/viewer' })).toBe('wss://jetson.lan:8443');
    expect(wsOrigin({})).toBe('ws://localhost:8000');
  });
});
