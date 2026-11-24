import { describe, expect, it, vi } from 'vitest';
import { fakeNetwork } from './fake-socket';
import { ReconnectingSocket } from './socket';

function socket() {
  const net = fakeNetwork();
  const handlers = { onOpen: vi.fn(), onText: vi.fn(), onLost: vi.fn() };
  const link = new ReconnectingSocket(
    { url: 'ws://test/ws/x', reconnectMinMs: 100, reconnectMaxMs: 400, ...net.options },
    handlers
  );
  return { net, handlers, link };
}

describe('ReconnectingSocket', () => {
  it('reconnects after a loss, and says why it was lost', () => {
    const { net, handlers, link } = socket();
    link.start();
    net.sockets[0]!.open();
    net.sockets[0]!.close(1011, 'ws://node:8000/x: refused');
    expect(handlers.onLost).toHaveBeenCalledWith('ws://node:8000/x: refused');
    expect(link.open).toBe(false);
    net.timers.shift()!();
    expect(net.sockets).toHaveLength(2);
    net.sockets[1]!.open();
    expect(handlers.onOpen).toHaveBeenCalledTimes(2);
  });

  it('does not reconnect once stopped, and a stale socket cannot take over', () => {
    const { net, handlers, link } = socket();
    link.start();
    const first = net.sockets[0]!;
    link.stop();
    link.start(); // a stop→start inside the first socket's close handshake
    first.onclose?.({ code: 1000, reason: '' } as CloseEvent);
    expect(handlers.onLost).not.toHaveBeenCalled();
    net.sockets[1]!.open();
    expect(link.open).toBe(true);
    link.stop();
    for (const timer of net.timers) timer();
    expect(net.sockets).toHaveLength(2);
  });

  it('sends only on an open socket and queues nothing', () => {
    const { net, link } = socket();
    link.start();
    expect(link.send('early')).toBe(false);
    net.sockets[0]!.open();
    expect(link.send('now')).toBe(true);
    expect(net.sockets[0]!.sent).toEqual(['now']);
  });

  it('hands over text messages only', () => {
    const { net, handlers, link } = socket();
    link.start();
    net.sockets[0]!.open();
    net.sockets[0]!.onmessage?.({ data: new ArrayBuffer(2) } as MessageEvent);
    net.sockets[0]!.deliver({ type: 'x' });
    expect(handlers.onText).toHaveBeenCalledExactlyOnceWith('{"type":"x"}');
  });
});
