import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fakeNetwork } from '#lib/fake-socket.ts';
import type { CameraEntry } from '#lib/protocol/generated.ts';
import { CamerasLink } from './cameras';

// The links derive their socket address from the page, as in a browser.
beforeEach(() => vi.stubGlobal('location', { href: 'https://viewer.test/control' }));
afterEach(() => vi.unstubAllGlobals());

const ENTRY: CameraEntry = {
  device_id: 'kitchen-phone-id',
  board_id: 'board-1',
  name: 'kitchen',
  connected: true,
  report: null,
  frame_age_s: 0.4
};

function link() {
  const net = fakeNetwork();
  const onMessage = vi.fn();
  const onLink = vi.fn();
  const cameras = new CamerasLink({ ...net.options, onMessage, onLink });
  cameras.start();
  return { net, cameras, onMessage, onLink };
}

describe('the camera list link', () => {
  it('hands over the list the node pushes, parsed', () => {
    const { net, onMessage } = link();
    expect(net.sockets[0]!.url).toBe('wss://viewer.test/ws/cameras');
    net.sockets[0]!.open();
    net.sockets[0]!.deliver({ type: 'cameras', cameras: [ENTRY] });
    net.sockets[0]!.deliver({ type: 'cameras', cameras: [{ ...ENTRY, board_id: 'a/b' }] }); // refused whole
    expect(onMessage).toHaveBeenCalledExactlyOnceWith({ type: 'cameras', cameras: [ENTRY] });
  });

  it('sends a rename and a forget only while connected', () => {
    const { net, cameras } = link();
    expect(cameras.rename('kitchen-phone-id', 'left')).toBe(false);
    expect(cameras.forget('kitchen-phone-id')).toBe(false);
    net.sockets[0]!.open();
    expect(cameras.rename('kitchen-phone-id', 'left')).toBe(true);
    expect(cameras.rename('kitchen-phone-id', null)).toBe(true);
    expect(cameras.forget('kitchen-phone-id')).toBe(true);
    expect(net.sockets[0]!.texts).toEqual([
      { type: 'rename', device_id: 'kitchen-phone-id', name: 'left' },
      { type: 'rename', device_id: 'kitchen-phone-id', name: null },
      { type: 'forget', device_id: 'kitchen-phone-id' }
    ]);
  });
});
