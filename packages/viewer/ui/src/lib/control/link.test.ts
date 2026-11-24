import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fakeNetwork } from '$lib/fake-socket';
import { DEFAULT_CAMERA_CONFIG } from '$lib/protocol';
import { ControlLink } from './link';


// The links derive their socket address from the page, as in a browser.
beforeEach(() => vi.stubGlobal('location', { href: 'https://viewer.test/page' }));
afterEach(() => vi.unstubAllGlobals());

function link() {
  const net = fakeNetwork();
  const onMessage = vi.fn();
  const onLink = vi.fn();
  const control = new ControlLink({ boardId: 'phone', ...net.options, onMessage, onLink });
  control.start();
  return { net, control, onMessage, onLink };
}

describe('the control link', () => {
  it('hands over what the node pushes, parsed', () => {
    const { net, onMessage } = link();
    expect(net.sockets[0]!.url).toMatch(/\/ws\/boards\/phone\/control$/);
    net.sockets[0]!.open();
    net.sockets[0]!.deliver({ type: 'state', config: DEFAULT_CAMERA_CONFIG, report: null, phone_connected: false });
    net.sockets[0]!.deliver({ type: 'state', config: 'garbage' }); // dropped, not half-shown
    expect(onMessage).toHaveBeenCalledExactlyOnceWith({
      type: 'state',
      config: DEFAULT_CAMERA_CONFIG,
      report: null,
      phone_connected: false
    });
  });

  it('sends the whole config up, and says when it could not', () => {
    const { net, control } = link();
    const config = { ...DEFAULT_CAMERA_CONFIG, streaming: true };
    expect(control.configure(config)).toBe(false);
    net.sockets[0]!.open();
    expect(control.configure(config)).toBe(true);
    expect(net.sockets[0]!.texts).toEqual([{ type: 'configure', config }]);
  });

  it('reconnects after the node went away', () => {
    const { net, onLink } = link();
    net.sockets[0]!.open();
    net.sockets[0]!.close(1011, 'node down');
    expect(onLink).toHaveBeenLastCalledWith(false, 'node down');
    net.timers.shift()!();
    net.sockets[1]!.open();
    expect(onLink).toHaveBeenLastCalledWith(true, 'connected');
  });
});
