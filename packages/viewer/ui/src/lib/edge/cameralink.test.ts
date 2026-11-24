import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fakeNetwork } from '../fake-socket';
import { DEFAULT_CAMERA_CONFIG, type CameraReport } from '../protocol';
import { CameraLink } from './cameralink';

const REPORT: CameraReport = { devices: [], device: null, frame: null, ingest: 'idle' };
const LIVE: CameraReport = {
  devices: [{ id: 'a', label: 'Back Camera' }],
  device: 'Back Camera',
  frame: { width: 1280, height: 720 },
  ingest: 'idle'
};


// The links derive their socket address from the page, as in a browser.
beforeEach(() => vi.stubGlobal('location', { href: 'https://viewer.test/page' }));
afterEach(() => vi.unstubAllGlobals());

function link() {
  const net = fakeNetwork();
  const onConfig = vi.fn();
  const onPreviews = vi.fn();
  const onLink = vi.fn();
  const camera = new CameraLink({ boardId: 'left board', ...net.options, onConfig, onPreviews, onLink });
  return { net, camera, onConfig, onPreviews, onLink };
}

describe('the phone link', () => {
  it('connects to its board and sends the report on connect', () => {
    const { net, camera } = link();
    camera.report(REPORT); // before the socket opened: kept, not lost
    camera.start();
    expect(net.sockets[0]!.url).toMatch(/\/ws\/boards\/left%20board\/camera$/);
    net.sockets[0]!.open();
    expect(net.sockets[0]!.texts).toEqual([{ type: 'report', report: REPORT }]);
  });

  it('sends a report only when it changed', () => {
    const { net, camera } = link();
    camera.start();
    net.sockets[0]!.open();
    camera.report(REPORT);
    camera.report({ ...REPORT });
    camera.report(LIVE);
    expect(net.sockets[0]!.texts).toEqual([
      { type: 'report', report: REPORT },
      { type: 'report', report: LIVE }
    ]);
  });

  it('applies the config the node pushes', () => {
    const { net, camera, onConfig } = link();
    camera.start();
    net.sockets[0]!.open();
    const config = { ...DEFAULT_CAMERA_CONFIG, streaming: true };
    net.sockets[0]!.deliver({ type: 'config', config });
    expect(onConfig).toHaveBeenCalledExactlyOnceWith(config);
  });

  it('sends previews only while the node wants them', () => {
    const { net, camera, onPreviews } = link();
    camera.start();
    net.sockets[0]!.open();
    const jpeg = new Blob(['jpeg']);
    expect(camera.sendPreview(jpeg)).toBe(false);
    net.sockets[0]!.deliver({ type: 'previews', on: true });
    expect(onPreviews).toHaveBeenLastCalledWith(true);
    expect(camera.sendPreview(jpeg)).toBe(true);
    // The last one is still in the send buffer: the next would only arrive staler.
    net.sockets[0]!.bufferedAmount = 1;
    expect(camera.sendPreview(jpeg)).toBe(false);
    net.sockets[0]!.bufferedAmount = 0;
    net.sockets[0]!.deliver({ type: 'previews', on: false });
    expect(camera.sendPreview(jpeg)).toBe(false);
    expect(net.sockets[0]!.sent.filter((data) => data instanceof Blob)).toHaveLength(1);
  });

  it('keeps its config across a lost link and reports again on reconnect', () => {
    const { net, camera, onConfig, onPreviews, onLink } = link();
    camera.start();
    net.sockets[0]!.open();
    camera.report(LIVE);
    net.sockets[0]!.deliver({ type: 'previews', on: true });
    net.sockets[0]!.close(1011, 'node down');

    expect(onLink).toHaveBeenLastCalledWith(false, 'node down');
    expect(onPreviews).toHaveBeenLastCalledWith(false); // nobody is known to watch now
    expect(onConfig).not.toHaveBeenCalled(); // nothing reset the config

    net.timers.shift()!();
    net.sockets[1]!.open();
    expect(onLink).toHaveBeenLastCalledWith(true, 'connected');
    // A fresh socket hears the report again, though it did not change.
    expect(net.sockets[1]!.texts).toEqual([{ type: 'report', report: LIVE }]);
  });
});
