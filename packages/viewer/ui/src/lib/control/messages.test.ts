import { describe, expect, it } from 'vitest';
import { configureMessage, parseControlDown, parsePhoneDown, reportMessage } from './messages';
import { DEFAULT_CAMERA_CONFIG, type CameraReport } from '$lib/protocol';

const REPORT: CameraReport = {
  devices: [{ id: 'abc', label: 'Back Camera' }],
  device: 'Back Camera',
  frame: { width: 1920, height: 1080 },
  ingest: 'streaming'
};
const CONFIG = { crop: { x: 0.1, y: 0.2, w: 0.5, h: 0.6 }, device: 'Back Camera', resolution: '1080p', streaming: true };
const text = (message: unknown): string => JSON.stringify(message);

describe('the phone socket, down', () => {
  it('reads a config and the previews switch', () => {
    expect(parsePhoneDown(text({ type: 'config', config: CONFIG }))).toEqual({
      ok: true,
      value: { type: 'config', config: CONFIG }
    });
    expect(parsePhoneDown(text({ type: 'previews', on: true }))).toEqual({
      ok: true,
      value: { type: 'previews', on: true }
    });
  });

  it('refuses a config it cannot apply rather than applying half of one', () => {
    expect(parsePhoneDown(text({ type: 'config', config: { ...CONFIG, resolution: '8k' } })).ok).toBe(false);
    expect(parsePhoneDown(text({ type: 'state' })).ok).toBe(false); // the other socket's
    expect(parsePhoneDown('not json').ok).toBe(false);
  });
});

describe('the control socket, down', () => {
  it('reads the state the node holds', () => {
    const state = { config: CONFIG, report: REPORT, phone_connected: true };
    expect(parseControlDown(text({ type: 'state', ...state }))).toEqual({ ok: true, value: { type: 'state', state } });
    const fresh = { config: DEFAULT_CAMERA_CONFIG, report: null, phone_connected: false };
    expect(parseControlDown(text({ type: 'state', ...fresh })).ok).toBe(true);
  });

  it('refuses a state with a report it cannot read', () => {
    const report = { ...REPORT, ingest: 'streming' };
    expect(parseControlDown(text({ type: 'state', config: CONFIG, report, phone_connected: true })).ok).toBe(false);
    expect(parseControlDown(text({ type: 'state', config: CONFIG, report: null })).ok).toBe(false);
  });

  it('turns a pushed JPEG into an image source', () => {
    const parsed = parseControlDown(text({ type: 'preview', width: 480, height: 270, jpeg: '/9j/4AAQ' }));
    expect(parsed).toEqual({
      ok: true,
      value: { type: 'preview', picture: { width: 480, height: 270, src: 'data:image/jpeg;base64,/9j/4AAQ' } }
    });
    const frame = parseControlDown(text({ type: 'frame', seq: 41, width: 960, height: 648, jpeg: '/9j/' }));
    expect(frame.ok && frame.value.type === 'frame' && frame.value.frame.seq).toBe(41);
  });

  it('reads the health tick, with nothing yet streamed', () => {
    const health = { recognizer: 'chessq', fallback: null, received: 0, refused: 0, frame_age_s: null, fps: null };
    expect(parseControlDown(text({ type: 'health', ...health }))).toEqual({
      ok: true,
      value: { type: 'health', health }
    });
    expect(parseControlDown(text({ type: 'health', ...health, fps: 'fast' })).ok).toBe(false);
  });
});

describe('what goes up', () => {
  it('names its type, as board_reader_core.control parses it', () => {
    expect(JSON.parse(reportMessage(REPORT))).toEqual({ type: 'report', report: REPORT });
    expect(JSON.parse(configureMessage(DEFAULT_CAMERA_CONFIG))).toEqual({
      type: 'configure',
      config: DEFAULT_CAMERA_CONFIG
    });
  });
});
