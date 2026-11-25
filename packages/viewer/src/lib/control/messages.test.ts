import { describe, expect, it } from 'vitest';
import { configureMessage, jpegSrc, reportMessage } from './messages';
import { DEFAULT_CAMERA_CONFIG, parseWire } from '#lib/protocol/index.ts';
import { ControlDown, PhoneDown, type CameraReport } from '#lib/protocol/generated.ts';

const REPORT: CameraReport = {
  devices: [{ id: 'abc', label: 'Back Camera' }],
  device: 'Back Camera',
  frame: { width: 1920, height: 1080 },
  ingest: 'streaming',
  kind: 'phone'
};
const CONFIG = { crop: { x: 0.1, y: 0.2, w: 0.5, h: 0.6 }, device: 'Back Camera', resolution: '1080p', streaming: true, rules: 'standard', auto: false };
const text = (message: unknown): string => JSON.stringify(message);
const phoneDown = (raw: string) => parseWire(PhoneDown, raw);
const controlDown = (raw: string) => parseWire(ControlDown, raw);

describe('the phone socket, down', () => {
  it('reads a config and the previews switch', () => {
    expect(phoneDown(text({ type: 'config', config: CONFIG }))).toEqual({
      ok: true,
      value: { type: 'config', config: CONFIG }
    });
    expect(phoneDown(text({ type: 'previews', on: true }))).toEqual({
      ok: true,
      value: { type: 'previews', on: true }
    });
  });

  it('refuses a config it cannot apply rather than applying half of one', () => {
    expect(phoneDown(text({ type: 'config', config: { ...CONFIG, resolution: '8k' } })).ok).toBe(false);
    expect(phoneDown(text({ type: 'state' })).ok).toBe(false); // the other socket's
    expect(phoneDown('not json').ok).toBe(false);
  });
});

describe('the control socket, down', () => {
  it('reads the state the node holds', () => {
    const state = { type: 'state', config: CONFIG, report: REPORT, phone_connected: true, autoframe: 'manual' };
    expect(controlDown(text(state))).toEqual({ ok: true, value: state });
    const fresh = { config: DEFAULT_CAMERA_CONFIG, report: null, phone_connected: false, autoframe: 'idle' };
    expect(controlDown(text({ type: 'state', ...fresh })).ok).toBe(true);
  });

  it('refuses a state with a report it cannot read', () => {
    const report = { ...REPORT, ingest: 'streming' };
    expect(
      controlDown(text({ type: 'state', config: CONFIG, report, phone_connected: true, autoframe: 'idle' })).ok
    ).toBe(false);
    expect(controlDown(text({ type: 'state', config: CONFIG, report: null })).ok).toBe(false);
  });

  it('reads a pushed JPEG, and refuses one that is not base64', () => {
    const preview = { type: 'preview', width: 480, height: 270, jpeg: '/9j/4AAQ' };
    expect(controlDown(text(preview))).toEqual({ ok: true, value: preview });
    expect(jpegSrc(preview.jpeg)).toBe('data:image/jpeg;base64,/9j/4AAQ');
    const frame = controlDown(text({ type: 'frame', seq: 41, width: 960, height: 648, jpeg: '/9j/' }));
    expect(frame.ok && frame.value.type === 'frame' && frame.value.seq).toBe(41);
    expect(controlDown(text({ ...preview, jpeg: 'not base64!' })).ok).toBe(false);
  });

  it('reads the health tick, with nothing yet streamed', () => {
    const health = {
      type: 'health',
      recognizer: 'chessq',
      fallback: null,
      received: 0,
      refused: 0,
      frame_age_s: null,
      fps: null,
      direction: null
    };
    expect(controlDown(text(health))).toEqual({ ok: true, value: health });
    expect(controlDown(text({ ...health, fps: 'fast' })).ok).toBe(false);
  });

  it('reads which way the board lies, and refuses a side it does not know', () => {
    const health = { type: 'health', recognizer: 'chessq', fallback: null, received: 9, refused: 0, frame_age_s: 0.4, fps: 3 };
    const direction = { degrees: 89.5, white_side: 'left', camera_moved: true };
    expect(controlDown(text({ ...health, direction }))).toEqual({ ok: true, value: { ...health, direction } });
    const sideways = { ...direction, white_side: 'east' };
    expect(controlDown(text({ ...health, direction: sideways })).ok).toBe(false);
    expect(controlDown(text(health)).ok).toBe(false); // direction is required
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
