import { describe, expect, it } from 'vitest';
import { formatStart, removeStreamMessage, streamStatus } from './streams';
import { parseWire } from '#lib/protocol/index.ts';
import { StreamsDown, StreamsUp, type StreamEntry } from '#lib/protocol/generated.ts';

const STREAM: StreamEntry = {
  id: 'stream-1',
  url: 'https://www.youtube.com/watch?v=some-video',
  start_s: 3600,
  state: 'running',
  device_id: 'kitchen-stream-id',
  board_id: 'board-3',
  error: null
};

describe('the streams socket', () => {
  it('sends a removal the node takes', () => {
    expect(parseWire(StreamsUp, removeStreamMessage('stream-2'))).toEqual({
      ok: true,
      value: { type: 'remove_stream', stream_id: 'stream-2' }
    });
  });

  it('reads the list the node pushes', () => {
    const list = { type: 'streams', streams: [STREAM] };
    expect(parseWire(StreamsDown, JSON.stringify(list))).toEqual({ ok: true, value: list });
    const odd = { type: 'streams', streams: [{ ...STREAM, state: 'paused' }] };
    expect(parseWire(StreamsDown, JSON.stringify(odd)).ok).toBe(false);
  });
});

describe('a stream, in words', () => {
  it('names the board a running stream streams as', () => {
    expect(streamStatus(STREAM)).toEqual({ text: 'running as board-3', tone: 'ok' });
    expect(streamStatus({ ...STREAM, state: 'starting', board_id: null }).tone).toBe('wait');
  });

  it('says why a stream crashed or stopped', () => {
    const crashed = { ...STREAM, state: 'crashed', error: 'exited with code 1: 403' } as const;
    expect(streamStatus(crashed)).toEqual({ text: 'crashed, starting again: exited with code 1: 403', tone: 'bad' });
    expect(streamStatus({ ...STREAM, state: 'stopped' }).text).toBe('stopped: the video ended');
    expect(streamStatus({ ...STREAM, state: 'stopped', error: 'no edge here' })).toEqual({
      text: 'stopped: no edge here',
      tone: 'bad'
    });
  });
});

describe('formatStart', () => {
  it('writes a start as h:mm:ss, or m:ss under an hour', () => {
    expect(formatStart(3600)).toBe('1:00:00');
    expect(formatStart(3725.5)).toBe('1:02:05');
    expect(formatStart(75)).toBe('1:15');
    expect(formatStart(0)).toBe('0:00');
  });
});
