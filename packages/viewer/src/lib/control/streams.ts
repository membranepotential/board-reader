/**
 * The streams the node runs as cameras (`WS /streams`, DESIGN D22), as /control
 * shows them: what each one is doing, in words, and the message that removes one.
 * Removing a stream stops its process and forgets its camera, so its board id is
 * free again.
 */

import type { RemoveStreamMessage, StreamEntry } from '#lib/protocol/generated.ts';

export function removeStreamMessage(streamId: string): string {
  const message: RemoveStreamMessage = { type: 'remove_stream', stream_id: streamId };
  return JSON.stringify(message);
}

/** How a stream's state reads: `ok` streams, `wait` is on its way, `bad` needs a look. */
export interface StreamStatus {
  readonly text: string;
  readonly tone: 'ok' | 'wait' | 'bad';
}

export function streamStatus(stream: StreamEntry): StreamStatus {
  switch (stream.state) {
    case 'starting':
      return { text: 'starting…', tone: 'wait' };
    case 'running':
      return { text: `running as ${stream.board_id ?? 'a new camera'}`, tone: 'ok' };
    case 'crashed':
      return { text: `crashed, starting again: ${stream.error ?? 'no reason given'}`, tone: 'bad' };
    case 'stopped':
      return stream.error === null
        ? { text: 'stopped: the video ended', tone: 'wait' }
        : { text: `stopped: ${stream.error}`, tone: 'bad' };
  }
}

/** A start position in seconds as `h:mm:ss` (or `m:ss` under an hour). */
export function formatStart(seconds: number): string {
  const whole = Math.floor(seconds);
  const [h, m, s] = [Math.floor(whole / 3600), Math.floor((whole % 3600) / 60), whole % 60];
  const pad = (n: number): string => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}
