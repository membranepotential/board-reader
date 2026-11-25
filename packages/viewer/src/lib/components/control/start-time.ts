/**
 * A video's start position as an operator types it: `90`, `1:30`, `1:02:03`, or
 * nothing (the beginning). Seconds may carry a fraction (`1:30.5`).
 */

export type StartTime =
  | { readonly ok: true; readonly seconds: number | null }
  | { readonly ok: false; readonly reason: string };

const PATTERN = /^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$/;

export function parseStartTime(text: string): StartTime {
  const typed = text.trim();
  if (typed === '') return { ok: true, seconds: null };
  const match = PATTERN.exec(typed);
  if (match === null) return { ok: false, reason: 'write the start as seconds, m:ss or h:mm:ss' };
  // With one colon the first group is minutes; with two it is hours.
  const [hours, minutes] = match[2] === undefined ? [0, Number(match[1] ?? 0)] : [Number(match[1]), Number(match[2])];
  const seconds = Number(match[3]);
  if ((match[1] !== undefined && seconds >= 60) || (match[2] !== undefined && minutes >= 60)) {
    return { ok: false, reason: 'minutes and seconds go up to 59' };
  }
  return { ok: true, seconds: hours * 3600 + minutes * 60 + seconds };
}
