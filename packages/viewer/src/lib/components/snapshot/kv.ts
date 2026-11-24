/**
 * The key/value evidence row — the workhorse of every stage body.
 *
 * The hand-written page built these tables by joining `<tr><th>…` strings into
 * `innerHTML`, which is why it also needed an `esc()` helper for every value that came
 * from the node (upload ids, stored placements, prior constraints). Rows are values
 * here and Svelte does the escaping, so that helper — and the whole class of bug it
 * defended against — is gone from this page.
 */

/** How a value reads: neutral, good, worth a look, or wrong. */
export type Tone = 'plain' | 'ok' | 'warn' | 'bad';

export interface KvRow {
  readonly key: string;
  readonly value: string;
  readonly tone: Tone;
}

export function kv(key: string, value: string, tone: Tone = 'plain'): KvRow {
  return { key, value, tone };
}

/** `kv` with the tone chosen by a predicate — the shape most stage tables want. */
export function kvIf(key: string, value: string, warn: boolean): KvRow {
  return { key, value, tone: warn ? 'warn' : 'ok' };
}
