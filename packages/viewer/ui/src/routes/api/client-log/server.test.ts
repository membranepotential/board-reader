/** `POST /api/client-log`, called directly. */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const env = vi.hoisted((): Record<string, string | undefined> => ({}));
vi.mock('$app/env/private', () => ({
  get BR_CLIENT_LOG_DIR() {
    return env['BR_CLIENT_LOG_DIR'];
  }
}));

import { POST } from './+server';

type Event = Parameters<typeof POST>[0];

async function post(body: string): Promise<{ kept: number }> {
  const request = new Request('http://viewer/api/client-log', { method: 'POST', body });
  const res = await POST({ request, getClientAddress: () => '192.168.1.20' } as unknown as Event);
  expect(res.status).toBe(200); // never an error, whatever arrives
  return res.json();
}

const notes = (n: number) => JSON.stringify({ notes: Array.from({ length: n }, (_, t) => ({ t, kind: 'tick' })) });

let dir: string;
beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'client-log-'));
  env['BR_CLIENT_LOG_DIR'] = path.join(dir, 'logs');
});
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

const written = (): Array<Record<string, unknown>> => {
  const [file, ...more] = fs.readdirSync(path.join(dir, 'logs'));
  expect(more).toEqual([]); // one file per server start
  expect(file).toMatch(/^\d{8}-\d{6}\.ndjson$/);
  return fs
    .readFileSync(path.join(dir, 'logs', file as string), 'utf8')
    .trimEnd()
    .split('\n')
    .map((line) => JSON.parse(line));
};

describe('the client log', () => {
  it('appends each note as NDJSON, stamped with its arrival and its client', async () => {
    const batch = { notes: [{ t: 1, kind: 'camera-stall-reopen' }, { t: 2, kind: 'pagehide' }] };
    expect(await post(JSON.stringify(batch))).toEqual({ kept: 2 });
    expect(await post(notes(1))).toEqual({ kept: 1 });
    const lines = written();
    expect(lines.map((line) => line['kind'])).toEqual(['camera-stall-reopen', 'pagehide', 'tick']);
    expect(lines.every((line) => typeof line['received_ts'] === 'number' && line['client'] === '192.168.1.20')).toBe(
      true
    );
  });

  it('accepts and drops everything when logging is off', async () => {
    env['BR_CLIENT_LOG_DIR'] = undefined;
    expect(await post(notes(1))).toEqual({ kept: 0 });
  });

  it('keeps nothing from garbage', async () => {
    expect(await post('not json')).toEqual({ kept: 0 });
    expect(await post(JSON.stringify({ notes: 'nope' }))).toEqual({ kept: 0 });
    expect(fs.existsSync(path.join(dir, 'logs'))).toBe(false);
  });

  it('drops an oversized body whole and truncates an oversized batch', async () => {
    expect(await post(JSON.stringify({ notes: [{ t: 1, pad: 'y'.repeat(200_000) }] }))).toEqual({ kept: 0 });
    expect(await post(notes(600))).toEqual({ kept: 500 });
  });
});
