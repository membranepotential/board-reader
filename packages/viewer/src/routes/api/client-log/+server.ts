/**
 * `POST /api/client-log` — the phone's trial notes, kept on the viewer's own disk.
 *
 * The `/camera` page batches notes about itself (a stalled camera reopened, the page
 * hidden) and posts them here, so a failed phone session is diagnosable from the
 * machine the operator sits at. They are about the client, so they never go to the
 * node; this route is more specific than `/api/[...path]`, so it wins.
 *
 * Off unless `BR_CLIENT_LOG_DIR` is set (src/env.ts; `server/main.ts` defaults it
 * for a real server): then batches are accepted and dropped. The sink is reachable by anything on
 * the LAN, so the body, the notes per batch and the file are bounded; the page's own
 * batches are a few hundred bytes every few seconds. The answer is always
 * `200 {"kept": n}`, so logging never shows the page an error.
 */

import { BR_CLIENT_LOG_DIR } from '$app/env/private';
import fs from 'node:fs/promises';
import path from 'node:path';

import type { RequestHandler } from './$types';

export const prerender = false;

const MAX_BODY = 128 * 1024;
const MAX_NOTES = 500;
const MAX_FILE = 50 * 1024 * 1024;

/** One file per server start, named by its local start time: `20261124-090507.ndjson`. */
const STARTED = new Date();
const p = (n: number): string => String(n).padStart(2, '0');
const LOG_NAME =
  `${STARTED.getFullYear()}${p(STARTED.getMonth() + 1)}${p(STARTED.getDate())}-` +
  `${p(STARTED.getHours())}${p(STARTED.getMinutes())}${p(STARTED.getSeconds())}.ndjson`;

/** One note as the page sent it: a JSON object, opaque here. */
type Note = Readonly<Record<string, unknown>>;

const isNote = (value: unknown): value is Note =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** The notes in a body, at most `MAX_NOTES`; none for anything malformed. */
function parseNotes(body: string): Note[] {
  try {
    const payload: unknown = JSON.parse(body);
    const notes = isNote(payload) ? payload['notes'] : undefined;
    return Array.isArray(notes) ? notes.slice(0, MAX_NOTES).filter(isNote) : [];
  } catch {
    return [];
  }
}

const sizeOf = (file: string): Promise<number> =>
  fs.stat(file).then(
    (stat) => stat.size,
    () => 0
  );

export const POST: RequestHandler = async ({ request, getClientAddress }) => {
  const body = await request.arrayBuffer();
  const dir = BR_CLIENT_LOG_DIR;
  const notes = body.byteLength <= MAX_BODY ? parseNotes(new TextDecoder().decode(body)) : [];
  if (!dir || notes.length === 0) return Response.json({ kept: 0 });

  const file = path.join(dir, LOG_NAME);
  if ((await sizeOf(file)) >= MAX_FILE) return Response.json({ kept: 0 });
  const stamp = { received_ts: Date.now() / 1000, client: getClientAddress() };
  try {
    // Open-append-close per batch: no handle to outlive a reload, rotation or crash.
    await fs.mkdir(dir, { recursive: true });
    await fs.appendFile(file, notes.map((note) => `${JSON.stringify({ ...note, ...stamp })}\n`).join(''));
  } catch (error) {
    console.error(`client-log ${file}: ${(error as Error).message}`);
    return Response.json({ kept: 0 });
  }
  return Response.json({ kept: notes.length });
};
