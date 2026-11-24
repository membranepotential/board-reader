/**
 * Run the viewer's PGN decoder over the gate takes and compare it with their truth.
 *
 *     node tools/pgn_eval.ts                      # the seven gate takes, replay guard-only
 *     node tools/pgn_eval.ts --replay cq-v6 20260827-175539
 *     node tools/pgn_eval.ts --param margin=20 --param temperature=3
 *
 * Feeds `recordings/<take>/replays/<replay>/events.ndjson` (the events the node
 * published on that replay, in order) to `GameDecoder`, the same module the live page
 * runs, and derives the true move list from `truth.json`: each segment's placement is
 * reached from the one before by one legal ply, a takeback, or two plies. Per take it
 * prints the plies right before the first divergence, where that divergence is, how many
 * moves are committed, and the latency from each true move's segment start (the frame
 * before the piece is lifted) to the read that committed it.
 *
 * Recorded events keep the top two classes per square (`evidence`), not the whole
 * `square_dist`, so each square's distribution is rebuilt from those two with the rest
 * spread evenly over the other eleven classes.
 *
 * The gate takes come from `tools/gate_table.py --replay-args`, the one list of them.
 */

import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { join, resolve } from 'node:path';

// The viewer's modules import extensionless paths (`./generated`), which Vite resolves
// and Node does not; this probes `.ts` for them, so the harness runs the shipped code.
registerHooks({
  resolve(specifier, context, nextResolve) {
    try {
      return nextResolve(specifier, context);
    } catch (error) {
      if (!specifier.startsWith('.')) throw error;
      return nextResolve(`${specifier}.ts`, context);
    }
  }
});

const ROOT = resolve(import.meta.dirname, '..');
const VIEWER = join(ROOT, 'packages/viewer');
const { GameDecoder, DEFAULT_PARAMS } = await import(join(VIEWER, 'src/lib/game/decoder.ts'));
const { PositionEvent } = await import(join(VIEWER, 'src/lib/protocol/generated.ts'));
const { CLASSES } = await import(join(VIEWER, 'src/lib/protocol/index.ts'));
const { Chess } = await import(join(VIEWER, 'node_modules/chessops/dist/esm/chess.js'));
const { makeBoardFen } = await import(join(VIEWER, 'node_modules/chessops/dist/esm/fen.js'));
const { makeSan } = await import(join(VIEWER, 'node_modules/chessops/dist/esm/san.js'));

type DecoderModule = typeof import('../packages/viewer/src/lib/game/decoder.ts');
type Observation = import('../packages/viewer/src/lib/game/decoder.ts').Observation;
type DecodedGame = import('../packages/viewer/src/lib/game/decoder.ts').DecodedGame;
type DecoderParams = import('../packages/viewer/src/lib/game/decoder.ts').DecoderParams;
type Position = import('chessops/chess').Chess;

// ---- the recorded events ------------------------------------------------------

type Evidence = readonly (readonly [string, number])[];

interface RecordedLine {
  readonly utc: string;
  readonly changed_utc?: string;
  readonly status: string;
  readonly placement: string;
  readonly pending?: readonly string[];
  readonly settled?: { readonly placement: string };
  readonly evidence: readonly Evidence[];
}

/** The two named classes, the rest spread evenly over the other eleven. */
function distribution(evidence: readonly Evidence[]): Observation['dist'] {
  const rows = Array.from({ length: 8 }, (_, row) =>
    Array.from({ length: 8 }, (_, col) => {
      const top = evidence[row * 8 + col]!;
      const named = top.reduce((sum, [, p]) => sum + p, 0);
      const rest = Math.max(0, 1 - named) / (CLASSES.length - top.length);
      return CLASSES.map((name: string) => top.find(([c]) => c === name)?.[1] ?? rest);
    })
  );
  return PositionEvent.shape.square_dist.parse(rows);
}

/**
 * One line as a read. Three generations of lines exist (`.claude/rules/recording.md`):
 * since 2026-11-24 `placement` is the settled board (a `pending` key marks the line);
 * before, the settled board sat in a `settled` block. Only the newest lines carry
 * `changed_utc`; for the others the read's own time stands in.
 */
function observation(line: RecordedLine): Observation | null {
  if (line.status !== 'ok') return null;
  const settled = 'pending' in line ? line.placement : (line.settled?.placement ?? line.placement);
  return {
    dist: distribution(line.evidence),
    at: Date.parse(line.utc),
    settled: { placement: settled, since: Date.parse(line.changed_utc ?? line.utc) }
  };
}

function ndjson<T>(path: string): T[] {
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as T);
}

// ---- the truth ----------------------------------------------------------------

interface Truth {
  readonly segments: readonly { readonly from_seq: number; readonly placement: string }[];
}

interface TrueMove {
  readonly san: string;
  /** Capture time of the segment's first frame, ms. */
  readonly at: number;
}

function moves(position: Position): { san: string; next: Position }[] {
  const out: { san: string; next: Position }[] = [];
  for (const [from, dests] of position.allDests()) {
    for (const to of dests) {
      const pawn = position.board.get(from)?.role === 'pawn';
      const last = Math.floor(to / 8) === 0 || Math.floor(to / 8) === 7;
      for (const promotion of pawn && last ? (['queen', 'knight', 'rook', 'bishop'] as const) : [undefined]) {
        const move = promotion === undefined ? { from, to } : { from, to, promotion };
        const next = position.clone();
        const san = makeSan(next, move);
        next.play(move);
        out.push({ san, next });
      }
    }
  }
  return out;
}

/** The true game: one ply per segment, or a takeback, or (if the truth merged two) two plies. */
function trueGame(truth: Truth, frameTimes: Map<number, number>): { moves: TrueMove[]; notes: string[] } {
  const stack: { position: Position; move: TrueMove | null }[] = [{ position: Chess.default(), move: null }];
  const notes: string[] = [];
  const top = () => stack[stack.length - 1]!.position;
  if (truth.segments[0]?.placement !== makeBoardFen(top().board)) notes.push('truth does not start from the initial position');
  for (const segment of truth.segments.slice(1)) {
    const at = frameTimes.get(segment.from_seq) ?? NaN;
    const back = stack[stack.length - 2];
    if (back !== undefined && makeBoardFen(back.position.board) === segment.placement) {
      notes.push(`takeback of ${stack.pop()!.move!.san} at seq ${segment.from_seq}`);
      continue;
    }
    const one = moves(top()).find(({ next }) => makeBoardFen(next.board) === segment.placement);
    if (one !== undefined) {
      stack.push({ position: one.next, move: { san: one.san, at } });
      continue;
    }
    const two = moves(top())
      .flatMap((first) => moves(first.next).map((second) => ({ first, second })))
      .find(({ second }) => makeBoardFen(second.next.board) === segment.placement);
    if (two === undefined) {
      notes.push(`segment at seq ${segment.from_seq} is not reachable in two plies; truth stops there`);
      break;
    }
    notes.push(`segment at seq ${segment.from_seq} holds two plies`);
    stack.push({ position: two.first.next, move: { san: two.first.san, at } });
    stack.push({ position: two.second.next, move: { san: two.second.san, at } });
  }
  return { moves: stack.flatMap(({ move }) => (move === null ? [] : [move])), notes };
}

// ---- one take -----------------------------------------------------------------

interface TakeResult {
  readonly take: string;
  readonly reads: number;
  readonly truth: readonly TrueMove[];
  readonly decoded: DecodedGame;
  /** Plies equal to the truth before the first difference. */
  readonly correct: number;
  readonly committed: number;
  /** Per correctly committed true ply: seconds from its segment start to its commit. */
  readonly latencies: readonly number[];
  readonly notes: readonly string[];
}

function runTake(take: string, replay: string, params: DecoderParams, Decoder: DecoderModule['GameDecoder']): TakeResult {
  const dir = join(ROOT, 'recordings', take);
  const replayDir = join(dir, 'replays', replay);
  const frames = ndjson<{ type: string; seq: number; capture_ts: number }>(join(replayDir, 'frames.ndjson'));
  const frameTimes = new Map(frames.filter((f) => f.type === 'frame').map((f) => [f.seq, f.capture_ts * 1000]));
  const { moves: truth, notes } = trueGame(JSON.parse(readFileSync(join(dir, 'truth.json'), 'utf8')) as Truth, frameTimes);

  const decoder = new Decoder(params);
  const committedAt: (number | undefined)[] = [];
  let reads = 0;
  for (const line of ndjson<RecordedLine>(join(replayDir, 'events.ndjson'))) {
    const read = observation(line);
    if (read === null) continue;
    reads += 1;
    const game = decoder.push(read);
    // A ply counts as committed once it and every ply before it are committed and true.
    for (let i = 0; i < game.moves.length && i < truth.length; i++) {
      const move = game.moves[i]!;
      if (!move.committed || move.san !== truth[i]!.san) break;
      committedAt[i] ??= read.at;
    }
  }
  const decoded = decoder.game;
  let correct = 0;
  while (correct < truth.length && correct < decoded.moves.length && decoded.moves[correct]!.san === truth[correct]!.san) {
    correct += 1;
  }
  const latencies = truth
    .slice(0, correct)
    .flatMap((move, i) => (committedAt[i] === undefined ? [] : [(committedAt[i]! - move.at) / 1000]));
  return {
    take,
    reads,
    truth,
    decoded,
    correct,
    committed: decoded.moves.filter((m) => m.committed).length,
    latencies,
    notes
  };
}

// ---- the report ---------------------------------------------------------------

function sanList(sans: readonly string[]): string {
  return sans.map((san, i) => (i % 2 === 0 ? `${i / 2 + 1}.${san}` : san)).join(' ');
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function report(result: TakeResult): string {
  const { take, reads, truth, decoded, correct, committed, latencies, notes } = result;
  const divergence =
    correct === truth.length && correct === decoded.moves.length
      ? 'none'
      : `ply ${correct + 1}: truth ${truth[correct]?.san ?? '(end)'}, decoded ${decoded.moves[correct]?.san ?? '(end)'}`;
  const latency = latencies.length === 0 ? '-' : `${median(latencies).toFixed(1)} / ${Math.max(...latencies).toFixed(1)}`;
  return [
    `| ${take} | ${reads} | ${correct}/${truth.length} | ${decoded.moves.length} | ${committed} | ${divergence} | ${latency} |`,
    `    truth:   ${sanList(truth.map((m) => m.san))}`,
    `    decoded: ${sanList(decoded.moves.map((m) => m.san))}`,
    ...notes.map((note) => `    note: ${note}`)
  ].join('\n');
}

function parseArgs(argv: readonly string[]): { replay: string; takes: string[]; params: DecoderParams } {
  let replay = 'guard-only';
  const takes: string[] = [];
  const params: Record<string, number> = { ...DEFAULT_PARAMS };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === '--replay') replay = argv[++i]!;
    else if (arg === '--param') {
      const [key, value] = argv[++i]!.split('=') as [string, string];
      if (!(key in params)) throw new Error(`unknown decoder parameter ${key}`);
      params[key] = Number(value);
    } else takes.push(arg);
  }
  if (takes.length === 0) {
    const listed = execFileSync('uv', ['run', 'python', 'tools/gate_table.py', '--replay-args'], { cwd: ROOT, encoding: 'utf8' });
    takes.push(...listed.split('\n').filter(Boolean).map((line) => line.split(' ')[0]!));
  }
  return { replay, takes, params: params as unknown as DecoderParams };
}

const { replay, takes, params } = parseArgs(process.argv.slice(2));
console.log(`replay ${replay}; params ${JSON.stringify(params)}\n`);
console.log('| take | reads | correct/true plies | decoded | committed | first divergence | latency s (median / max) |');
console.log('|---|---|---|---|---|---|---|');
const results = takes.map((take) => runTake(take, replay, params, GameDecoder));
for (const result of results) console.log(report(result));
const correct = results.reduce((sum, r) => sum + r.correct, 0);
const total = results.reduce((sum, r) => sum + r.truth.length, 0);
const perfect = results.filter((r) => r.correct === r.truth.length && r.decoded.moves.length === r.truth.length).length;
const all = results.flatMap((r) => r.latencies);
console.log(
  `\ntotal: ${correct}/${total} plies before the first divergence, ${perfect}/${results.length} games exact; ` +
    `commit latency median ${all.length ? median(all).toFixed(1) : '-'} s, max ${all.length ? Math.max(...all).toFixed(1) : '-'} s`
);
