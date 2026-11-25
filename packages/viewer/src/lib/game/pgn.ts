/**
 * A decoded game as PGN, and as a Lichess analysis link.
 *
 * chessops writes the PGN. Each move carries `{ [%emt h:mm:ss] }`, the time since the
 * move before (since the first read, for the first move), which Lichess reads on import
 * (lila `modules/study/src/main/CommentParser.scala`). The game has no clock, so there
 * is no `%clk`.
 */

import {
  ChildNode,
  defaultGame,
  defaultHeaders,
  emptyHeaders,
  makeComment,
  makeOutcome,
  makePgn,
  type Game,
  type Node,
  type PgnNodeData
} from 'chessops/pgn';

import type { DecodedGame } from './decoder.ts';

/** `YYYY.MM.DD` and `HH:MM:SS`, in UTC, as the PGN `UTCDate`/`UTCTime` tags spell them. */
function utcTags(at: number): { date: string; time: string } {
  const [date, time] = new Date(at).toISOString().split('T') as [string, string];
  return { date: date.replaceAll('-', '.'), time: time.slice(0, 8) };
}

function withMoves(game: Game<PgnNodeData>, moves: readonly PgnNodeData[]): Game<PgnNodeData> {
  moves.reduce<Node<PgnNodeData>>((node, data) => {
    const child = new ChildNode(data);
    node.children.push(child);
    return child;
  }, game.moves);
  return game;
}

/**
 * The whole PGN: the seven-tag roster, the UTC date and time, timed moves. `board` is
 * what people call the board (its camera's name, else its id) and goes in `Site`, so
 * games copied from two boards say which board each was played on.
 */
export function gamePgn(decoded: DecodedGame, board: string): string {
  const game = defaultGame<PgnNodeData>(defaultHeaders);
  game.headers.set('Site', board);
  game.headers.set('Result', makeOutcome(decoded.outcome));
  if (decoded.startedAt !== null) {
    const { date, time } = utcTags(decoded.startedAt);
    game.headers.set('Date', date);
    game.headers.set('UTCDate', date);
    game.headers.set('UTCTime', time);
  }
  let previous = decoded.startedAt;
  const moves = decoded.moves.map(({ san, at }) => {
    const emt = previous === null ? 0 : Math.max(0, Math.round((at - previous) / 1000));
    previous = at;
    return { san, comments: [makeComment({ emt })] };
  });
  return makePgn(withMoves(game, moves));
}

export const LICHESS_ANALYSIS = 'https://lichess.org/analysis/pgn/';

/**
 * The game opened in Lichess's analysis board.
 *
 * Lichess URL-decodes the path, turns `_` and `+` into spaces and parses the result
 * with chessops (lila `UserAnalysis.pgn`, `ui/analyse/src/pgnImport.ts`). That route
 * keeps the moves only, so the link carries the movetext alone, spaces as `_`. Lichess
 * reads the first 5000 characters of the path, about 700 plies.
 */
export function lichessAnalysisUrl(decoded: DecodedGame): string {
  const moves = decoded.moves.map(({ san }) => ({ san }));
  const movetext = makePgn(withMoves(defaultGame<PgnNodeData>(emptyHeaders), moves)).trim();
  return LICHESS_ANALYSIS + encodeURIComponent(movetext).replaceAll('%20', '_');
}
