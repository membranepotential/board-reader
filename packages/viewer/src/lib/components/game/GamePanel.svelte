<script lang="ts">
  /**
   * One board's game, decoded from its reads as they arrive: the move list, the PGN, a
   * Copy button, a link that opens the game in Lichess analysis, and New game.
   *
   * The decoder (`#lib/game/decoder.ts`) is the application layer DESIGN D1 keeps out of
   * the node: it lives here, per board, and the positions store stays a store of the
   * latest event only. A move the decoder has not committed yet is shown dimmed; it is
   * still the best guess, so Copy and the link include it.
   */
  import { GameDecoder, observationOf, type DecodedGame } from '#lib/game/decoder.ts';
  import { gamePgn, lichessAnalysisUrl } from '#lib/game/pgn.ts';
  import type { PositionEvent } from '#lib/protocol/generated.ts';

  /** `board` is what people call the board; it goes in the PGN's `Site` tag. */
  let { event, board }: { event: PositionEvent; board: string } = $props();

  /** At this many squares of disagreement the board is likely not the game (a hand
   *  covers two or three). */
  const MISMATCH_NOTE = 6;

  // Not reactive: the page reads `game`, which every read replaces.
  let decoder = new GameDecoder();
  let game: DecodedGame = $state.raw(decoder.game);

  function feed(current: PositionEvent): void {
    const read = observationOf(current);
    if (read !== null) game = decoder.push(read);
  }

  // An effect, not a `$derived`: each read is folded into the decoder's history, which a
  // derivation must not mutate. The decoder ignores a read it has seen, so a rerun of
  // this effect changes nothing.
  $effect(() => feed(event));

  /** Starts again from the initial position with the next read, not the current one,
   *  which shows the board as it was while the pieces were being set up. */
  function newGame(): void {
    decoder = new GameDecoder();
    game = decoder.game;
    copied = null;
    copyFailed = false;
  }

  const pgn = $derived(gamePgn(game, board));
  const lichess = $derived(lichessAnalysisUrl(game));
  const provisional = $derived(game.moves.filter((move) => !move.committed).length);
  /** Keeps the move list's scroll position right: a list the user left at the bottom
   *  follows the game as rows arrive; one they scrolled up stays where they put it. */
  function followTail(el: HTMLElement): () => void {
    let following = true;
    const onScroll = () => {
      following = el.scrollTop + el.clientHeight >= el.scrollHeight - 4;
    };
    const rowsChanged = new MutationObserver(() => {
      if (following) el.scrollTop = el.scrollHeight;
    });
    el.addEventListener('scroll', onScroll, { passive: true });
    rowsChanged.observe(el, { childList: true, subtree: true, characterData: true });
    el.scrollTop = el.scrollHeight;
    return () => {
      rowsChanged.disconnect();
      el.removeEventListener('scroll', onScroll);
    };
  }

  /** Moves in pairs, as a score sheet: one row per move number. */
  const rows = $derived(
    Array.from({ length: Math.ceil(game.moves.length / 2) }, (_, i) => ({
      number: i + 1,
      plies: game.moves.slice(2 * i, 2 * i + 2)
    }))
  );

  /** The PGN last copied: the button says "Copied" until the game moves on. */
  let copied: string | null = $state(null);
  let copyFailed = $state(false);
  /** Whether the PGN expander is open: the user's toggle, opened for them when the
   *  clipboard refuses. Bound, never `open={...}`: a one-way attribute is set again on
   *  every update and closed the expander on each new read. */
  let pgnOpen = $state(false);

  /** The page may be plain http on the LAN, where `navigator.clipboard` is missing;
   *  `execCommand('copy')` on a selected textarea still works there. */
  function copyBySelection(text: string): boolean {
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.append(area);
    area.select();
    try {
      return document.execCommand('copy');
    } catch {
      return false;
    } finally {
      area.remove();
    }
  }

  async function copy(): Promise<void> {
    const text = pgn;
    let ok: boolean;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      ok = copyBySelection(text);
    }
    copied = ok ? text : null;
    copyFailed = !ok;
    if (!ok) pgnOpen = true;
  }
</script>

<section class="game-panel" aria-label="game">
  <h3>
    <span>game</span>
    <span class="count">
      {game.moves.length} plies{provisional > 0 ? ` · ${provisional} provisional` : ''}
    </span>
  </h3>

  {#if game.moves.length === 0}
    <p class="empty">No moves yet. The game starts from the initial position.</p>
  {:else}
    <ol class="moves" {@attach followTail}>
      {#each rows as row (row.number)}
        <li>
          <span class="number">{row.number}.</span>
          {#each row.plies as ply, i (i)}
            <span class="ply" class:provisional={!ply.committed}>{ply.san}</span>
          {/each}
        </li>
      {/each}
    </ol>
  {/if}

  {#if game.mismatch >= MISMATCH_NOTE}
    <p class="note">
      The board disagrees with this game on {game.mismatch} squares. If the game did not start
      from the initial position, set the board up and start a new game.
    </p>
  {/if}

  <div class="actions">
    <button type="button" class="copy" onclick={copy}>
      {copied === pgn ? 'Copied' : 'Copy PGN'}
    </button>
    <a class="lichess" href={lichess} target="_blank" rel="noopener noreferrer">
      Open in Lichess analysis
    </a>
    <button type="button" class="new-game" onclick={newGame}>New game</button>
  </div>
  {#if copyFailed}
    <p class="note">The browser refused the clipboard. Select the PGN below and copy it.</p>
  {/if}

  <details bind:open={pgnOpen}>
    <summary>PGN</summary>
    <pre class="pgn">{pgn}</pre>
  </details>
</section>

<style>
  /* Fills the board card's game drawer (GameDrawer.svelte), which is as tall as the
     board: the header, notes and buttons keep their height and only the move list
     scrolls, so the card never grows with the game. */
  .game-panel {
    box-sizing: border-box;
    height: 100%;
    display: flex;
    flex-direction: column;
    background: #161a20;
    border: 1px solid #272b33;
    border-radius: 6px;
    padding: 10px;
    font-size: 13px;
  }
  .game-panel > * {
    flex: none;
  }
  h3 {
    font-size: 14px;
    margin: 0 0 8px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .count {
    font-size: 12px;
    font-weight: normal;
    color: #9aa3ad;
  }
  .empty {
    color: #9aa3ad;
    margin: 0 0 8px;
  }
  .game-panel > .moves,
  .game-panel > .empty {
    flex: 1 1 auto;
    min-height: 0;
  }
  .moves {
    list-style: none;
    margin: 0 0 8px;
    padding: 0;
    overflow-y: auto;
    font-family: ui-monospace, monospace;
  }
  .moves li {
    display: grid;
    grid-template-columns: 2.5em 1fr 1fr;
    gap: 6px;
    padding: 1px 0;
  }
  .number {
    color: #6b7280;
  }
  .ply.provisional {
    color: #8a94a3;
    font-style: italic;
  }
  .note {
    color: #fc8;
    margin: 0 0 10px;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
  }
  button {
    font: inherit;
    background: #23303f;
    color: #cfe0f5;
    border: 1px solid #34465c;
    border-radius: 6px;
    padding: 4px 10px;
    cursor: pointer;
  }
  button:hover {
    background: #2b3b4f;
  }
  .lichess {
    color: #6aa8ff;
  }
  summary {
    cursor: pointer;
    color: #9aa3ad;
  }
  /* Its own scroll box, so opening the PGN does not push the buttons off the panel. */
  .pgn {
    max-height: 7.5em;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    margin: 6px 0 0;
    user-select: all;
  }
</style>
