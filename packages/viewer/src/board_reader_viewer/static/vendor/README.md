# Vendored: chessground

Board UI for the `/snapshot` result view. Vendored locally (no CDN) so the
viewer works on a LAN with no internet access.

- **Package:** `chessground`
- **Version:** 9.2.1
- **Published:** 2025-05-15 (well before the 7-day supply-chain seasoning cutoff —
  see `[tool.uv] exclude-newer` in the repo's `pyproject.toml` files for the
  seasoning policy; this JS dep was checked against the same principle by hand)
- **Source:** https://registry.npmjs.org/chessground/-/chessground-9.2.1.tgz
  (official npm registry tarball, sha1 `0c11a80d8f41e827e73bb9bbe1ba542e84753de0`,
  verified against the registry's published `dist.shasum`)
- **Upstream:** https://github.com/lichess-org/chessground

## Files

| File | From tarball | Notes |
|---|---|---|
| `chessground.min.js` | `dist/chessground.min.js` | ESM bundle — load with `<script type="module">` and `import { Chessground } from ...`. There is **no** UMD/global build; `package.json` sets `"type": "module"`. |
| `chessground.base.css` | `assets/chessground.base.css` | Layout/structure (required). |
| `chessground.brown.css` | `assets/chessground.brown.css` | Board square theme. Squares are an embedded base64 SVG data URI — no separate image files. |
| `chessground.cburnett.css` | `assets/chessground.cburnett.css` | Piece set. All 12 piece glyphs are embedded base64 SVG data URIs directly in the CSS (`background-image: url('data:image/svg+xml;base64,...')`) — **no external SVG files to vendor**, nothing else references relative paths. |
| `LICENSE.chessground` | `LICENSE` | Chessground's own GPL-3.0 license text. |

## Licenses

- **Chessground itself:** GPL-3.0-or-later (per its `package.json` `license`
  field and `README.md`: "Chessground is distributed under the GPL-3.0 license
  (or any later version, at your option). When you use Chessground for your
  website, your combined work may be distributed only under the GPL.").
- **cburnett piece set** (bundled in `chessground.cburnett.css`): **GPLv2+**,
  authored by Colin M.L. Burnett — confirmed against
  [lichess-org/lila's `COPYING.md`](https://github.com/lichess-org/lila/blob/master/COPYING.md),
  which lists `public/piece/cburnett | Colin M.L. Burnett | GPLv2+`. (Note:
  this piece set is *also* separately available on Wikimedia Commons under
  CC-BY-SA 3.0 from the same author, but the copy shipped inside this npm
  package is under the GPLv2+ grant recorded by lila — that's the license
  that applies to these vendored files.)

Combined effect: this board-reader repo already ships under its own license;
serving these assets to the viewer's browser clients falls under GPL-3.0's
"conveying" provisions the same way any GPL'd JS served to a browser does — see
chessground's own README pointer to https://greendrake.info/publications/js-gpl
for the reasoning lichess relies on.
