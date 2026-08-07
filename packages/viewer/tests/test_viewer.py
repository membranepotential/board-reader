"""Tests for the viewer static server."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from board_reader_viewer.server import _STATIC, create_app


def test_healthz():
    assert TestClient(create_app()).get("/healthz").json() == {"status": "ok"}


def test_index_served_and_ws_url_injected(monkeypatch):
    monkeypatch.setenv("BR_INFERENCE_WS", "ws://example.test:9000/viewer")
    # Re-import not needed: the env is read per-request inside the route.
    html = TestClient(create_app()).get("/").text
    assert "ws://example.test:9000/viewer" in html
    assert "__BR_INFERENCE_WS__" not in html  # placeholder fully substituted


def test_index_default_ws_url(monkeypatch):
    monkeypatch.delenv("BR_INFERENCE_WS", raising=False)
    html = TestClient(create_app()).get("/").text
    assert "ws://localhost:8000/viewer" in html


def test_index_page_shows_per_board_freshness(monkeypatch):
    """A live stream has to *look* live: every card carries the age of its last event,
    ticking, and marks itself stale once nothing has arrived for 5s."""
    html = TestClient(create_app()).get("/").text
    assert "const STALE_MS = 5000;" in html
    assert 'class="age"' in html
    assert "function fmtAge" in html
    assert "const stale = age > STALE_MS;" in html
    assert 'rec.card.classList.toggle("stale", stale);' in html
    assert ".board-card.stale .stale-chip { display: inline-block; }" in html
    assert ".board-card.stale .board, .board-card.stale .meta > :not(.live) { filter:" in html
    # How old it is stays legible on a stale card, and a stream that dies mid-move
    # must not pulse "moving" for ever.
    assert ".board-card.stale .motion-chip { opacity: 0.45; animation: none; }" in html
    # Freshness is arrival time at the viewer, not the event's capture stamp — a skewed
    # edge clock must not make a live stream look stale.
    assert "lastAt: performance.now()," in html
    # Recovery is immediate: an event re-runs the same update rather than waiting a
    # tick — and it runs before the calibrate button is re-evaluated, so the button
    # does not stay disabled-as-stale for a tick after the stream comes back.
    render_fn = html.split("function render(ev) {")[1].split("\nsetInterval(")[0]
    assert render_fn.index("freshen(rec, now);") < render_fn.index("updateCalibState(rec);")


def test_index_page_shows_per_board_event_rate():
    """~1 ev/s at rest vs a 10-15 ev/s burst is the whole point of the adaptive
    forwarder (DESIGN D4) — the card states it as a rolling figure."""
    html = TestClient(create_app()).get("/").text
    assert "const RATE_WINDOW_MS = 3000;" in html
    assert 'class="eps"' in html
    assert "while (rec.stamps.length && now - rec.stamps[0] > RATE_WINDOW_MS) rec.stamps.shift();" in html
    # (n-1)/span, not n/window: with a 1 fps keepalive the window holds 4 stamps
    # spanning 3s — that is 1.0/s, and n/window would claim 1.3/s.
    assert "const eps = span > 0 ? (n - 1) / span : n / (RATE_WINDOW_MS / 1000);" in html
    assert 'eps.toFixed(1) + " ev/s"' in html


def test_index_page_shows_motion_state_from_stability():
    """DESIGN D3: stability is 1.0 at rest. Below 0.7 the board reads as in motion —
    a chip, plus the existing bar, and never a crash when the field is absent."""
    html = TestClient(create_app()).get("/").text
    assert "const MOVING_BELOW = 0.7;" in html
    assert 'class="motion-chip' in html
    assert 'rec.stability = typeof ev.stability === "number" ? ev.stability : null;' in html
    assert 'rec.stability === null ? "motion unknown"' in html
    assert 'rec.stability < MOVING_BELOW ? "moving" : "at rest"' in html
    assert ".motion-chip.moving {" in html and "animation: pulse" in html
    assert 'card.querySelector(".stab > i").style.width' in html   # the bar stays


def test_index_page_renders_the_board_with_vendored_chessground():
    """The live board is the same chessground the snapshot page uses, from the same
    local vendor assets — no CDN, no hand-rolled glyph grid. Chessground needs a box
    that already has real size, so the card is in the DOM before the board is made."""
    html = TestClient(create_app()).get("/").text
    assert 'import { Chessground } from "/static/vendor/chessground.min.js";' in html
    assert "/static/vendor/chessground.base.css" in html
    assert "/static/vendor/chessground.brown.css" in html
    assert "/static/vendor/chessground.cburnett.css" in html
    assert 'orientation: "white", coordinates: true, viewOnly: true,' in html
    assert 'document.getElementById("boards").appendChild(card);' in html
    assert ".board { width: var(--board-size); height: var(--board-size); }" in html
    # The wire string never reaches chessground raw: it goes through the 8x8 sanitizer.
    assert "fen: gridToFen(grid)," in html


def test_index_page_flashes_squares_that_changed():
    """A replay is followable by eye only if the squares that changed announce
    themselves — a green autoShape ring decaying to pale, then gone. The diff is
    against the previous event's grid — the only history the viewer keeps
    (DESIGN D1: no moves, no legality, no game state)."""
    html = TestClient(create_app()).get("/").text
    assert "prevGrid: null," in html
    assert "if (rec.prevGrid !== null && rec.prevGrid[r][c] !== grid[r][c]) {" in html
    assert "rec.prevGrid = grid;" in html
    assert "const FLASH_MS = 900;" in html
    assert 'brush: age < FLASH_MS / 2 ? "green" : "paleGreen"' in html
    # The ring decays on the one shared ticker — never a setTimeout per changed
    # square, which would leak a timer per event.
    assert "if (age > FLASH_MS) { rec.changedAt.delete(key); continue; }" in html
    assert "if (rec.changedAt.size) paintShapes(rec, now);" in html


def test_index_page_marks_occluded_squares():
    """A hand over the board should not assert pieces confidently: under 0.5 max
    per-square probability the square gets a yellow ring, and the legend says so."""
    html = TestClient(create_app()).get("/").text
    assert "const LOW_CONF = 0.5;" in html
    assert "if (squareConfidence(ev.square_dist, r, c) < LOW_CONF) rec.low.push(squareKey(r, c));" in html
    assert 'rec.low.map((key) => ({ orig: key, brush: "yellow" }))' in html
    assert "low confidence" in html and "just changed" in html   # the legend
    # An event without square_dist must still render: unknown reads as confident.
    assert "if (!dist || !dist[r] || !dist[r][c]) return 1;" in html


def test_index_page_updates_the_board_in_place_with_one_ticker():
    """Per-event DOM work stays surgical — one Chessground per card, created once and
    fed with .set() (re-creating it per event would rebuild 64 squares and kill the
    piece transitions). Ageing is one interval for all boards, not one per board."""
    html = TestClient(create_app()).get("/").text
    render_fn = html.split("function render(ev) {")[1].split("\nsetInterval(")[0]
    assert "innerHTML" not in render_fn
    assert "setInterval" not in render_fn and "setTimeout" not in render_fn
    assert "Chessground(" not in render_fn         # created in ensureCard, once
    assert "rec.cg.set({" in render_fn
    assert html.count("Chessground(card.querySelector(\".board\"), {") == 1
    assert html.count("setInterval(") == 1
    assert "const TICK_MS = 200;" in html


def test_index_page_tolerates_malformed_placements():
    """The grid is indexed positionally by the change diff and the square nodes, so
    fenToGrid always yields 8x8 of strings whatever arrives on the wire."""
    html = TestClient(create_app()).get("/").text
    fn = html.split("function fenToGrid(placement) {")[1].split("\n}\n")[0]
    assert 'String(placement ?? "").split("/")' in fn
    assert 'for (const ch of ranks[r] ?? "") {' in fn
    assert 'while (cells.length < 8) cells.push("");' in fn


def test_index_page_keeps_no_game_state():
    """DESIGN D1 guard: the viewer renders an evolving placement and nothing more —
    no move reconstruction, no legality, no PGN."""
    html = TestClient(create_app()).get("/").text.lower()
    for forbidden in ("legal", "pgn", "castl", "en passant", "san("):
        assert forbidden not in html, forbidden


def test_index_page_derives_the_inference_http_origin_from_the_ws_url():
    """The node serves the socket and its HTTP API on one origin, so the frame and
    calibrate endpoints are derived from the injected WS URL — no second placeholder
    to get out of step with it."""
    html = TestClient(create_app()).get("/").text
    assert "__BR_INFERENCE_HTTP__" not in html
    assert "const INFERENCE_HTTP = httpOrigin(WS_URL);" in html
    assert 'u.protocol = u.protocol === "wss:" ? "https:" : "http:";' in html
    assert 'return wsUrl.replace(/^ws/, "http").replace(/\\/[^/]*$/, "");' in html


def test_index_page_has_a_collapsible_frame_preview_left_of_the_board():
    """Seeing what the node actually ingested is how you tell a bad reading from a bad
    frame. It docks on the board's left, is fetched only while open, and survives a
    node that has no frame (or no such endpoint) with a placeholder."""
    html = TestClient(create_app()).get("/").text
    assert 'class="preview-tab"' in html and 'class="preview-pane"' in html
    # Tab first, then the pane, then the board column: the preview grows between the
    # tab and the board, so the tab stays anchored where it was clicked.
    body = html.split('<div class="card-body">')[1].split('<div class="meta">')[0]
    assert body.index("preview-tab") < body.index("preview-pane") < body.index('class="board"')
    assert ".preview-pane { width: 0; overflow: hidden; transition: width 220ms ease;" in html
    assert ".preview.open .preview-pane { width: var(--board-size); margin-left: 8px; }" in html
    # Explicit pixel box, never a percentage height through an auto-height ancestor.
    assert ".preview-pane img { width: var(--board-size); height: var(--board-size);" in html
    # Collapsed costs nothing; open refetches per event, cache-busted by seq.
    assert "if (!rec.previewOpen) return;" in html
    assert "`${INFERENCE_HTTP}/boards/${encodeURIComponent(rec.id)}/frame?seq=${bust}`" in html
    assert 'rec.img.onerror = () => { rec.img.classList.add("absent");' in html
    assert "no frame yet" in html


def test_index_page_calibrates_from_the_live_frame():
    """One-tap calibration from the stream: the ingested frame IS the rectified board,
    so its four corners are the board's. The server's verdict is surfaced verbatim —
    it refuses calibrations that would poison the bank."""
    html = TestClient(create_app()).get("/").text
    assert 'class="calib"' in html
    assert "The board must CURRENTLY show the STARTING position" in html
    assert "const bmp = await createImageBitmap(blob);" in html
    assert "const qs = [0, 0, bmp.width, 0, bmp.width, bmp.height, 0, bmp.height].join(\",\");" in html
    assert '`${INFERENCE_HTTP}/calibrate?corners=${qs}`' in html
    assert 'method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" },' in html
    # Refusals keep the server's own words; success states what the bank now holds.
    assert 'reason = (await r.json()).error ?? reason;' in html
    assert 'showBanner(rec, "err", reason);' in html
    assert "`Calibrated ${d.classes_calibrated}/13 classes${rotNote} — `" in html
    assert "`bank now holds ${d.exemplars} exemplars`" in html
    # One banner timer per card, replaced rather than stacked.
    assert "clearTimeout(rec.bannerTimer);" in html


def test_index_page_refuses_to_calibrate_from_a_reading_that_is_not_worth_it():
    """Calibrating off a stale, non-ok or mid-motion reading poisons the bank. The
    button disables itself and says why in its title rather than failing silently."""
    html = TestClient(create_app()).get("/").text
    reason = html.split("function calibBlockReason(rec) {")[1].split("\n}")[0]
    assert 'if (rec.stale) return "the stream is stale' in reason
    assert 'if (rec.lastSeq === null) return "no frame has arrived yet";' in reason
    assert 'if (rec.status !== "ok")' in reason
    assert "rec.stability !== null && rec.stability < MOVING_BELOW" in reason
    assert "rec.calib.disabled = reason !== null;" in html
    assert "rec.calib.title = reason ??" in html
    # Re-evaluated when the reading changes AND when the card goes stale on the ticker.
    assert html.count("updateCalibState(rec);") >= 4


def test_snapshot_served_and_http_url_injected(monkeypatch):
    monkeypatch.setenv("BR_INFERENCE_HTTP", "http://example.test:9000")
    html = TestClient(create_app()).get("/snapshot").text
    assert "http://example.test:9000" in html
    assert "__BR_INFERENCE_HTTP__" not in html


def test_snapshot_default_http_url(monkeypatch):
    monkeypatch.delenv("BR_INFERENCE_HTTP", raising=False)
    html = TestClient(create_app()).get("/snapshot").text
    assert "http://localhost:8000" in html


def test_snapshot_page_has_v2_flow():
    """Auto-detect + labeled/rotatable corners + photo-with-corners estimate."""
    html = TestClient(create_app()).get("/snapshot").text
    assert '"/detect"' in html or "/detect" in html          # auto-detect call
    assert "rotate labels" in html                            # orientation fix-up
    assert "/estimate?corners=" in html                       # photo + corners mode
    assert 'style.display = "block"' in html                  # the display:none regression guard
    assert "-webkit-touch-callout" in html                    # iOS long-press fix


def test_snapshot_page_required_element_ids_present():
    """IDs the JS (and this test suite) rely on — a rework must not rename them."""
    html = TestClient(create_app()).get("/snapshot").text
    for element_id in (
        "rotate", "undo", "reset", "read", "calib", "file",
        "result", "cornerHint", "board", "inspector", "histList",
    ):
        assert f'id="{element_id}"' in html, element_id


def test_snapshot_page_renders_the_pipeline_as_ordered_stages():
    """The page is a diagnostic surface: the reading algorithm is shown stage by stage,
    in the order it runs, each stage carrying a state and a headline."""
    html = TestClient(create_app()).get("/snapshot").text
    stages = [
        "capture", "detect", "rectify", "sample", "classify", "orient", "repair", "output",
    ]
    positions = []
    for stage in stages:
        marker = f'id="step-{stage}"'
        assert marker in html, stage
        positions.append(html.index(marker))
    assert positions == sorted(positions)  # laid out in pipeline order
    assert (
        'const STEPS = ["capture", "detect", "rectify", "sample", "classify", '
        '"orient", "repair", "output"];'
    ) in html
    assert "function setStep" in html                       # state + headline per stage
    chip = html.split("const CHIP = {")[1].split("}")[0]
    for state in ("idle", "run", "ok", "warn", "fail"):
        assert f"{state}:" in chip, state


def test_snapshot_page_details_are_always_present_but_collapsed():
    """Full detail for every stage always exists — no opt-in checkbox — and every stage
    body starts collapsed (a <details> with no `open` attribute in the markup)."""
    html = TestClient(create_app()).get("/snapshot").text
    body = html.split("<main>")[1].split("</main>")[0]
    assert "<details open" not in body and "<details  open" not in body
    assert body.count("<details>") == 8                      # one per stage
    # The old opt-in diagnostics checkbox is gone; debug images are always requested.
    assert "diagToggle" not in html
    assert '"/estimate?corners=" + qs + "&debug=1"' in html


def test_snapshot_page_has_no_emoji():
    """State is carried by words and colour, never by emoji."""
    html = TestClient(create_app()).get("/snapshot").text
    offenders = [
        c for c in html
        if 0x1F300 <= ord(c) <= 0x1FAFF
        or 0x2600 <= ord(c) <= 0x27BF
        or ord(c) in (0xFE0F, 0x20E3)
        or 0x2B00 <= ord(c) <= 0x2BFF
    ]
    assert not offenders, offenders


def test_snapshot_page_keeps_the_photo_on_screen_beside_the_board():
    """The captured photo is never hidden behind a disclosure: it is the input every
    stage refers to, shown at the board's own size, next to it when there is room."""
    html = TestClient(create_app()).get("/snapshot").text
    body = html.split("<main>")[1].split("</main>")[0]
    assert 'class="photo-col"' in body and 'class="board-col"' in body
    assert body.index('class="photo-col"') < body.index('class="board-col"')
    assert ".photo-col, .board-col { flex: 1 1 340px; max-width: 440px" in html
    assert ".photo-frame { width: 100%; aspect-ratio: 1;" in html
    # The canvas is sized in JS: a percentage max-height would resolve against an
    # auto-height ancestor, be ignored, and clip a portrait photo at the bottom.
    assert "function fitPhoto" in html
    assert "new ResizeObserver(() => { fitPhoto(); renderHandles(); })" in html


def test_snapshot_page_is_mobile_first():
    """Zero-tap primary flow: capture -> auto-detect -> auto-read, corners as fallback."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "viewport-fit=cover" in html                        # safe-area support for sticky bar
    # capture="environment" must NOT be set: on iOS Safari it forces the camera and
    # makes the photo library (cam roll) unreachable — the bare input shows the
    # native camera+library sheet on both platforms.
    assert "capture=" not in html
    assert 'id="file"' in html and 'accept="image/*"' in html  # file input intact
    assert "capture-btn" in html                                # big primary CTA, not a bare input
    assert "await runRead()" in html                            # auto-detect success auto-reads
    assert 'openStep("detect", true)' in html                   # detection failure surfaces itself
    assert 'id="actionBar"' in html                             # run controls, sticky on a phone
    assert "env(safe-area-inset-bottom)" in html
    assert "min-height: 48px" in html                            # ≥48px tall buttons


def test_snapshot_page_corner_handles_are_finger_sized():
    """Corner drag handles are real, thumb-sized DOM elements, not precision-tap canvas
    pixels. 34px (down from 44) so four of them stop hiding the very board edges they are
    meant to be aligned with — precision comes from the loupe, which draws them too."""
    html = TestClient(create_app()).get("/snapshot").text
    assert ".corner-handle" in html
    assert "width: 34px; height: 34px" in html
    assert "margin: -17px 0 0 -17px" in html          # still centred on their own point
    assert "pointerdown" in html and "pointermove" in html and "pointerup" in html


def test_snapshot_page_corner_widget_does_not_block_page_scroll():
    """Regression guard: neither the corner photo/canvas nor the result board may swallow
    all touch scrolling — verified for real with CDP touch dispatch in a Chromium context
    (mouse-based Playwright interactions don't exercise touch-action at all).

    Only a drag starting on a .corner-handle ever captures the pointer. The canvas/stage
    are touch-action: pan-y UNCONDITIONALLY (no state-dependent toggle) and the canvas's
    own pointerdown never calls preventDefault/setPointerCapture — touch-action is decided
    once, at touchstart, by whichever element was actually hit, so reassigning capture to a
    different element afterwards can't retroactively change that decision. Placing a new
    corner is therefore a plain, uncontinued tap (creates the point + its handle); fine-tuning
    it is always a deliberately separate, subsequent gesture on that handle.
    """
    html = TestClient(create_app()).get("/snapshot").text
    # The generic canvas rule and the corner stage must not be a blanket touch-action: none.
    assert "canvas { max-width: 100%; touch-action: pan-y" in html
    assert ".corner-stage { position: relative; line-height: 0; touch-action: pan-y;" in html
    assert ".corner-stage.placing" not in html                     # no state-toggled exception
    assert 'cornerStage.classList.toggle("placing"' not in html
    # touch-action: none exists, but only scoped to the handle itself.
    assert ".corner-handle {" in html and "touch-action: none;" in html
    # The canvas's own pointerdown handler must never call preventDefault/setPointerCapture —
    # only attachHandleEvents (the per-handle listeners) may.
    view_listener = html.split('view.addEventListener("pointerdown"')[1].split("});")[0]
    assert "preventDefault" not in view_listener
    assert "setPointerCapture" not in view_listener
    # The (viewOnly) result board must not block scrolling either.
    assert ".board-wrap { width: 100%; max-width: 460px; aspect-ratio: 1; touch-action: manipulation" in html


def test_snapshot_page_handles_three_estimate_response_shapes():
    """/estimate can answer with a normal event, an autocalibrated event, or occupancy-only
    (DESIGN §4.6 / TESTING Option G) — the page must render all three distinctly.
    """
    html = TestClient(create_app()).get("/snapshot").text

    # Mode 2: autocalibrated — render normally, plus a small non-blocking success banner.
    assert "ev.autocalibrated" in html
    assert 'id="autocalBanner"' in html
    assert "Auto-calibrated from the starting position" in html
    assert "${ev.autocalibrated.matches}/64" in html
    assert "setTimeout(() => { el.style.display = \"none\"; }, 6000)" in html  # auto-dismiss

    # Mode 3: occupancy-only — no placement/confidence, prominent hint, occupancy on the board.
    assert 'ev.mode === "occupancy"' in html
    assert "renderOccupancy" in html
    assert 'id="occupancyHint"' in html
    assert "ev.occupied" in html
    assert 'fen: "8/8/8/8/8/8/8/8"' in html                     # empty board, occupancy only
    # Placement-dependent UI is hidden in occupancy mode.
    assert 'id="boardHint"' in html and 'id="linksRow"' in html and 'id="stmRow"' in html
    # Rotate-labels is meaningless without a placement to validate orientation against.
    assert "updateRotateVisibility" in html
    assert 'lastMode !== "occupancy"' in html


def test_snapshot_page_has_click_to_inspect_then_toggle():
    """First click on a square inspects it (never mutates); clicking the SAME square
    again toggles through its top-3 classes (protocol.CLASSES order) — the fast fix.
    Correct-clicking must never enable chessground's own move interaction.
    """
    html = TestClient(create_app()).get("/snapshot").text
    # Canonical board_reader_core.protocol.CLASSES order, verified against the source.
    assert (
        'const CLASSES = ["empty", "P", "N", "B", "R", "Q", "K", '
        '"p", "n", "b", "r", "q", "k"];'
    ) in html
    assert "function cycleSquare" in html
    assert "function top3Classes" in html
    assert "function applyCorrections" in html
    assert (
        "if (selected && selected.r === r && selected.c === c) cycleSquare(r, c);\n"
        "  else selectSquare(r, c);"
    ) in html
    assert 'brush: "blue"' in html                              # corrected-square marker
    assert 'brush: "red"' in html                                # low-confidence marker retained
    # A plain click listener — never preventDefault/setPointerCapture — so a tap
    # that turns into a scroll never gets hijacked (tap = tap, not touch-capture).
    board_click = html.split('$("board").addEventListener("click"')[1]
    board_click = board_click.split("});")[0]
    assert "preventDefault" not in board_click
    assert "setPointerCapture" not in board_click
    assert "viewOnly: true" in html                              # chessground move interaction stays off


def test_snapshot_page_has_square_inspector_structure():
    """The piece switcher: the crop the model saw, what it read vs what is shown, the
    full distribution, and direct assignment. Empty is privileged; the 12 piece states
    exploit colour symmetry. It sits beside the board, or as a sheet on a phone.
    """
    html = TestClient(create_app()).get("/snapshot").text

    # Beside the board when there is room; a bottom sheet on a narrow screen.
    assert 'id="inspector"' in html
    body = html.split("<main>")[1].split("</main>")[0]
    assert body.index('id="board"') < body.index('id="inspector"')
    assert "#inspector { flex: 1 1 320px; min-width: 0; }" in html
    assert "@media (max-width: 759px)" in html and "#inspector.open { position: fixed" in html

    # Header: square name + what the model read + what is shown now + close.
    assert 'id="cpSquare"' in html
    assert 'id="cpRead"' in html and 'id="cpCurrent"' in html
    assert 'id="cpClose"' in html

    # The crop the classifier actually saw, sliced out of the debug mosaic.
    assert 'id="cpCrop"' in html
    assert "function setCropThumb" in html
    assert "background-size: 800% 800%" in html                 # a uniform 8x8 tiling

    # Empty is privileged: its own full-width row, above the 12-piece grids.
    assert 'id="cpEmptyRow"' in html and 'id="cpEmptyBar"' in html and 'id="cpEmptyPct"' in html
    insp = html.split('id="inspector"')[1].split("</main>")[0]
    assert insp.index('id="cpEmptyRow"') < insp.index('id="cpGridWhite"')

    # 2x6 with colour symmetry: a White row then a Black row, same 6 types in both,
    # vertically aligned by construction (one shared PANEL_TYPES array drives both).
    assert "const PANEL_TYPES" in html
    panel_types_line = html.split("const PANEL_TYPES")[1].split("\n")[0]
    for piece_type in ("K", "Q", "R", "B", "N", "P"):
        assert f'"{piece_type}"' in panel_types_line
    assert '["white", "black"].forEach' in html                # one shared type list, two colours
    assert ".cp-grid { display: grid; grid-template-columns: repeat(6, 1fr)" in html

    # Reused vendored cburnett sprites via chessground's own CSS piece classes — no
    # separate glyph assets.
    assert '<piece class="${color} ${ROLE_NAMES[type]}">' in html
    assert 'cell.className = "cp-cell"' in html

    # Current selection outlined; the model's original argmax subtly marked; the full
    # ranked distribution spelled out underneath.
    assert "idx === curIdx ? \" selected\"" in html
    assert "cp-argmax-dot" in html and "idx === argmaxIdx" in html
    assert "function rankedClasses" in html and 'id="cpRanked"' in html

    # Negligible-probability cells (<1%) are dimmed but remain tappable (cell.onclick is
    # always assigned, independent of the "faint" class).
    assert '(pct < 1 ? " faint" : "")' in html
    assert "cell.onclick = () => assignClass(idx);" in html

    # Opening a square never changes it — assignment is always a separate action.
    assert "function selectSquare" in html
    assert "function assignClass" in html
    assert "function closeInspector" in html
    select_fn = html.split("function selectSquare")[1].split("\n}")[0]
    assert "assignClass" not in select_fn and "corrections.set" not in select_fn

    # Keyboard: FEN letters, so case carries colour exactly as in the placement string.
    assert "/^[kqrbnpKQRBNP]$/.test(k)" in html

    # Clicking outside the board/inspector closes it — a plain click listener
    # (scroll-safe), and it must use the dispatch-time composedPath, not a live
    # .contains() check against a subtree its own handlers rebuild.
    outside_click = html.split("Clicking outside the board and the inspector")[1]
    assert "closeInspector()" in outside_click
    assert "ev.composedPath()" in outside_click
    assert "preventDefault" not in outside_click.split("});")[0]

    # Highlighting the active square on the board via a distinct autoShape (green, not
    # reused from the existing red/blue low-confidence/corrected brushes).
    assert 'brush: "green"' in html


def test_snapshot_page_lists_recent_positions():
    """Every read this session is logged, and selecting one restores the whole pipeline
    view — photo, corners, geometry, crops, distributions and its own corrections."""
    html = TestClient(create_app()).get("/snapshot").text
    assert 'id="histList"' in html and 'id="histClear"' in html
    assert "Recent positions" in html
    assert "function pushRead" in html and "function restoreRead" in html
    assert "const READS_CAP" in html
    # Corrections belong to the read they were made against.
    assert "active.corrections = [...corrections]" in html
    assert "corrections = new Map(restored ? restored.corrections : [])" in html
    # Restoring reinstates the earlier photo AND its corner labelling, so the evidence
    # on screen always belongs to one single read.
    assert "capState = e.cap; detectState = e.detect;" in html
    assert "rot = e.rot; cornersEdited = e.cornersEdited;" in html


def test_snapshot_page_has_reset_corrections_button():
    html = TestClient(create_app()).get("/snapshot").text
    assert 'id="resetCorrections"' in html
    assert "function resetCorrections" in html
    assert '"resetCorrections").style.display = "none"' in html  # hidden until a correction exists


def test_snapshot_page_resets_corrections_on_rotate():
    """Rotating labels changes which corner is a8 — stale per-square corrections from
    the old mapping must not silently carry over."""
    html = TestClient(create_app()).get("/snapshot").text
    rotate_handler = html.split("function rotateLabels() {")[1].split("}")[0]
    assert "resetCorrections()" in rotate_handler


def test_snapshot_page_has_low_light_warning():
    """A top-level warnings:["low_light"] array (which may be absent) shows an amber
    banner near the result, and the stage that measured it explains the threshold."""
    html = TestClient(create_app()).get("/snapshot").text
    assert 'id="lowLightWarning"' in html
    assert 'class="banner warn"' in html
    assert "function updateLowLightWarning" in html
    assert 'ev.warnings?.includes("low_light")' in html
    assert "Photo is very dark" in html
    # The geometry stage mirrors the server's own thresholds so a warning is explained,
    # not just announced (board_reader_inference.server LOW_LIGHT_MEAN / LOW_ANGLE_HW).
    assert "const LOW_ANGLE_HW = 0.62;" in html
    assert "const LOW_LIGHT_MEAN = 55.0;" in html


def test_snapshot_page_uses_vendored_chessground():
    """Result board is chessground, loaded from local vendor assets (no CDN)."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "/static/vendor/chessground.min.js" in html
    assert "/static/vendor/chessground.base.css" in html
    assert "/static/vendor/chessground.brown.css" in html
    assert "/static/vendor/chessground.cburnett.css" in html
    assert "Chessground(" in html
    assert "viewOnly" in html
    assert "autoShapes" in html                                # low-confidence markers


def test_snapshot_page_always_shows_the_recognition_evidence():
    """This page IS the diagnostic surface: the sampling boxes, the model-input mosaic,
    the per-square confidence map and the raw event are always fetched and always
    rendered — collapsed, never opt-in."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "debug=1" in html
    assert 'id="diagBoxes"' in html and 'id="diagMosaic"' in html
    assert 'id="confMap"' in html and "function renderConfMap" in html
    assert 'id="lowChips"' in html and "function lowConfidenceSquares" in html
    assert 'id="rawEvent"' in html and "function updateRawEvent" in html
    # The raw event drops the two base64 images (they are shown as images) and the
    # 8x8x13 tensor (it is shown as the confidence map + the inspector).
    raw = html.split("function updateRawEvent")[1].split("\n}")[0]
    assert "delete shallow.debug;" in raw
    assert "square_dist" in raw
    # Estimator-state changes the server volunteers are surfaced, not swallowed.
    assert "function renderEstimatorNotes" in html
    assert "ev.bank_refreshed" in html and "ev.autocalibrated" in html
    # A material census is the cheapest sanity check on a read.
    assert "function renderMaterial" in html


def test_static_vendor_serves_chessground_assets():
    client = TestClient(create_app())

    js = client.get("/static/vendor/chessground.min.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]

    css = client.get("/static/vendor/chessground.base.css")
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]


def test_vendor_readme_exists():
    readme = (_STATIC / "vendor" / "README.md").read_text(encoding="utf-8")
    assert "chessground" in readme
    assert "9.2.1" in readme
    assert "GPL" in readme


def test_viewer_does_not_depend_on_node_packages():
    """The viewer is a pure static server — it must not import core/edge/inference."""
    import board_reader_viewer.server as srv

    text = Path(srv.__file__).read_text(encoding="utf-8")
    for forbidden in ("board_reader_core", "board_reader_edge", "board_reader_inference"):
        assert forbidden not in text


def test_snapshot_page_can_confirm_a_position_as_ground_truth():
    """The correction panel is the only place a human ever states what a board really
    holds; confirming keeps that answer instead of discarding it on reload."""
    html = TestClient(create_app()).get("/snapshot").text
    assert 'id="confirmPosition"' in html and 'id="labelHint"' in html
    # Posts the board AS SHOWN (base read with corrections layered on) plus how many
    # squares the operator changed, against the upload id the read came back with.
    assert '"/labels/" + encodeURIComponent(lastUpload)' in html
    assert "placement: lastPlacement, corrections: corrections.size" in html


def test_snapshot_page_explains_rather_than_hides_labelling_without_retention():
    """Without BR_SAVE_UPLOADS the response carries no upload id, so a label would have
    no image to belong to. The button stays put and says so — silently vanishing is what
    made operators believe the page had no save-labelling button at all."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "lastUpload = ev.upload ?? null" in html
    offer = html.split("function updateConfirmButton() {")[1].split("\n}\n")[0]
    assert 'lastMode === "position"' in offer      # occupancy has no placement to state
    assert "if (!lastUpload) {" in offer
    assert "btn.disabled = true;" in offer
    assert "hint.textContent = NO_RETENTION_HINT;" in offer
    assert "BR_SAVE_UPLOADS" in html
    # ...and it is never hidden for lack of an upload id: the only thing that hides it is
    # having no position on screen at all.
    assert 'btn.style.display' not in offer


def test_snapshot_page_invalidates_a_saved_label_when_a_square_changes():
    """A label describes one exact position — editing a square after saving must not
    leave the button claiming the new board was confirmed."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "updateConfirmButton();   // editing a square invalidates any label already saved" in html
    assert "savedLabel !== null && savedLabel === lastPlacement" in html
    # Save -> saved (disabled) -> actionable again once the shown placement changes.
    assert '"Labelling saved" : (savedLabel === null ? "Save labelling" : "Update labelling")' in html


def test_snapshot_page_promotes_the_save_labelling_button_to_the_board_column():
    """Saving the labelling is a first-class action on the position, so it lives with the
    position — a real primary button under the board, not a text-button buried in the
    collapsed stage-7 body where it read as missing."""
    html = TestClient(create_app()).get("/snapshot").text
    body = html.split("<main>")[1].split("</main>")[0]
    board_col = body.split('class="board-col"')[1].split("</div>\n\n")[0]
    assert 'id="confirmPosition"' in board_col
    assert 'id="labelHint"' in board_col
    assert 'id="labelRow"' in board_col
    # Promoted out of stage 7 entirely, and no longer a small text-button.
    stage7 = body.split('id="step-output"')[1]
    assert "confirmPosition" not in stage7 and "labelHint" not in stage7
    assert '<button id="confirmPosition" type="button">' in html  # no class="text-btn"


def test_snapshot_page_absorbs_the_servers_autorotation_into_its_own_labels():
    """A read posted with &autorotate=1 comes back rotated by k quarter-turns. The page
    must adopt that k into its own corner labelling, or the board shows the rotated
    position while the handles, the corner table, the rectified preview and the per-square
    crop all still describe the pre-rotation frame.

    orderedCorners()[i] = pts[(i - rot + 4) % 4], and reindexing the lattice by
    rot90(., k) is a +k cyclic shift of the ordered corners — so rot -= k.
    """
    html = TestClient(create_app()).get("/snapshot").text
    assert "function absorbAutorotation" in html
    fn = html.split("function absorbAutorotation(ev, restored) {")[1].split("\n}\n")[0]
    assert "rot = (rot - k + 4) % 4;" in fn
    # Never on the restore path: a stored entry's rot already includes the absorption.
    assert "if (restored !== null || !ev.autorotated) return;" in fn
    assert "autoRotApplied = e.autoRot;" in html
    # Re-render what depends on the corners, but never re-read: the event in hand is
    # already expressed in the new frame.
    assert "draw();" in fn and "maybeRectify();" in fn
    assert "runRead" not in fn
    # Absorbed BEFORE the orientation stage reports and BEFORE the history entry is made.
    render_fn = html.split("function render(ev, restored = null) {")[1].split("\n}\n")[0]
    assert render_fn.index("absorbAutorotation(ev, restored)") < render_fn.index("updateOrientStep(ev)")
    assert render_fn.index("absorbAutorotation(ev, restored)") < render_fn.index("pushRead(ev)")
    # The corners stay auto-detected, so later reads still ask for auto-rotation.
    assert "cornersEdited = true" not in fn
    # And the page says plainly what happened, in both places that used to mislead.
    assert "and this page adopted it" in html
    assert "adopted by this page" in html


def test_snapshot_page_has_an_optional_hover_magnifier_that_shows_the_handles():
    """Hovering the photo can magnify it, off by default. Mouse only — touch has no hover
    and the scroll invariants must survive — and the loupe draws the corner quad and its
    labelled handles, which are DOM elements and therefore invisible in a magnified copy
    of the canvas alone."""
    html = TestClient(create_app()).get("/snapshot").text
    assert 'id="magnifier"' in html
    assert 'aria-pressed="false"' in html                    # off by default
    assert 'button.secondary[aria-pressed="true"]' in html    # state drives the styling
    assert "let magnifierOn = false;" in html
    # Mouse-only hover, and never while a drag already owns the loupe.
    hover = html.split('cornerStage.addEventListener("pointermove"')[1].split("});")[0]
    assert 'ev.pointerType !== "mouse"' in hover
    assert "dragIdx !== null" in hover
    assert "preventDefault" not in hover and "setPointerCapture" not in hover
    assert 'cornerStage.addEventListener("pointerleave"' in html
    # The handles are drawn INTO the loupe, in loupe coordinates.
    assert "function drawLoupeCorners" in html
    assert "drawLoupeCorners(p, Z);" in html
    assert "const at = (q) => [(q[0] - p[0]) * Z + cx, (q[1] - p[1]) * Z + cy];" in html
    assert "lctx.fillText(labelOf(i)" in html


def test_snapshot_page_lists_every_retained_image_not_just_this_session():
    """Recent positions merges this session's reads with every photo the inference node
    still holds, newest first — a photo read here restores in full, a node-only photo is
    loaded and read again for real."""
    html = TestClient(create_app()).get("/snapshot").text
    assert '"/uploads?limit=" + UPLOADS_LIMIT' in html
    assert "function refreshUploads" in html
    assert "function historyRows" in html
    assert "function nodeRow" in html and "function sessionRow" in html
    assert 'kind: "node"' in html and 'kind: "session"' in html
    # A read this session owns its full event, so it restores rather than re-reads; it is
    # matched to the node's record by upload id and never listed twice.
    assert "const owned = new Set(reads.map((e) => e.upload).filter(Boolean));" in html
    assert "if (owned.has(u.id)) continue;" in html
    assert "row.onclick = () => restoreRead(e.id);" in html
    assert "row.onclick = () => loadUpload(u);" in html
    # Thumbnails come from the node, lazily and long-side capped.
    assert '"/uploads/" + encodeURIComponent(id) + "/photo"' in html
    assert 'uploadPhotoUrl(u.id, 160)' in html and 'loading="lazy"' in html
    # Refreshed on load, after every completed read, and after every saved labelling.
    assert html.count("refreshUploads();") >= 3


def test_snapshot_page_loading_a_retained_photo_is_an_honest_read():
    """Clicking a node-held photo rebuilds stages 1-3 from THAT image and the corners the
    node recorded for it, then runs a real read — no faking a result from the summary."""
    html = TestClient(create_app()).get("/snapshot").text
    fn = html.split("async function loadUpload(u) {")[1].split("\n}\n")[0]
    assert "drawSourceFrom(im)" in fn
    assert "parseCorners(u.corners, k)" in fn
    assert "rot = 0; autoRotApplied = 0; dragIdx = null;" in fn
    assert "cornersEdited = false;" in fn
    assert "await runRead();" in fn
    assert "resetSteps(\"capture\")" in fn and "applyCapState();" in fn
    # No usable corners on record: detect them again rather than invent them.
    assert "await autoDetect();" in fn
    # The photo is fetched as a blob: a cross-origin <img> would taint the source canvas,
    # which this page reads back (getImageData for luma, toDataURL for the thumbnail).
    assert "URL.createObjectURL(await r.blob())" in fn
    assert "function parseCorners" in html
    assert "n.length !== 8" in html                        # a8,h8,h1,a1 as 8 numbers


def test_snapshot_page_degrades_gracefully_without_the_uploads_endpoint():
    """A node that predates /uploads, or has retention off, must cost the page nothing:
    the session list stands and the reason is stated."""
    html = TestClient(create_app()).get("/snapshot").text
    fn = html.split("async function refreshUploads() {")[1].split("\n}\n")[0]
    assert "} catch {" in fn                                # a failed fetch is not fatal
    assert "serverUploads = [];" in fn
    assert "retention = null;" in fn
    assert "renderHistory();" in fn
    note = html.split("function historyNote() {")[1].split("\n}\n")[0]
    assert "Photo retention is off on the inference node" in note
    assert "only this session's reads are listed" in note
    # Clearing drops only this session's entries, and the copy says so.
    assert ">Clear this session<" in html
    assert "Clear removes only this session's entries" in html
