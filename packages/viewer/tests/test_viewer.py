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
        "capture", "detect", "rectify", "sample", "classify", "orient", "output",
    ]
    positions = []
    for stage in stages:
        marker = f'id="step-{stage}"'
        assert marker in html, stage
        positions.append(html.index(marker))
    assert positions == sorted(positions)  # laid out in pipeline order
    assert 'const STEPS = ["capture", "detect", "rectify", "sample", "classify", "orient", "output"];' in html
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
    assert body.count("<details>") == 7                      # one per stage
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
    """Corner drag handles are real ≥44px DOM elements, not precision-tap canvas pixels."""
    html = TestClient(create_app()).get("/snapshot").text
    assert ".corner-handle" in html
    assert "width: 44px; height: 44px" in html
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


def test_snapshot_page_only_offers_labelling_when_a_photo_was_retained():
    """Without BR_SAVE_UPLOADS the response carries no upload id, so a label would
    have no image to belong to and could never be replayed by an eval."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "lastUpload = ev.upload ?? null" in html
    offer = html.split("function updateConfirmButton() {")[1].split("\n}")[0]
    assert 'lastMode === "position"' in offer
    assert 'btn.style.display = offer ? "inline-block" : "none"' in offer


def test_snapshot_page_invalidates_a_saved_label_when_a_square_changes():
    """A label describes one exact position — editing a square after saving must not
    leave the button claiming the new board was confirmed."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "updateConfirmButton();   // editing a square invalidates any label already saved" in html
    assert "savedLabel !== null && savedLabel === lastPlacement" in html
