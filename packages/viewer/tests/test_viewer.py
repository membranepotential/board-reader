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
    """IDs the JS (and this test suite) rely on — a mobile rework must not rename them."""
    html = TestClient(create_app()).get("/snapshot").text
    for element_id in (
        "rotate", "undo", "reset", "read", "calib", "file",
        "diagToggle", "result", "cornerHint", "diag",
    ):
        assert f'id="{element_id}"' in html, element_id


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
    assert "cornersDetails.open = true" in html                 # corner UI opens only as fallback
    assert 'id="actionBar"' in html                             # sticky bottom action bar
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
    assert ".corner-stage { position: relative; line-height: 0; touch-action: pan-y; }" in html
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
    assert ".board-wrap { width: 100%; max-width: 420px; aspect-ratio: 1; touch-action: manipulation" in html


def test_snapshot_page_handles_three_estimate_response_shapes():
    """/estimate can answer with a normal event, an autocalibrated event, or occupancy-only
    (DESIGN §4.6 / TESTING Option G) — the page must render all three distinctly.
    """
    html = TestClient(create_app()).get("/snapshot").text

    # Mode 2: autocalibrated — render normally, plus a small non-blocking success banner.
    assert "ev.autocalibrated" in html
    assert 'id="autocalBanner"' in html
    assert "Auto-calibrated from the starting position" in html
    assert "${autocal.matches}/64" in html
    assert "setTimeout(() => { el.style.display = \"none\"; }, 6000)" in html  # auto-dismiss

    # Mode 3: occupancy-only — no placement/confidence, prominent hint, occupancy on the board.
    assert 'ev.mode === "occupancy"' in html
    assert "renderOccupancy" in html
    assert 'id="occupancyHint"' in html
    assert "ev.occupied" in html
    assert 'fen: "8/8/8/8/8/8/8/8"' in html                     # empty board, occupancy only
    # Placement-dependent UI is hidden in occupancy mode.
    assert 'id="metaRow"' in html and 'id="linksRow"' in html and 'id="stmRow"' in html
    # Rotate-labels is meaningless without a placement to validate orientation against.
    assert "updateRotateVisibility" in html
    assert 'lastMode !== "occupancy"' in html


def test_snapshot_page_has_tap_to_correct_squares():
    """Tapping a result square cycles its top-3 classes (protocol.CLASSES order);
    correct-clicking must never enable chessground's own move interaction.
    """
    html = TestClient(create_app()).get("/snapshot").text
    # Canonical board_reader_core.protocol.CLASSES order, verified against the source.
    assert (
        'const CLASSES = ["empty", "P", "N", "B", "R", "Q", "K", '
        '"p", "n", "b", "r", "q", "k"];'
    ) in html
    assert "function cycleSquareCorrection" in html
    assert "function top3Classes" in html
    assert "function applyCorrections" in html
    assert 'brush: "blue"' in html                              # corrected-square marker
    assert 'brush: "red"' in html                                # low-confidence marker retained
    # A plain click listener — never preventDefault/setPointerCapture — so a tap
    # that turns into a scroll never gets hijacked (tap = tap, not touch-capture).
    board_click = html.split('document.getElementById("board").addEventListener("click"')[1]
    board_click = board_click.split("});")[0]
    assert "preventDefault" not in board_click
    assert "setPointerCapture" not in board_click
    assert "viewOnly: true" in html                              # chessground move interaction stays off


def test_snapshot_page_has_correction_panel_structure():
    """Felix's spec: argmax by default; tap advances AND opens a distribution panel with
    direct selection; empty is privileged; the 12 piece states exploit color symmetry.
    """
    html = TestClient(create_app()).get("/snapshot").text

    # The panel is a mobile-first bottom sheet, stacked ABOVE the sticky action bar (never
    # covering the board, which stays inline in the scrollable content above both).
    assert 'id="bottomStack"' in html
    assert 'id="correctionPanel"' in html and 'class="correction-panel"' in html
    bottom_stack = html.split('<div id="bottomStack">')[1].split("<script")[0]
    assert bottom_stack.index('id="correctionPanel"') < bottom_stack.index('id="actionBar"')

    # Header: square name + current state/probability + a close affordance.
    assert 'id="cpSquare"' in html
    assert 'id="cpCurrent"' in html
    assert 'id="cpClose"' in html

    # Empty is privileged: its own full-width row, first, above the 12-piece grid, with a
    # probability bar of its own.
    assert 'id="cpEmptyRow"' in html and 'id="cpEmptyBar"' in html and 'id="cpEmptyPct"' in html
    panel_html = html.split('id="correctionPanel"')[1].split("</script>")[0]
    assert panel_html.index('id="cpEmptyRow"') < panel_html.index('id="cpGrid"')

    # 2x6 grid with color symmetry: White row then Black row, same 6 types in both,
    # vertically aligned by construction (one shared PANEL_TYPES array drives both rows).
    assert "const PANEL_TYPES" in html
    panel_types_line = html.split("const PANEL_TYPES")[1].split("\n")[0]
    for piece_type in ("K", "Q", "R", "B", "N", "P"):
        assert f'"{piece_type}"' in panel_types_line
    assert '["white", "black"].forEach' in html                # one shared type list, two colors
    assert ".cp-grid { display: grid; grid-template-columns: repeat(6, 1fr)" in html

    # Reused vendored cburnett sprites via chessground's own CSS piece classes — no
    # separate glyph assets.
    assert '<piece class="${color} ${ROLE_NAMES[type]}">' in html
    assert '"white"' in html and 'cell.className = "cp-cell"' in html   # per-cell selection styling

    # Current selection outlined; the model's original argmax subtly marked.
    assert "idx === curIdx ? \" selected\"" in html
    assert "cp-argmax-dot" in html and "idx === argmaxIdx" in html

    # Negligible-probability cells (<1%) are dimmed but remain tappable (cell.onclick is
    # always assigned, independent of the "faint" class).
    assert '(pct < 1 ? " faint" : "")' in html
    assert "cell.onclick = () => selectPanelClass(idx);" in html

    # Tapping a square both advances (fast path) AND opens the panel for it.
    assert "function selectPanelClass" in html
    assert "function openPanelFor" in html
    assert "function closePanel" in html
    cycle_fn = html.split("function cycleSquareCorrection")[1].split("\n\n")[0]
    assert "openPanelFor(r, c)" in cycle_fn

    # Tapping outside the board/panel closes it — a plain click listener (scroll-safe).
    outside_click = html.split("Tapping anywhere outside the board and the panel")[1]
    assert "closePanel()" in outside_click
    assert "preventDefault" not in outside_click.split("});")[0]

    # Highlighting the active square on the board via a distinct autoShape (green, not
    # reused from the existing red/blue low-confidence/corrected brushes).
    assert 'brush: "green"' in html


def test_snapshot_page_has_reset_corrections_button():
    html = TestClient(create_app()).get("/snapshot").text
    assert 'id="resetCorrections"' in html
    assert "function resetCorrections" in html
    assert '"resetCorrections").style.display = "none"' in html  # hidden until a correction exists


def test_snapshot_page_resets_corrections_on_rotate():
    """Rotating labels changes which corner is a8 — stale per-square corrections from
    the old mapping must not silently carry over."""
    html = TestClient(create_app()).get("/snapshot").text
    rotate_handler = html.split("rotateBtn.onclick = ")[1].split("\n")[0]
    assert "resetCorrections()" in rotate_handler


def test_snapshot_page_has_low_light_warning():
    """A top-level warnings:["low_light"] array (which may be absent) shows a small
    amber line near the result, same styling family as the occupancy hint."""
    html = TestClient(create_app()).get("/snapshot").text
    assert 'id="lowLightWarning"' in html
    assert "class=\"low-light-hint\"" in html
    assert "function updateLowLightWarning" in html
    assert 'ev.warnings?.includes("low_light")' in html
    assert "Photo is very dark" in html


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


def test_snapshot_page_has_diagnostics_toggle():
    """Debug images are opt-in and hidden until the server returns a debug payload."""
    html = TestClient(create_app()).get("/snapshot").text
    assert "diagToggle" in html
    assert "debug=1" in html
    assert 'id="diag"' in html


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
