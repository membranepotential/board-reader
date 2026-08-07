"""Tests for the phone camera page — the DESIGN §4.1 "virtual Pi" in a browser.

The page is a single static asset, so these are structural assertions against the
served HTML, in the style of the existing index/snapshot tests: they pin the
behaviours that are load-bearing and easy to lose in a rework (the wire contract, the
edge semantics it mirrors, the touch invariants, the secure-context explainer).
"""

from __future__ import annotations

from starlette.testclient import TestClient

from board_reader_viewer import tls
from board_reader_viewer.server import _banner, create_app


def camera_html() -> str:
    response = TestClient(create_app()).get("/camera")
    assert response.status_code == 200
    return response.text


def test_camera_page_is_served():
    assert "board-reader · camera" in camera_html()


def test_other_routes_still_work():
    client = TestClient(create_app())
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
    assert client.get("/snapshot").status_code == 200


def test_camera_page_needs_no_injection():
    """Every endpoint it uses is same-origin and relative — that is the whole point:
    one origin means the operator accepts one certificate on the phone, not two. A
    leftover placeholder would mean the page is reaching across origins again."""
    html = camera_html()
    assert "__BR_INFERENCE_WS__" not in html
    assert "__BR_INFERENCE_HTTP__" not in html
    for endpoint in ('wsUrl("/ws/ingest")', 'wsUrl("/ws/viewer")',
                     '"/api/detect"', "`/api/calibrate?corners=${qs}`"):
        assert endpoint in html, endpoint
    # ws:// over http, wss:// over https — never a hardcoded scheme.
    assert 'u.protocol = u.protocol === "https:" ? "wss:" : "ws:";' in html


def test_camera_page_asks_for_the_rear_camera_and_keeps_the_screen_awake():
    html = camera_html()
    assert 'facingMode: { ideal: "environment" }' in html
    # `ideal`, not `exact`: a laptop with only a front camera still gets a preview.
    assert 'facingMode: { exact:' not in html
    assert 'navigator.wakeLock?.request("screen")' in html
    # A screen lock is released when the page is hidden; coming back must re-take one.
    assert 'document.visibilityState === "visible" && streaming && wakeLock === null' in html


def test_camera_page_explains_an_insecure_context_rather_than_failing_silently():
    """Over plain http the camera is never offered at all — nothing is wrong with the
    permissions. The page has to say that, and say both ways out."""
    html = camera_html()
    assert "if (!window.isSecureContext) { block(" in html
    assert "board-reader-viewer --host 0.0.0.0 --ssl" in html
    assert "adb reverse tcp:${PORT} tcp:${PORT}" in html
    assert "http://localhost:${PORT}/camera" in html
    # The port comes from the page's own URL, so the instructions are copy-pasteable.
    assert "const PORT = location.port ||" in html
    # Capture failures are named, not swallowed into "something went wrong".
    for error in ("NotAllowedError", "NotFoundError", "NotReadableError",
                  "OverconstrainedError"):
        assert error in html, error


def test_camera_page_sends_the_two_part_wire_frame_with_corners():
    """board_reader_core.wire: a JSON header (text) then the JPEG (binary), in that
    order, plus the quad the node needs to find the board inside the crop."""
    html = camera_html()
    send = html.split("async function sendFrame(decision, captureTs) {")[1].split("\n}\n")[0]
    assert 'type: "frame",' in send
    assert "board_id: boardId()," in send
    assert "seq: seq++," in send
    assert "capture_ts: captureTs," in send
    assert "stability: decision.stability," in send
    assert "burst_id: decision.burstId," in send
    assert "corners," in send
    assert "ws.send(JSON.stringify({" in send
    # The JPEG follows its own header, never the other way round.
    assert send.index("ws.send(JSON.stringify({") < send.index("ws.send(blob);")


def test_camera_page_rebases_the_corners_into_the_image_it_actually_sends():
    """The node never sees the full camera frame, so corners in camera coordinates
    would point outside the image it holds: they are translated by the crop origin and
    scaled by the downscale, in canonical a8,h8,h1,a1 order."""
    html = camera_html()
    assert "const sx = w / box.w, sy = h / box.h;" in html
    assert "orderedCorners().flatMap(([x, y]) =>" in html
    assert "Math.round((x - box.x) * sx * 10) / 10" in html
    assert "Math.round((y - box.y) * sy * 10) / 10" in html
    assert 'const CORNER_ORDER = ["a8", "h8", "h1", "a1"];' in html
    assert "return CORNER_ORDER.map((_, k) => pts[(k - rot + 4) % 4]);" in html


def test_camera_page_crops_with_headroom_above_the_board():
    """Pieces are TALL: a crop drawn on the playing surface decapitates the back rank at
    any real camera angle, which is the same failure tall boxes exist to avoid."""
    html = camera_html()
    assert "const MARGIN = 0.10;" in html
    assert "const HEADROOM = 0.30;" in html
    box = html.split("function cropBox() {")[1].split("\n}")[0]
    assert "Math.floor(y0 - (MARGIN + HEADROOM) * h)" in box   # extra, above the top edge
    assert "Math.ceil(y1 + MARGIN * h)" in box
    assert "Math.floor(x0 - MARGIN * w)" in box and "Math.ceil(x1 + MARGIN * w)" in box
    # Never outside the frame.
    assert "Math.max(0," in box and "Math.min(video.videoWidth," in box
    # And the operator can see exactly what is being sent.
    assert 'id="cropRect"' in html and "cropRect.setAttribute" in html


def test_camera_page_downscales_and_encodes_the_stream_frame():
    html = camera_html()
    assert "const SEND_LONG_SIDE = 960;" in html
    assert "const JPEG_QUALITY = 0.75;" in html
    assert "Math.min(1, SEND_LONG_SIDE / Math.max(box.w, box.h))" in html
    assert 'canvas.toBlob(res, "image/jpeg", quality)' in html


def test_camera_page_mirrors_the_edge_motion_meter():
    """A JS twin of board_reader_edge.motion.MotionMeter: gain curve over the mean
    absolute difference of a 32x32 gray thumbnail, then a fast-attack/slow-release
    smoother, and 1.0 on the very first frame (D3: a hint, never a gate)."""
    html = camera_html()
    assert "class MotionMeter {" in html
    assert "downscale = 32, gain = 80, attack = 0.7, release = 0.4" in html
    assert "noiseFloor = 0.004" in html
    assert "Math.exp(-this.gain * Math.max(diff - this.noiseFloor, 0))" in html
    assert "const weight = instant < this.stability ? this.attack : this.release;" in html
    assert "(1 - weight) * this.stability + weight * instant" in html
    assert "if (prev === null) { this.stability = 1; return 1; }" in html
    # cv2's BGR2GRAY weights, so a scene scores the same here as on the edge.
    assert "0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]" in html


def test_camera_page_mirrors_the_edge_adaptive_forwarder():
    """board_reader_edge.forwarder (DESIGN D4): keepalive at rest, burst during motion,
    eager entry and patient exit, one shared burst_id per burst, every decision made
    against capture time (D5) rather than a separate clock."""
    html = camera_html()
    assert "const KEEPALIVE_FPS = 1.0;" in html
    assert "const BURST_FPS = 12.0;" in html
    assert "const ENTER_BURST = 0.7;" in html
    assert "const EXIT_BURST = 0.85;" in html
    assert "const EXIT_HOLD_S = 0.7;" in html
    track = html.split("trackBurst(stability, captureTs, boardId) {")[1].split("\n  }")[0]
    assert "if (stability < ENTER_BURST) {" in track
    # Named from its own never-reset counter, NOT from seq: seq advances only on
    # frames actually sent, so a fully-dropped burst would reuse its id.
    assert "this.burstId = `${boardId}:b${this.burstCount++}`;" in track
    assert "if (stability <= EXIT_BURST) this.calmSince = null;" in track
    assert "captureTs - this.calmSince >= EXIT_HOLD_S - DUE_EPS" in track
    due = html.split("due(captureTs) {")[1].split("\n  }")[0]
    # The rate is read at decision time, so entering a burst raises it immediately
    # instead of waiting out the keepalive interval already pending.
    assert "const fps = this.bursting ? BURST_FPS : KEEPALIVE_FPS;" in due
    assert "if (elapsed < 0) return true;" in due                  # clock stepped back
    # Every sampled frame reaches the meter — dropping frames upstream would make
    # stability lie — so the sampler runs faster than any forward rate.
    assert "const CAPTURE_FPS = 20;" in html
    assert "const stability = this.meter.update(source);" in html


def test_camera_page_drops_frames_rather_than_queueing_them():
    html = camera_html()
    assert "const MAX_BUFFERED = 192 * 1024;" in html
    assert "if (encoding || ws.bufferedAmount > MAX_BUFFERED) { dropped++; return; }" in html


def test_camera_page_reconnects_the_ingest_socket_with_backoff():
    html = camera_html()
    assert "const RECONNECT_MIN_MS = 500, RECONNECT_MAX_MS = 8000;" in html
    assert "reconnectMs = Math.min(reconnectMs * 2, RECONNECT_MAX_MS);" in html
    # A close carries the relay's reason when the inference node is what is down.
    assert 'setConn(ev.reason ? "reconnecting — " + ev.reason : "reconnecting…", "warn");' in html
    # Acks are per board: another board's stream must not move this page's counter.
    assert 'typeof d.ack === "number" && d.board_id === boardId()' in html
    # Every socket handler closes over ITS socket and stands down once replaced —
    # a stale close must neither null the live `ws` nor spawn a duplicate connection.
    connect = html.split("function connectIngest() {")[1].split("\n}")[0]
    assert "const sock = new WebSocket(wsUrl(\"/ws/ingest\"));" in connect
    assert connect.count("if (sock !== ws) return;") >= 3
    assert "setTimeout(() => { if (streaming && ws === null) connectIngest(); }, reconnectMs);" in connect


def test_camera_page_corner_handles_are_finger_sized_and_scroll_safe():
    """Same touch contract as the snapshot page: only a drag that starts ON a handle
    ever takes the pointer, so a thumb anywhere else still scrolls the page."""
    html = camera_html()
    assert ".corner-handle { position: absolute; width: 34px; height: 34px;" in html
    assert "margin: -17px 0 0 -17px" in html
    assert ".camera-stage { position: relative; line-height: 0; touch-action: pan-y;" in html
    assert ".camera-stage.placing" not in html            # no state-toggled exception
    handle = html.split("function attachHandleEvents(el, idx) {")[1].split("\n}")[0]
    assert "el.setPointerCapture(ev.pointerId);" in handle
    assert "pointerdown" in handle and "pointermove" in handle and "pointerup" in handle
    # Neither the stage nor the video ever captures a pointer or preventDefaults a touch.
    assert "cameraStage.setPointerCapture" not in html
    assert 'video.addEventListener("pointerdown"' not in html
    # Handles are positioned in CSS px off the video's rect, so a reflow must re-lay them.
    assert "new ResizeObserver(() => draw()).observe(cameraStage);" in html


def test_camera_page_orientation_is_the_operators():
    """Geometry cannot know which corner is a8 (DESIGN decision #15): detection replaces
    the quad, never the labelling, and rotate labels is how the operator states it."""
    html = camera_html()
    assert 'id="rotate"' in html and "rotate labels" in html
    assert "rot = (rot + 1) % 4;" in html
    assert "function labelOf(i) { return CORNER_ORDER[(i + rot) % 4]; }" in html
    detect = html.split("async function detectCorners() {")[1].split("\n}\n")[0]
    assert "pts = corners.map(([x, y]) => [Math.round(x / k), Math.round(y / k)]);" in detect
    assert "rot = 0" not in detect                       # a detection never relabels
    assert "geometry cannot tell which corner " in html
    # Detection is offered at a sane size, and its refusal is the node's own words.
    assert "const DETECT_LONG_SIDE = 1600;" in html
    assert "(await r.json()).error ?? reason" in detect


def test_camera_page_persists_the_crop_per_board_as_fractions():
    """The camera may well come back at a different resolution; a crop stored in the old
    resolution's pixels would silently point at the wrong part of the board."""
    html = camera_html()
    assert 'function cornerKey(id) { return "board-reader.camera.corners." + id; }' in html
    save = html.split("function saveCorners() {")[1].split("\n}")[0]
    assert "[x / video.videoWidth, y / video.videoHeight]" in save
    load = html.split("function loadCorners() {")[1].split("\n}")[0]
    assert "[x * video.videoWidth, y * video.videoHeight]" in load
    # A quad always exists, so the page is streamable the moment the camera opens.
    assert "function defaultCorners()" in html
    assert "if (!loadCorners()) defaultCorners();" in html


def test_camera_page_calibrates_from_a_full_resolution_frame():
    """Calibration reads one exemplar per square, so it must never be handed the
    downscaled stream frame. Corners go up in the full frame's own pixels."""
    html = camera_html()
    calib = html.split("async function calibrate() {")[1].split("\n}\n")[0]
    assert "full.width = video.videoWidth; full.height = video.videoHeight;" in calib
    assert "const qs = orderedCorners().flat().map(Math.round).join(\",\");" in calib
    assert "`/api/calibrate?corners=${qs}`" in calib
    assert 'method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" },' in calib
    # The same wording the live page uses — one claim, made in one voice.
    assert "The board must CURRENTLY show the STARTING position" in html
    # The node's refusal is surfaced verbatim; success states what the bank now holds.
    assert "reason = (await r.json()).error ?? reason;" in calib
    assert "`bank now holds ${d.exemplars} exemplars`" in calib


def test_camera_page_refuses_to_calibrate_from_a_view_that_is_not_worth_it():
    """Calibrating mid-motion, or with no live camera at all, poisons the bank. The
    button disables itself and says why in its title rather than failing silently."""
    html = camera_html()
    reason = html.split("function calibBlockReason() {")[1].split("\n}")[0]
    assert 'if (!streaming) return "start streaming first' in reason
    assert 'if (pts.length !== 4) return "set the four board corners first";' in reason
    assert "meter.stability < MOVING_BELOW" in reason
    assert "const MOVING_BELOW = 0.7;" in html
    assert "calibBtn.disabled = reason !== null;" in html
    assert "calibBtn.title = reason ??" in html


def test_camera_page_shows_what_the_reader_sees():
    """The phone is the edge AND the operator's only screen, so it renders the position
    coming back — the same vendored chessground, fed from the same viewer socket,
    filtered to this board."""
    html = camera_html()
    assert 'import { Chessground } from "/static/vendor/chessground.min.js";' in html
    assert "/static/vendor/chessground.base.css" in html
    assert "/static/vendor/chessground.cburnett.css" in html
    assert 'orientation: "white", coordinates: true, viewOnly: true,' in html
    # Chessground takes over the element it is handed and sizes itself from its box, so
    # that box must already be square and real when it is created.
    assert ".board-wrap { width: 100%; max-width: 380px; aspect-ratio: 1;" in html
    assert "#board { width: 100%; height: 100%; }" in html
    # One board, set in place; the wire string is sanitized to 8x8 before it gets there.
    assert html.count("Chessground($(\"board\"), {") == 1
    assert "fen: gridToFen(grid)," in html
    assert "if (ev.board_id !== boardId()) return;" in html
    assert "const LOW_CONF = 0.5;" in html
    assert 'autoShapes: low.map((key) => ({ orig: key, brush: "yellow" }))' in html


def test_camera_page_status_strip_states_the_stream():
    html = camera_html()
    for element_id in ("statSent", "statAck", "statFps", "statMode", "statStab",
                       "stabBar", "conn", "banner", "boardId"):
        assert f'id="{element_id}"' in html, element_id
    # (n-1)/span, not n/window — the same honest rate the live page reports.
    assert "const fps = span > 0 ? (n - 1) / span" in html
    assert 'forwarder.bursting ? "burst — a move is under way"' in html


def test_camera_page_tolerates_malformed_placements():
    html = camera_html()
    fn = html.split("function fenToGrid(placement) {")[1].split("\n}\n")[0]
    assert 'String(placement ?? "").split("/")' in fn
    assert 'for (const ch of ranks[r] ?? "") {' in fn
    assert 'while (cells.length < 8) cells.push("");' in fn


def test_camera_page_keeps_no_game_state():
    """DESIGN D1 guard: the page streams frames and renders a placement, nothing more."""
    html = camera_html().lower()
    for forbidden in ("legal", "pgn", "castl", "en passant", "san("):
        assert forbidden not in html, forbidden


def test_camera_page_has_no_emoji():
    """State is carried by words and colour, never by emoji."""
    html = camera_html()
    offenders = [
        c for c in html
        if 0x1F300 <= ord(c) <= 0x1FAFF
        or 0x2600 <= ord(c) <= 0x27BF
        or ord(c) in (0xFE0F, 0x20E3)
        or 0x2B00 <= ord(c) <= 0x2BFF
    ]
    assert not offenders, offenders


def test_camera_page_is_mobile_first():
    html = camera_html()
    assert "viewport-fit=cover" in html
    assert "env(safe-area-inset-bottom)" in html
    assert "min-height: 48px" in html
    assert "playsinline" in html and "muted" in html   # iOS refuses to inline otherwise


def test_startup_prints_the_https_address_to_type_on_the_phone(monkeypatch, tmp_path):
    """The whole point of --ssl is that a phone can reach this page, so startup has to
    hand the operator the exact URL — nobody guesses their own LAN address."""
    monkeypatch.setattr(tls, "local_ipv4_addresses", lambda: ("192.168.1.7",))
    cert = tls.ensure_certificate(tmp_path)
    banner = "\n".join(_banner("0.0.0.0", 8080, cert=cert))
    assert "https://192.168.1.7:8080/camera" in banner
    assert "https://localhost:8080/camera" in banner
    assert str(cert.certfile) in banner
    assert "192.168.1.7" in banner            # what the certificate is valid for
    assert "your phone will warn once" in banner


def test_startup_without_ssl_says_why_the_camera_will_refuse():
    """Plain http is a perfectly good way to run the viewer and a dead end for the
    camera page; the operator should learn that at startup, not from the phone."""
    banner = "\n".join(_banner("127.0.0.1", 8080, cert=None))
    assert "getUserMedia needs a secure context" in banner
    assert "--ssl" in banner
    assert "adb reverse tcp:8080 tcp:8080" in banner
    assert "pass --host 0.0.0.0 to reach it from a phone" in banner
