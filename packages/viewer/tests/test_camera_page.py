"""Tests for the phone camera page — the DESIGN §4.1 "virtual Pi" in a browser.

The page is a single static asset, so these are structural assertions against the
served HTML, in the style of the existing index/snapshot tests: they pin the
behaviours that are load-bearing and easy to lose in a rework (the wire contract, the
edge semantics it mirrors, the touch invariants, the secure-context explainer).
"""

from __future__ import annotations

import re

from starlette.testclient import TestClient

from board_reader_viewer import tls
from board_reader_viewer.server import _banner, create_app


def camera_html() -> str:
    response = TestClient(create_app()).get("/camera")
    assert response.status_code == 200
    return response.text


def _uncommented(source: str) -> str:
    """`source` with its comments stripped.

    The page names several of the anti-patterns these tests assert the absence of —
    `object-fit: contain`, `cg.redrawAll()` — precisely *because* its comments explain
    why it does not use them. A negative assertion therefore has to be made against the
    code, never against the prose.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"^[ \t]*//.*$", "", source, flags=re.M)


def camera_css() -> str:
    return _uncommented(camera_html().split("<style>")[1].split("</style>")[0])


def camera_script() -> str:
    return _uncommented(
        camera_html().split('<script type="module">')[1].split("</script>")[0]
    )


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
    """Half of the old touch contract is unchanged and still load-bearing: only a drag
    that starts ON a handle ever takes the pointer, so the stage and the video never
    capture a pointer and never preventDefault a touch.

    The other half changed with the rework. The stage is a full-screen viewfinder now and
    nothing on the page scrolls, so it is `touch-action: none` rather than the `pan-y` it
    carried inside a card stack — which also stops iOS pull-to-refresh firing over the
    picture. The one surface that does scroll is the sheet, and its scroll stays inside
    it. Handles are the 44px touch minimum: this is an instrument held at arm's length,
    not a thumbnail."""
    html = camera_html()
    assert (".corner-handle { position: absolute; width: 44px; height: 44px; "
            "margin: -22px 0 0 -22px;") in html
    assert ".camera-stage { position: fixed; inset: 0; height: 100dvh; background: #000;" in html
    assert "touch-action: none; overflow: hidden; z-index: 0;" in html
    assert ".camera-stage.placing" not in html            # no state-toggled exception
    handle = html.split("function attachHandleEvents(el, idx) {")[1].split("\n}")[0]
    assert "el.setPointerCapture(ev.pointerId);" in handle
    assert "pointerdown" in handle and "pointermove" in handle and "pointerup" in handle
    # Neither the stage nor the video ever captures a pointer or preventDefaults a touch.
    assert "cameraStage.setPointerCapture" not in html
    assert 'video.addEventListener("pointerdown"' not in html
    # The sheet is the exception, and the only one: it scrolls, and its overscroll does
    # not become the page's.
    sheet = html.split("#sheet { ")[1].split("}")[0]
    assert "overflow-y: auto" in sheet and "overscroll-behavior: contain" in sheet
    assert "touch-action: pan-y" in sheet
    # Handles are positioned in CSS px off the video's rect, so a reflow must re-lay them
    # — and re-fit the picture they are laid out against.
    assert "new ResizeObserver(() => { fitPicture(); draw(); }).observe(cameraStage);" in html


def test_camera_page_is_landscape_first_with_portrait_as_the_override():
    """iOS Safari implements no `screen.orientation.lock()`, so "landscape by default"
    can only ever be a layout default and never a lock: the base stylesheet IS the
    landscape layout and `@media (orientation: portrait)` overrides it.

    The chrome's own thickness is stated once per orientation as a custom property and
    read back by JS, so fitting the picture does not duplicate the layout in script."""
    html = camera_html()
    assert "@media (orientation: portrait) {" in html
    assert "--rail: 150px; --dock: 0px;" in html                     # landscape: a rail
    assert ".camera-stage { --rail: 0px; --dock: 150px; }" in html   # portrait: a dock
    inner = html.split("function stageInner() {")[1].split("\n}")[0]
    assert 'parseFloat(cs.getPropertyValue("--rail"))' in inner
    assert 'parseFloat(cs.getPropertyValue("--dock"))' in inner
    # One DOM order serves both: a right-hand column in landscape, a 3x2 grid dock in
    # portrait. The primary control is last, where the thumb rests.
    assert "grid-template-columns: repeat(3, 1fr);" in html
    rail = html.split('<nav class="rail">')[1].split("</nav>")[0]
    assert re.findall(r'id="(\w+)"', rail) == ["more", "position", "detect", "rotate",
                                               "calib", "stream"]


def test_camera_page_picture_is_a_sized_wrapper_not_a_letterboxed_element():
    """One coordinate space, deliberately. JS gives `.picture` exactly the video's aspect
    ratio inside the stage's inner box, so the element's box IS the picture and
    `rect.width / video.videoWidth` stays a single uniform scale.

    `object-fit: contain` would letterbox INSIDE the element and introduce a second space
    that every touch and every handle would have to be routed through. The one thing that
    must never happen is screen space leaking into `cropBox()`/`sendFrame()`: the preview
    would still look correct while the corners on the wire pointed at the wrong pixels."""
    html = camera_html()
    fit = html.split("function fitPicture() {")[1].split("\n}")[0]
    assert "const k = Math.min(box.w / src.w, box.h / src.h);" in fit
    assert 'picture.style.width = w + "px";' in fit
    assert 'picture.style.height = h + "px";' in fit
    # The video and the overlay fill that box exactly; nothing fits itself inside it.
    assert (".picture video { position: absolute; inset: 0; width: 100%; height: 100%; "
            "display: block; }") in html
    assert 'preserveAspectRatio="none"' in html
    assert "object-fit" not in camera_css()
    # ...so the same single scale converts in both directions, with no letterbox term.
    assert "const scale = video.videoWidth ? rect.width / video.videoWidth : 1;" in html
    assert "const x = (ev.clientX - r.left) * (video.videoWidth / r.width);" in html


def test_camera_page_orientation_is_the_operators():
    """Geometry cannot know which corner is a8 (DESIGN decision #15): detection replaces
    the quad, never the labelling, and rotate labels is how the operator states it."""
    html = camera_html()
    assert 'id="rotate"' in html and "rotate labels" in html
    assert "rot = (rot + 1) % 4;" in html
    assert "function labelOf(i) { return CORNER_ORDER[(i + rot) % 4]; }" in html
    detect = html.split(
        "async function detectCorners({ auto = false } = {}) {")[1].split("\n}\n")[0]
    assert "pts = corners.map(([x, y]) => [Math.round(x / k), Math.round(y / k)]);" in detect
    assert "rot = 0" not in detect                       # a detection never relabels
    assert "geometry cannot tell which corner " in html
    # Detection is offered at a sane size, and its refusal is the node's own words.
    assert "const DETECT_LONG_SIDE = 1600;" in html
    assert "(await r.json()).error ?? reason" in detect


def test_camera_page_looks_for_the_board_by_itself():
    """The common case should need no framing step at all: point the phone at the board
    and the page finds it. Each of the four gates on that path is a measurement rather
    than a preference.

    `MotionMeter.update` returns 1.0 on its very first frame by construction, so a bare
    stability gate passes on frame one — before autoexposure and autofocus have settled,
    spending the attempt on a dark, soft frame. Hence a settle delay, and a floor
    stricter than MOVING_BELOW so a detect frame is exemplar-grade rather than merely
    "not moving". `PlaneTracker.lose()` freezes the quad and keeps tracking from it
    precisely so a hand sweeping over the board recovers by itself, so `lost` fires on
    essentially every move of a real game — hence loss must be SUSTAINED before it counts
    as a reason to re-detect. And `/detect` costs ~0.1 s on a hit but 2-4 s on a MISS,
    because a miss exhausts every coarse candidate — hence retries back off and
    eventually give up rather than poll a page pointed at a wall forever."""
    html = camera_html()
    assert "const AUTO_SETTLE_MS = 2500;" in html
    assert "const AUTO_STABILITY = 0.9;" in html
    assert "const AUTO_LOST_MS = 6000;" in html
    assert "const AUTO_BACKOFF_MS = [4000, 8000, 16000, 32000];" in html
    tick = html.split("function autoDetectTick() {")[1].split("\n}\n")[0]
    # Nothing else may own the camera or the quad, and a hidden page sees nothing.
    assert "if (detectBusy || calibBusy || dragIdx !== null || document.hidden) return;" in tick
    assert "if (!video.videoWidth || autoGaveUp()) return;" in tick
    assert "if (now - cameraLiveAt < AUTO_SETTLE_MS || now < autoNextAt) return;" in tick
    assert "if (meter.stability < AUTO_STABILITY) return;" in tick
    assert 'if (trackState === "lost") { if (lostSince === null) lostSince = now; }' in tick
    assert "const sustainedLoss = lostSince !== null && now - lostSince >= AUTO_LOST_MS;" in tick
    # First sight, or a sustained loss — never a board that is already framed and tracked.
    assert "if (quadKnown && !sustainedLoss) return;" in tick
    assert "cameraLiveAt = performance.now();" in html
    # Every attempt costs the back-off, hit or miss; a hit has already zeroed the count.
    detect = html.split(
        "async function detectCorners({ auto = false } = {}) {")[1].split("\n}\n")[0]
    assert "autoNextAt = performance.now()" in detect
    assert "+ AUTO_BACKOFF_MS[Math.min(autoFails, AUTO_BACKOFF_MS.length - 1)];" in detect
    assert "function autoGaveUp() { return autoFails >= AUTO_BACKOFF_MS.length; }" in html
    # Giving up is not silence: the hint line says so and offers the retry as a tap.
    hint = html.split("function paintHint() {")[1].split("\n}\n")[0]
    assert "if (autoGaveUp()) {" in hint
    assert '{ label: "look again", run: () => detectCorners() });' in hint
    # An automatic miss IS silent, though — four toasts for one wall helps nobody.
    assert 'if (!auto) showBanner("warn", reason' in detect
    # A re-detection keeps `rot`, so it can land the labels on a different physical
    # corner than the operator chose. That is the one case where the page changes a quad
    # the operator authored, so the warning stays up until rotate labels is pressed.
    assert "if (auto && reframing) {" in detect
    assert 'rotateBtn.classList.add("attention");' in detect
    assert "{ sticky: true });" in detect
    assert 'rotateBtn.classList.remove("attention");' in html


def test_camera_page_auto_detection_reuses_the_manual_path():
    """The automatic path calls the same `detectCorners()` the button does, with a flag.

    That is what keeps `test_camera_page_orientation_is_the_operators` honest: it slices
    on `detectCorners` alone to assert that a detection never relabels, so a parallel
    auto implementation would silently stop covering the code that test exists to cover."""
    html = camera_html()
    assert html.count("async function detectCorners") == 1
    assert html.count('fetch("/api/detect"') == 1
    assert "detectCorners({ auto: true });" in html
    assert 'detectBtn.addEventListener("click", () => detectCorners());' in html
    detect = html.split(
        "async function detectCorners({ auto = false } = {}) {")[1].split("\n}\n")[0]
    # The two callers differ only in what they say and what they reset — never in what
    # they do to the quad.
    assert "if (!auto) resetAutoDetect();" in detect
    assert "quadKnown = true;" in detect


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


def test_camera_page_survives_the_frame_changing_shape():
    """Rotating the phone can swap videoWidth/videoHeight under a LIVE track — iOS Safari
    rotates it, Android Chrome usually does not, so a landscape-first page has to handle
    both. Corners are stored as FRACTIONS of the frame, so the same fraction becomes a
    different physical point the moment the aspect flips, and cropBox() would start
    clamping against dimensions its corners were never expressed in."""
    html = camera_html()
    assert "let frameW = 0, frameH = 0;" in html
    resized = html.split('video.addEventListener("resize", () => {')[1].split("\n});")[0]
    assert "if (!w || !h || (w === frameW && h === frameH)) return;" in resized
    assert "pts = pts.map(([x, y]) => [x * w / frameW, y * h / frameH]);" in resized
    assert "saveCorners();" in resized
    # A new frame size is a new tracker, re-anchored on the rescaled quad as it is built.
    assert "tracker = null;" in resized
    assert "ensureTracker();" in resized
    # ...and a new picture to lay the handles out against.
    assert "fitPicture();" in resized and "draw();" in resized
    # Only the operator can confirm the quad still sits on the board, so the page asks.
    assert 'showBanner("warn", "The camera frame changed shape' in resized
    # Opening a lens forgets the old size, so the incoming frame reads as a first
    # sighting rather than as the phone having been rotated mid-stream.
    assert "frameW = 0; frameH = 0;" in html
    assert "frameW = video.videoWidth; frameH = video.videoHeight;" in html


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
    button disables itself and says why rather than failing silently — and it says it in
    BOTH places, because `title` is unreachable on a touch screen and this page runs on
    nothing else. The hint line is the one an operator can actually read.

    The gate is the CAMERA, not the socket: calibration reads the live video element and
    posts it to /api/calibrate, so requiring the ingest stream to be up would be a
    requirement it does not actually have."""
    html = camera_html()
    reason = html.split("function calibBlockReason() {")[1].split("\n}")[0]
    assert 'if (!video.videoWidth) return "the camera is not running";' in reason
    assert "if (!streaming)" not in reason
    assert 'if (pts.length !== 4) return "set the four board corners first";' in reason
    assert "meter.stability < MOVING_BELOW" in reason
    assert "const MOVING_BELOW = 0.7;" in html
    assert "calibBtn.disabled = reason !== null;" in html
    assert "calibBtn.title = reason ??" in html
    hint = html.split("function paintHint() {")[1].split("\n}\n")[0]
    assert ('if (reason !== null) { setHint("calib:" + reason, '
            '"Calibrate is unavailable — " + reason); return; }') in hint
    # Not while streaming: during a game the board is moving half the time, and the
    # sentence would flicker on every move for no gain.
    assert "const reason = streaming ? null : calibBlockReason();" in hint


def test_camera_page_tracks_the_board_plane():
    """DESIGN §4.4.1(a): a bump is recoverable ON THE EDGE. The page tracks the board
    plane frame to frame, so the quad follows the camera instead of waiting for the
    operator — which is also the only reason a walk around the board is possible."""
    html = camera_html()
    assert 'import { PlaneTracker, toGray } from "/static/tracker.js";' in html
    assert "const TRACK_LONG_SIDE = 400;" in html
    assert "const TRACK_SUBDIVISIONS = 1;" in html
    track = html.split("function trackFrame() {")[1].split("\n}\n")[0]
    # The operator's own drag always wins: the tracker never fights for the same corners.
    assert "if (pts.length !== 4 || dragIdx !== null) return;" in track
    assert "const result = t.update(toGray(data, trackGray));" in track
    # Lost -> the quad FREEZES where it last fitted. DESIGN §4.4.1: a mis-localised
    # camera yields confidently wrong crops, which is worse than an honest gap.
    assert 'if (result.state !== "locked") return;' in track
    assert "pts = result.corners.map(([x, y]) => [x * video.videoWidth," in track
    # 20 writes a second to localStorage is not a persistence strategy.
    assert "const TRACK_SAVE_MS = 2000;" in html
    assert "if (now - lastCornerSave > TRACK_SAVE_MS)" in track
    # The state is visible, never silent — a frozen quad looks different from a live one.
    assert 'id="trackChip"' in html
    assert '.quad-line.lost {' in html
    assert 'lost: ["tracking — lost, quad frozen", "bad"],' in html


def test_camera_page_re_anchors_tracking_on_every_deliberate_corner_change():
    """Tracking drifts; the operator is the correction. Every way of stating the quad by
    hand re-seeds the tracker from it, so a drifted crop is one touch from right."""
    html = camera_html()
    assert html.count("anchorTracker();") >= 5      # drag, detect, reset, board switch, toggle
    handle = html.split("function attachHandleEvents(el, idx) {")[1].split("\n}\n")[0]
    assert "anchorTracker();" in handle
    detect = html.split(
        "async function detectCorners({ auto = false } = {}) {")[1].split("\n}\n")[0]
    assert "anchorTracker();" in detect
    anchor = html.split("function anchorTracker() {")[1].split("\n}")[0]
    assert "tracker.reset(pts.map(([x, y]) => [x / video.videoWidth, y / video.videoHeight]));" in anchor
    # Tracking can be switched off outright — a tracker that misbehaves must never be
    # something the operator has to fight.
    assert 'id="trackToggle"' in html
    assert "trackBtn.addEventListener(\"click\", () => setTracking(!trackingOn()));" in html


def test_camera_page_capture_loop_belongs_to_the_camera_not_the_stream():
    """Tracking, the stability reading and automatic detection are live instruments in
    their own right, so the loop starts when the camera does — the stream is one consumer
    of it, and the only thing streaming decides is whether a frame is forwarded."""
    html = camera_html()
    assert "frameTimer = setInterval(frameTick, Math.round(1000 / CAPTURE_FPS));" in html
    loop = html.split("function frameTick() {")[1].split("\n}\n")[0]
    assert "trackFrame();" in loop
    assert "autoDetectTick();" in loop
    # Not streaming still has to feed the meter: `calibBlockReason` and the automatic
    # detector both read stability, and a meter fed only while streaming would report a
    # stale 1.0.
    assert "(meter.update(video), null)" in loop
    assert "const decision = streaming" in loop
    assert "? forwarder.decide(video, { boardId: boardId(), captureTs })" in loop
    started = html.split('video.addEventListener("loadedmetadata", () => {')[1].split("\n}")[0]
    assert "startFrameLoop();" in started
    assert "ensureTracker();" in started
    # Stopping the stream must not stop tracking.
    stop = html.split("function stopStreaming() {")[1].split("\n}\n")[0]
    assert "frameTimer" not in stop


def test_camera_page_lets_the_operator_choose_the_lens():
    """iOS >= 16.3 exposes every rear lens, but nothing identifies which is the ultra
    wide: deviceIds are per-session, labels are localised, order is not guaranteed. So
    the operator picks — and the pick is remembered by LABEL, the only stable property."""
    html = camera_html()
    assert 'id="lens"' in html and 'id="lensBox"' in html
    assert "ultra wide" in html                      # the hint names what to look for
    assert "barrel distortion" in html               # ...and what it costs
    open_camera = html.split("async function openCamera(deviceId) {")[1].split("\n}\n")[0]
    # iOS hands out one camera at a time: the outgoing tracks must be stopped first.
    assert "for (const track of cameraTracks) track.stop();" in open_camera
    assert "deviceId: { exact: deviceId }" in open_camera
    assert "tracker = null;" in open_camera          # new lens, new frame size, new view
    assert 'localStorage.setItem(LENS_KEY, label)' in html
    start = html.split("async function startCamera() {")[1].split("\n}\n")[0]
    # Labels only exist after permission, so the remembered lens is a second pass.
    assert "const lenses = await refreshLenses();" in start
    assert "lenses.find((lens) => lens.label === saved)" in start
    # A single-camera device gets no picker at all: a select with one option is furniture.
    assert '$("lensBox").hidden = lenses.length < 2;' in html


def test_camera_page_shows_what_the_reader_sees():
    """The phone is the edge AND the operator's only screen, so it renders the position
    coming back — the same vendored chessground, fed from the same viewer socket,
    filtered to this board."""
    html = camera_html()
    assert 'import { Chessground } from "/static/vendor/chessground.min.js";' in html
    assert "/static/vendor/chessground.base.css" in html
    assert "/static/vendor/chessground.cburnett.css" in html
    assert 'orientation: "white", coordinates: true, viewOnly: true,' in html
    # `aspect-ratio: 1` is LOAD-BEARING, not cosmetic: chessground derives its
    # container's height from this wrap's own aspect ratio, so a wrap that is not square
    # renders non-square squares and stretched pieces. The panel sizes the wrap; the wrap
    # keeps it square.
    assert ".board-wrap { width: 100%; aspect-ratio: 1;" in html
    assert "#board { width: 100%; height: 100%; }" in html
    # One board, set in place; the wire string is sanitized to 8x8 before it gets there.
    assert html.count("Chessground($(\"board\"), {") == 1
    assert "fen: gridToFen(grid)," in html
    assert "if (ev.board_id !== boardId()) return;" in html
    assert "const LOW_CONF = 0.5;" in html
    assert 'autoShapes: low.map((key) => ({ orig: key, brush: "yellow" }))' in html


def test_camera_page_position_panel_lets_chessground_size_itself():
    """chessground takes over the element it is handed, sizes itself from that element's
    box, and re-measures through its OWN ResizeObserver. So the panel must never take the
    box away, and must never rebuild the board.

    `display: none` here is not merely worse than `visibility: hidden`, it is broken: a
    0x0 box makes chessground's own observer fire, and it then computes the container
    size as 0/0 and writes `NaNpx`. `cg.redrawAll()` is the other trap — it re-runs the
    full teardown (`innerHTML = ""`) and re-observes WITHOUT disconnecting the previous
    observer, i.e. one leaked observer and one rebuild per call, which wired to a drag
    would be per pointermove. The panel calls nothing on resize; chessground heals
    itself."""
    html = camera_html()
    assert ".pos-panel.hidden { visibility: hidden; pointer-events: none; }" in html
    assert 'posPanel.classList.toggle("hidden", !posState.shown);' in html
    assert "posPanel.style.display" not in html
    assert "redrawAll" not in camera_script()
    # ...and it is a SIBLING of the stage, not a child of it, for the same reason: with no
    # camera the stage is display: none'd outright, which would take the board's box with
    # it if the board lived inside.
    stage = html.split('<div class="camera-stage" id="stage">')[1].split("\n</div>")[0]
    assert 'id="posPanel"' not in stage
    assert '$("stage").style.display = "none";' in html
    # One degree of freedom, on purpose: the grip writes the WIDTH, and .board-wrap's
    # aspect ratio supplies the height.
    grip = html.split('posGrip.addEventListener("pointermove", (ev) => {')[1].split("\n});")[0]
    assert "posState.width = posResize.width + (ev.clientX - posResize.x);" in grip
    assert "applyPos();" in grip
    assert "posState.height" not in html
    # Geometry survives a reload and is clamped back into the viewport every time: a
    # panel parked against the edge of a landscape phone would otherwise be off-screen in
    # portrait, with no way to reach its own drag strip.
    assert 'const POS_KEY = "board-reader.camera.position";' in html
    apply_pos = html.split("function applyPos() {")[1].split("\n}")[0]
    assert "Math.round(Math.max(POS_MIN, Math.min(POS_MAX, posState.width)))" in apply_pos
    assert "posState.left = Math.max(4, Math.min(window.innerWidth - width - 4, left));" in apply_pos
    assert "posState.top = Math.max(4, Math.min(window.innerHeight - height - 4, top));" in apply_pos
    assert "localStorage.setItem(POS_KEY, JSON.stringify(posState));" in html
    assert 'window.addEventListener("resize", () => { fitPicture(); draw(); applyPos(); });' in html
    # And the box is real BEFORE chessground is ever handed the element.
    assert html.rindex("applyPos();") < html.index('Chessground($("board"), {')


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
