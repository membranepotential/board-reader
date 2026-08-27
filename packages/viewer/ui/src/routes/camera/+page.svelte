<script lang="ts">
  /**
   * /camera — the DESIGN §4.1 "virtual Pi": a phone held over a board, doing on the
   * browser's terms what the edge node does on the Pi's (measure motion, forward
   * adaptively, stamp capture time) and streaming the result into the inference
   * node.
   *
   * It is a VIEWFINDER, not a settings page. The picture gets the screen, a short
   * column of translucent pills sits beside it, one sentence of context sits under
   * it, and everything else lives in a sheet that is closed by default. It is used
   * one-handed, at arm's length, pointed at a physical board; it must be usable
   * without reading anything. No emoji anywhere: state is text + colour.
   *
   * `Viewfinder.svelte` owns the layout and its header is the one place it is
   * described — one fixed box, safe-area insets taken once as padding, and a single
   * flex line whose direction is the only difference between the two orientations.
   *
   * Everything it talks to is same-origin and relative (`$lib/api`), relayed by the
   * viewer. That is deliberate: `getUserMedia` needs a secure context, and one
   * origin means the operator accepts one certificate on the phone instead of two.
   *
   * This file is the imperative shell and nothing else. The state lives in
   * `$lib/components/camera/state` as three unions, the mathematics lives in
   * `$lib/geometry` and `$lib/edge`, and the chrome lives in the components beside
   * it. What is left here is capture, the one frame loop, and the four requests.
   *
   * PRERENDERED: nothing below may touch a browser API at module scope or during
   * SSR. Every `getUserMedia`, `localStorage`, `WakeLock` and `location` read is
   * inside `onMount` or an event handler.
   */
  import { onMount } from 'svelte';
  import type { DrawShape } from '@lichess-org/chessground/draw';

  import { calibrate as postCalibrate, detect as postDetect } from '$lib/api';
  import { gridToFen } from '$lib/board/fen';
  import {
    checkQuadShape,
    absorbCalibrateRotation,
    cornersInCrop,
    cropBox,
    nextRotation,
    orderedCorners,
    type FrameSize
  } from '$lib/geometry/corners';
  import { PlaneTracker, toGray } from '$lib/geometry/tracker';
  import {
    AdaptiveForwarder,
    CAPTURE_FPS,
    MotionMeter,
    MOTION_DEFAULTS,
    sampleThumbnail,
    type ForwardDecision
  } from '$lib/edge/motion';
  import { IngestClient, type IngestStatus } from '$lib/edge/ingest';
  import { tenengrad } from '$lib/edge/sharpness';
  import { TrialLog } from '$lib/edge/triallog';
  import { Heartbeat } from '$lib/edge/heartbeat';
  import { EMPTY_PLACEMENT, type CornerQuad, type Point } from '$lib/protocol';
  import { PositionStream } from '$lib/stores/positions.svelte';
  import { pendingShapes } from '$lib/components/live/shapes';

  import Banner from '$lib/components/camera/Banner.svelte';
  import BlockedNotice from '$lib/components/camera/BlockedNotice.svelte';
  import HintBar from '$lib/components/camera/HintBar.svelte';
  import PositionPanel from '$lib/components/camera/PositionPanel.svelte';
  import Rail from '$lib/components/camera/Rail.svelte';
  import SettingsSheet from '$lib/components/camera/SettingsSheet.svelte';
  import Viewfinder from '$lib/components/camera/Viewfinder.svelte';
  import {
    AUTO_DETECT_START,
    BANNER_MS,
    BOARD_KEY,
    CALIBRATE_QUALITY,
    CALIBRATE_SAMPLES,
    CALIBRATE_SAMPLE_MS,
    CALIBRATE_SCORE_LONG_SIDE,
    DETECT_LONG_SIDE,
    JPEG_QUALITY,
    LENS_KEY,
    PANEL_KEY,
    PANEL_START,
    SEND_LONG_SIDE,
    TICK_MS,
    TRACKING_KEY,
    TRACK_ITERATIONS,
    TRACK_LONG_SIDE,
    TRACK_SAVE_MS,
    TRACK_SUBDIVISIONS,
    backoffFor,
    AUTO_CALIBRATE_COOLDOWN_MS,
    AUTO_CALIBRATE_START,
    calibrateBlockReason,
    cornerKey,
    denormaliseQuad,
    freshFrameClock,
    hintFor,
    isFramed,
    normaliseQuad,
    noteFrameClock,
    noteTracking,
    parseStoredFraming,
    parseStoredPanel,
    placeholderQuad,
    ratePerSecond,
    rebaseQuad,
    serialiseFraming,
    QUAD_VERIFY_SHIFT,
    quadShift,
    REDETECT_START,
    noteRedetectMiss,
    noteRedetection,
    shouldRedetect,
    shouldAutoCalibrate,
    shouldAutoDetect,
    toQuad,
    trackingChip,
    withinRateWindow,
    type Activity,
    type AutoCalibrate,
    type AutoDetect,
    type BannerKind,
    type BannerMessage,
    type CameraState,
    type CornerIndex,
    type Framing,
    type Lens,
    type PanelState,
    type Redetect,
    type TrackingState
  } from '$lib/components/camera/state';

  // ---- the state ---------------------------------------------------------------
  // `$state.raw` throughout for the hot values: every update below replaces the
  // whole object, and the quad is handed straight to the tracker and the wire, so
  // it must stay a plain array rather than a reactive proxy.

  let camera = $state.raw<CameraState>({ kind: 'idle' });
  let activity = $state.raw<Activity>({ kind: 'idle' });
  let auto = $state.raw<AutoDetect>(AUTO_DETECT_START);
  let autoCalib = $state.raw<AutoCalibrate>(AUTO_CALIBRATE_START);
  let redetect = $state.raw<Redetect>(REDETECT_START);

  let tracking = $state(true);
  let trackingState = $state<TrackingState>('off');
  let stability = $state(1);
  let bursting = $state(false);

  let streaming = $state(false);
  let connStatus = $state<IngestStatus>('idle');
  let connDetail = $state('idle');
  let sentCount = $state(0);
  let ackedSeq = $state<number | null>(null);
  let droppedCount = $state(0);
  let fps = $state(0);

  let boardId = $state('phone');
  let calibrations = $state(0);
  let labelsNeedCheck = $state(false);
  let banner = $state.raw<BannerMessage | null>(null);
  let sheetOpen = $state(false);
  let lenses = $state.raw<readonly Lens[]>([]);
  let lensId = $state<string | null>(null);
  let panel = $state.raw<PanelState>(PANEL_START);
  let video = $state<HTMLVideoElement | undefined>();
  // The 2026-08-27 hard block: crops extend toward the top of the image, so a
  // portrait frame reads every piece into its neighbour. Screen orientation is
  // the early signal (before the camera even opens); the frame's own aspect is
  // the ground truth for what would actually be sent.
  let screenPortrait = $state(false);
  let port = $state('8080');

  // ---- the plumbing, deliberately NOT reactive ---------------------------------
  // None of this is rendered; making it reactive would only cost a proxy on the
  // hot path.

  const meter = new MotionMeter();
  const forwarder = new AdaptiveForwarder(meter);
  const positions = new PositionStream();
  // The phone's half of a trial's story — every lifecycle event worth a
  // post-mortem goes through `trial.note`, batched to the viewer's log sink.
  const trial = new TrialLog();
  // A quiet stretch in an event-driven log is ambiguous — a phone sending happily
  // and a phone sending nothing both look like silence. The heartbeat samples the
  // ingest counters on a timer while streaming, so the log always says which.
  const heartbeat = new Heartbeat(trial, () => ({
    sent: ingest.sent,
    dropped: ingest.dropped,
    acked: ingest.acked,
    connected: ingest.connected,
    behind: ingest.behind
  }));
  let lastRefusal: string | null = null;
  const ingest = new IngestClient({
    onStatus: (status, detail) => {
      connStatus = status;
      connDetail = detail;
      trial.note('ingest-status', { status, detail });
    },
    onAck: () => {
      ackedSeq = ingest.acked;
    },
    // One bad frame is refused in the ack slot and the socket stays up. Said once
    // per distinct reason: at the burst rate a per-frame toast would rewrite the
    // banner twelve times a second and read as a flicker.
    onFrameRefused: (reason) => {
      if (reason === lastRefusal) return;
      lastRefusal = reason;
      trial.note('frame-refused', { reason });
      setBanner('warn', `the node refused a frame: ${reason}`);
    }
  });

  const MOTION_SIDE = MOTION_DEFAULTS.downscale;

  let stream: MediaStream | null = null;
  // Focus score of the last tracked frame, measured on the tracker's own gray
  // buffer (one extra pass, no extra readback). Rides each streamed frame so the
  // node can spend its read on the crisp frame of a shaky second.
  let lastSharpness: number | null = null;
  let tracker: PlaneTracker | null = null;
  let trackGray: Float32Array | null = null;
  let motionCanvas: HTMLCanvasElement | null = null;
  let motionCtx: CanvasRenderingContext2D | null = null;
  let trackCanvas: HTMLCanvasElement | null = null;
  let trackCtx: CanvasRenderingContext2D | null = null;
  let sendCanvas: HTMLCanvasElement | null = null;
  let sendCtx: CanvasRenderingContext2D | null = null;
  let wakeLock: WakeLockSentinel | null = null;

  /** The frame size the current corners are expressed against — see `onVideoResize`. */
  let lastFrame: FrameSize | null = null;
  /** The stall watchdog's whole memory — see `state.ts`, "camera stall watchdog". */
  let frameClock = freshFrameClock(0);
  let encoding = false;
  let lastCornerSave = 0;
  let stamps: number[] = [];
  let frameTimer: ReturnType<typeof setInterval> | null = null;
  let statusTimer: ReturnType<typeof setInterval> | null = null;
  let bannerTimer: ReturnType<typeof setTimeout> | null = null;

  // ---- derived ------------------------------------------------------------------

  const portrait = $derived(
    screenPortrait || (camera.kind === 'live' && camera.frame.height > camera.frame.width)
  );
  const calibrateBlock = $derived(calibrateBlockReason({ camera, activity, portrait }));
  const hint = $derived(
    hintFor({ camera, activity, auto, tracking: trackingState, streaming, calibrateBlock })
  );
  const trackChip = $derived(trackingChip(trackingState));
  $effect(() => {
    trial.note('portrait', { blocked: portrait });
  });
  const dragging = $derived(activity.kind === 'dragging' ? activity.corner : null);
  const connTone = $derived(
    connStatus === 'streaming' ? 'up' : connStatus === 'idle' ? '' : 'warn'
  );

  const board = $derived(positions.board(boardId));
  const placement = $derived(board === undefined ? EMPTY_PLACEMENT : gridToFen(board.grid));
  // Yellow: low confidence. Red: the prior decoder relabelled this square to make
  // the board possible — the drawn piece is the repair, not the raw read. Dashed
  // amber: the settled board hasn't adopted this square yet.
  const shapes = $derived<DrawShape[]>(
    board === undefined
      ? []
      : [
          ...board.low.map((square): DrawShape => ({ orig: square, brush: 'yellow' })),
          ...board.flipped.map((square): DrawShape => ({ orig: square, brush: 'red' })),
          ...pendingShapes(board.pending)
        ]
  );
  const panelStatus = $derived.by(() => {
    if (board === undefined) return 'waiting…';
    const confidence = board.event.confidence;
    const suffix = confidence === null ? '' : ` · confidence ${confidence.toFixed(2)}`;
    const repaired = board.flipped.length === 0 ? '' : ` · repaired ${board.flipped.length}`;
    return `${board.event.status} · seq ${board.event.seq}${suffix}${repaired}`;
  });

  // ---- little helpers -----------------------------------------------------------

  function readLocal(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null; // private mode: preferences simply do not persist
    }
  }

  function writeLocal(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private mode */
    }
  }

  function toBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob | null> {
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
  }

  /**
   * `sticky` is for the one message the operator must act on rather than watch go
   * by. One timer, replaced rather than stacked.
   */
  function setBanner(kind: BannerKind, text: string, sticky = false): void {
    banner = { kind, text, sticky };
    if (bannerTimer !== null) clearTimeout(bannerTimer);
    bannerTimer = sticky
      ? null
      : setTimeout(() => {
          banner = null;
        }, BANNER_MS);
  }

  function clearBanner(): void {
    if (bannerTimer !== null) clearTimeout(bannerTimer);
    bannerTimer = null;
    banner = null;
  }

  // ---- framing ------------------------------------------------------------------

  function setFraming(framing: Framing): void {
    if (camera.kind !== 'live') return;
    camera = { ...camera, framing };
  }

  function saveFraming(): void {
    if (camera.kind !== 'live') return;
    writeLocal(cornerKey(boardId), serialiseFraming(camera.framing, camera.frame));
  }

  /** The placeholder: not a framing, so the page goes looking. */
  function unframed(frame: FrameSize): Framing {
    return { quad: placeholderQuad(frame), rot: 0, origin: 'placeholder' };
  }

  // ---- board tracking (DESIGN §4.4.1(a), "bump / drift") ------------------------
  // The quad is not a fixed crop. A phone gets nudged, and it is carried around the
  // table to calibrate from a second angle — so the corners follow the board plane
  // frame to frame instead of waiting for the operator to drag them back.
  //
  // Tracking rather than re-detecting, specifically: geometry cannot say which
  // corner is a8 (DESIGN decision #15), so a fresh detection returns an UNLABELLED
  // quad and turns a bump into a §4.4.1(b) reposition. A tracked quad carries its
  // labels with it.

  /** Sized from the live frame, so a lens switch (a different resolution) rebuilds it. */
  function ensureTracker(frame: FrameSize): PlaneTracker | null {
    if (trackCanvas === null || frame.width === 0 || frame.height === 0) return null;
    const k = Math.min(1, TRACK_LONG_SIDE / Math.max(frame.width, frame.height));
    const width = Math.max(1, Math.round(frame.width * k));
    const height = Math.max(1, Math.round(frame.height * k));
    if (tracker === null || tracker.width !== width || tracker.height !== height) {
      trackCanvas.width = width;
      trackCanvas.height = height;
      trackGray = new Float32Array(width * height);
      tracker = new PlaneTracker({
        width,
        height,
        subdivisions: TRACK_SUBDIVISIONS,
        lk: { iterations: TRACK_ITERATIONS }
      });
      anchorTracker();
    }
    return tracker;
  }

  /**
   * Re-anchor on the operator's quad. Every deliberate corner change routes through
   * here — a handle drag, a detection, a reset, a board switch — so the operator
   * always wins against the tracker, and a drifted quad is one touch from correct.
   */
  function anchorTracker(): void {
    if (tracker === null || camera.kind !== 'live') return;
    tracker.reset(normaliseQuad(camera.framing.quad, camera.frame));
    trackingState = 'seeded';
  }

  function trackFrame(frame: FrameSize): void {
    if (!tracking) {
      trackingState = 'off';
      lastSharpness = null; // nothing measures it with the tracker off
      return;
    }
    // A drag in progress is the operator moving the corners; the tracker must not
    // fight for the same four numbers.
    if (activity.kind === 'dragging') return;
    const active = ensureTracker(frame);
    if (active === null || trackCtx === null || trackCanvas === null) return;
    if (trackGray === null || video === undefined) return;

    trackCtx.drawImage(video, 0, 0, trackCanvas.width, trackCanvas.height);
    const { data } = trackCtx.getImageData(0, 0, trackCanvas.width, trackCanvas.height);
    // `trackGray` is reused on purpose AND safely: `buildPyramid` copies level 0
    // (`gray.slice()`), so the tracker never holds this buffer as its previous
    // frame. Handing a pyramid a buffer the page then overwrites is the recorded
    // aliasing bug — every point measured itself, converged on zero displacement,
    // and the tracker reported a confident lock on a quad that never moved.
    const gray = toGray(data, trackGray);
    lastSharpness = tenengrad({ data: gray, width: trackCanvas.width, height: trackCanvas.height });
    const report = active.update(gray);
    trackingState = report.state;

    // Lost: the quad stays exactly where it last fitted. DESIGN §4.4.1 — a
    // mis-localised camera produces confidently wrong crops, which is worse than an
    // honest gap.
    if (report.state !== 'locked' || report.corners === null) return;
    const moved = toQuad(report.corners);
    if (moved === null || camera.kind !== 'live') return;
    setFraming({ ...camera.framing, quad: denormaliseQuad(moved, frame) });

    // Persisted on a timer, not per frame: this runs 20 times a second.
    const now = performance.now();
    if (now - lastCornerSave > TRACK_SAVE_MS) {
      lastCornerSave = now;
      saveFraming();
    }
  }

  // ---- the one loop -------------------------------------------------------------
  // It runs from the moment the camera is live rather than from the moment
  // streaming starts, because tracking, the stability reading and automatic
  // detection are all live instruments in their own right — the stream is only one
  // of its consumers.

  function grabThumbnail(): Float32Array | null {
    if (motionCtx === null || video === undefined) return null;
    // Sampled from the BOARD's crop once framed, the whole frame before: motion
    // on the full frame is diluted by everything that is not the board (resting
    // hands, background, exposure hunting over the scene), which is a large part
    // of why stability lived under 0.35 on the 2026-08-27 takes. The meter's
    // adaptive floor absorbs the one-frame spike when the region switches.
    if (camera.kind === 'live' && isFramed(camera.framing)) {
      const { framing, frame } = camera;
      const box = cropBox(orderedCorners(framing.quad, framing.rot), frame);
      motionCtx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, MOTION_SIDE, MOTION_SIDE);
    } else {
      motionCtx.drawImage(video, 0, 0, MOTION_SIDE, MOTION_SIDE);
    }
    const { data } = motionCtx.getImageData(0, 0, MOTION_SIDE, MOTION_SIDE);
    // A fresh array per frame: `MotionMeter` RETAINS what it is given as the next
    // frame's reference, so a reused buffer would difference every frame against
    // itself and read as perfectly still.
    return sampleThumbnail(data, MOTION_SIDE * MOTION_SIDE);
  }

  function frameTick(): void {
    if (camera.kind !== 'live') return;
    const frame = camera.frame;
    const captureTs = Date.now() / 1000;

    watchFrameClock();
    trackFrame(frame);

    const thumbnail = grabThumbnail();
    if (thumbnail === null) return;
    // The meter has to see EVERY sampled frame — dropping frames upstream would
    // make stability lie — so it is updated whether or not anything is streaming.
    // A portrait frame is metered but never forwarded — every crop the node
    // would cut from it reaches into the neighbouring square. The overlay says
    // why, and sending resumes by itself the moment the phone turns.
    const decision =
      streaming && !portrait
        ? forwarder.decide(thumbnail, { boardId, captureTs })
        : (meter.update(thumbnail), null);
    stability = meter.stability;
    bursting = forwarder.bursting;

    autoDetectTick();
    autoCalibTick();
    redetectTick();

    if (decision === null) return;
    // Calibration is reading the live camera and rebuilding the node's bank from
    // one frame; forwarding into a half-rebuilt bank produces readings nobody can
    // interpret, so the stream pauses (metered, not blind) until it is done.
    if (activity.kind === 'calibrating') return;
    if (!ingest.connected) return;
    // A frame still in the encoder means another frame would only make the reading
    // staler. Dropping is the correct answer at both rates (DESIGN D3/D4).
    if (encoding) {
      ingest.noteDropped();
      return;
    }
    // Same reasoning one step earlier: if the link is already behind, this frame is
    // going to be refused, so skip the canvas readback and the JPEG compression rather
    // than paying for them and throwing the result away. `sendFrame` still re-checks —
    // the buffer can fill during an encode that started while the link was clear.
    if (ingest.behind) {
      ingest.noteDropped();
      return;
    }
    encoding = true;
    void sendFrame(decision, captureTs).finally(() => {
      encoding = false;
    });
  }

  async function sendFrame(decision: ForwardDecision, captureTs: number): Promise<void> {
    if (camera.kind !== 'live' || video === undefined) return;
    if (sendCanvas === null || sendCtx === null) return;
    const { frame, framing } = camera;
    const ordered = orderedCorners(framing.quad, framing.rot);
    const box = cropBox(ordered, frame);
    const k = Math.min(1, SEND_LONG_SIDE / Math.max(box.w, box.h));
    const w = Math.max(1, Math.round(box.w * k));
    const h = Math.max(1, Math.round(box.h * k));
    sendCanvas.width = w;
    sendCanvas.height = h;
    sendCtx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, w, h);

    // The quad rebased into the pixels of the image actually being sent. Computed
    // BEFORE the encode await, beside the draw that fixed the pixels — a handle
    // dragged during encoding must not pair new corners with old pixels.
    const corners = cornersInCrop(ordered, box, { w, h });

    const jpeg = await toBlob(sendCanvas, JPEG_QUALITY);
    if (jpeg === null) return;
    const outcome = ingest.sendFrame({
      boardId,
      captureTs,
      stability: decision.stability,
      ...(lastSharpness === null ? {} : { sharpness: lastSharpness }),
      burstId: decision.burstId,
      corners,
      jpeg
    });
    if (outcome === 'sent') stamps.push(performance.now());
  }

  function startLoop(): void {
    if (frameTimer !== null) return;
    // The camera is sampled faster than anything is ever forwarded: the sampling
    // rate must comfortably exceed the burst rate or a "12 fps burst" is really
    // whatever the sampler managed.
    frameTimer = setInterval(frameTick, Math.round(1000 / CAPTURE_FPS));
  }

  function paintStatus(): void {
    const now = performance.now();
    stamps = withinRateWindow(stamps, now);
    fps = ratePerSecond(stamps, streaming);
    sentCount = ingest.sent;
    ackedSeq = ingest.acked;
    droppedCount = ingest.dropped;
  }

  // ---- camera stall recovery ------------------------------------------------------
  // A phone camera dies quietly: iOS mutes the track when the page is backgrounded
  // or the camera is taken by another app, and sometimes never revives it — the
  // preview then shows one frame forever while the loop streams it. Three layers,
  // all funnelling into `recoverCamera`: track events (the polite notification),
  // the visibility/pageshow handlers (the common cause), and the `video.currentTime`
  // watchdog (the backstop that needs no notification at all).

  function watchFrameClock(): void {
    if (video === undefined) return;
    // A calibration burst owns the camera; yanking the stream out from under it
    // would pair its frames with a dead device. The stall outlives the burst, so
    // the watchdog simply fires afterwards.
    if (activity.kind === 'calibrating') return;
    const { clock, action } = noteFrameClock(
      frameClock,
      video.currentTime,
      performance.now(),
      document.hidden
    );
    frameClock = clock;
    if (action === 'play') {
      trial.note('camera-stall-play');
      void video.play().catch(() => {});
    } else if (action === 'reopen') {
      trial.note('camera-stall-reopen', { attempt: clock.reopens });
      setBanner('warn', 'The camera stalled — restarting it…');
      void recoverCamera();
    }
  }

  /**
   * Reopen the same lens if it is still there, any camera if not. The framing is
   * untouched — corners belong to the board, and `onLoadedMetadata` restores them
   * for the incoming stream.
   *
   * One recovery at a time: the case being recovered from is exactly the one
   * where `getUserMedia` can block past the watchdog's next firing (a contended
   * device), and two interleaved recoveries end with one stream assigned and the
   * other's tracks never stopped — a camera held claimed for the session.
   */
  let recovering = false;
  async function recoverCamera(): Promise<void> {
    if (recovering) return;
    if (camera.kind !== 'live') return; // blocked/idle: `startCamera` owns bring-up
    recovering = true;
    try {
      await openCamera(currentDeviceId());
      trial.note('camera-recovered');
    } catch {
      try {
        await openCamera(null);
        trial.note('camera-recovered', { lens: 'fallback' });
      } catch (cause) {
        trial.note('camera-recover-failed', {
          name: cause instanceof Error ? cause.name : String(cause)
        });
        setBanner('err', 'The camera did not come back — reload the page or check the lens.');
      }
    } finally {
      recovering = false;
    }
  }

  // ---- streaming ----------------------------------------------------------------

  async function acquireWakeLock(): Promise<void> {
    try {
      wakeLock = (await navigator.wakeLock?.request('screen')) ?? null;
    } catch {
      wakeLock = null; // denied or unsupported: the screen may sleep, that is all
    }
  }

  function releaseWakeLock(): void {
    void wakeLock?.release().catch(() => {});
    wakeLock = null;
  }

  function toggleStreaming(): void {
    if (streaming) {
      streaming = false;
      ingest.stop();
      heartbeat.stop();
      releaseWakeLock();
      trial.note('streaming-off', { sent: ingest.sent, acked: ingest.acked, dropped: ingest.dropped });
      return;
    }
    if (portrait) return; // the start pill is disabled; this enforces it
    streaming = true;
    forwarder.reset();
    stamps = [];
    ingest.start();
    heartbeat.start();
    void acquireWakeLock();
    trial.note('streaming-on');
  }

  // ---- detect -------------------------------------------------------------------

  function autoDetectTick(): void {
    const now = performance.now();
    auto = noteTracking(auto, trackingState, now);
    if (!shouldAutoDetect({ auto, now, camera, activity, hidden: document.hidden, stability })) {
      return;
    }
    void detectCorners(true);
  }

  /** The zero-tap flow's second half: see `shouldAutoCalibrate` in state.ts. */
  function autoCalibTick(): void {
    const gate = {
      auto: autoCalib,
      now: performance.now(),
      camera,
      activity,
      portrait,
      streaming,
      stability,
      bursting,
      calibrations,
      grid: board?.grid ?? null
    };
    if (!shouldAutoCalibrate(gate)) return;
    autoCalib = { nextAt: gate.now + AUTO_CALIBRATE_COOLDOWN_MS, attempts: autoCalib.attempts + 1 };
    trial.note('auto-calibrate', { attempt: autoCalib.attempts });
    void runCalibrate(true);
  }

  /** What one `/detect` round trip came back with. */
  type DetectOutcome =
    | { readonly kind: 'quad'; readonly quad: CornerQuad }
    /** Every kind of miss, with the sentence the operator would be shown. */
    | { readonly kind: 'none'; readonly advice: string };

  /**
   * One `/detect` round trip: the live frame at `DETECT_LONG_SIDE`, the reply
   * unscaled back into frame pixels, and the shape check the node's quad has to
   * pass before anything is done with it.
   *
   * Shared by all three callers — the operator's "look again", the automatic first
   * framing, and the periodic drift check — so all three see the same picture at
   * the same resolution and judge the answer by the same rules.
   */
  async function detectQuad(frame: FrameSize): Promise<DetectOutcome> {
    if (video === undefined) return { kind: 'none', advice: 'the camera is not running' };
    const k = Math.min(1, DETECT_LONG_SIDE / Math.max(frame.width, frame.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(frame.width * k);
    canvas.height = Math.round(frame.height * k);
    canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height);
    const jpeg = await toBlob(canvas, 0.9);
    if (jpeg === null) {
      return { kind: 'none', advice: 'the browser produced no image to look at' };
    }
    const result = await postDetect(jpeg);
    if (!result.ok) {
      return {
        kind: 'none',
        advice: `${result.error.message} — drag the handles onto the board instead`
      };
    }
    const proposed = result.value.corners;
    const at = (p: Point): Point => [Math.round(p[0] / k), Math.round(p[1] / k)];
    const quad: CornerQuad = [at(proposed[0]), at(proposed[1]), at(proposed[2]), at(proposed[3])];
    const defect = checkQuadShape(quad);
    if (defect !== null) {
      return { kind: 'none', advice: `the node's quad ${defect.message} — set the corners by hand` };
    }
    return { kind: 'quad', quad };
  }

  /**
   * Geometry cannot know which corner is a8 (DESIGN decision #15), so a detection
   * replaces the quad but never the labelling — that stays the operator's, via
   * rotate labels.
   */
  async function detectCorners(isAuto = false): Promise<void> {
    if (activity.kind !== 'idle' || camera.kind !== 'live' || video === undefined) return;
    const frame = camera.frame;
    const reframing = isFramed(camera.framing);
    // The operator asking is worth a fresh run of chances, whatever the automatic
    // path has concluded about this scene.
    if (!isAuto) auto = AUTO_DETECT_START;
    let fails = auto.fails;
    activity = { kind: 'detecting', auto: isAuto };
    try {
      const outcome = await detectQuad(frame);
      if (outcome.kind === 'none') {
        fails++;
        // An automatic miss is silent: the hint line already says what is
        // happening, and a toast per attempt would be four toasts for one wall.
        if (!isAuto) setBanner('warn', outcome.advice);
        return;
      }
      const quad = outcome.quad;

      fails = 0;
      if (camera.kind !== 'live') return;
      auto = { ...auto, lostSince: null };
      const previousQuad = camera.framing.quad;
      setFraming({ quad, rot: camera.framing.rot, origin: 'detected' });
      saveFraming();
      anchorTracker();
      // The framing has been re-authored, and by the detector's own convention:
      // the drift baseline measured against the previous one is meaningless now.
      redetect = REDETECT_START;

      // A detection never touches `rot` (decision #15), so a re-detection after a
      // sustained loss can land the labels on a DIFFERENT physical corner than the
      // operator chose. That is the one case where the page changes a quad the
      // operator authored, so the warning stays up until rotate labels is pressed.
      // EXCEPT when the new quad landed on the old one point-for-point — then the
      // labels cannot have moved, and a verify of a restored framing that changed
      // nothing must not nag (it now runs on every page open).
      const shifted = quadShift(previousQuad, quad, frame) > QUAD_VERIFY_SHIFT;
      if (isAuto && reframing && !shifted) {
        trial.note('framing-verified', { moved: false });
      } else if (isAuto && reframing) {
        labelsNeedCheck = true;
        setBanner(
          'warn',
          'The board was found again after being lost. The labels may now sit on a ' +
            'different corner — press rotate labels until a8 is where you call a8.',
          true
        );
      } else {
        setBanner(
          'ok',
          'Board found. Check the labels — geometry cannot tell which corner is a8; ' +
            'use rotate labels if they sit wrong.'
        );
      }
    } catch (cause) {
      fails++;
      if (!isAuto) setBanner('err', String(cause));
    } finally {
      activity = { kind: 'idle' };
      // Every attempt costs the back-off, hit or miss; a hit has already zeroed the
      // count. `lostSince` is left alone: the loop owns it, and it may well have
      // been re-set while this request was in flight.
      auto = { ...auto, fails, nextAt: performance.now() + backoffFor(fails) };
    }
  }

  // ---- periodic re-detection: re-locking a drifting frame -----------------------
  // The tracker DRIFTS — a Lucas-Kanade plane tracker measures every frame against
  // the last one, and on a board standing still the errors accumulate into about a
  // square of walk over two minutes, silently, because the overlay is drawn where
  // the tracker THINKS the board is
  // (`docs/reports/2026-08-27-framing-drift.md`: 58 % of one session's wrong
  // squares). So a resting, streaming page asks the detector where the board really
  // is every `REDETECT_INTERVAL_MS` and re-anchors when the answer says it has
  // walked. Every judgement — the gate, the per-session baseline that cancels the
  // operator-vs-detector convention offset, the threshold, the counter — is in
  // `state.ts`. This is the timer, the request, and the three things a verdict can
  // move.

  function redetectTick(): void {
    if (
      !shouldRedetect({
        redetect,
        now: performance.now(),
        camera,
        activity,
        streaming,
        stability,
        bursting,
        portrait,
        hidden: document.hidden
      })
    ) {
      return;
    }
    void verifyFraming();
  }

  async function verifyFraming(): Promise<void> {
    if (camera.kind !== 'live') return;
    const frame = camera.frame;
    // The quad as it stands NOW, read beside the request that fixes the pixels:
    // the tracker keeps running through the round trip, and pairing the detector's
    // answer with a quad from several hundred ms later would measure the wait as
    // well as the drift.
    const tracked = camera.framing.quad;
    activity = { kind: 'verifying' };
    try {
      const outcome = await detectQuad(frame);
      if (outcome.kind === 'none') {
        // A miss is NEUTRAL — see `noteRedetectMiss`. It is also silent: nobody
        // asked for this check, and the framing it could not confirm is still the
        // best one anybody has.
        redetect = noteRedetectMiss(redetect, performance.now());
        trial.note('framing-redetect', { outcome: 'miss' });
        return;
      }
      const step = noteRedetection({
        redetect,
        now: performance.now(),
        detected: outcome.quad,
        tracked
      });
      redetect = step.redetect;
      const verdict = step.outcome;
      trial.note('framing-redetect', {
        outcome: verdict.kind,
        adopted: verdict.kind === 'adopt',
        ...('shift' in verdict ? { shift: Number(verdict.shift.toFixed(3)) } : {}),
        ...(verdict.kind === 'ambiguous' ? { margin: Number(verdict.margin.toFixed(2)) } : {})
      });
      if (verdict.kind === 'adopt') {
        adoptFraming(verdict.quad, frame);
      } else if (verdict.kind === 'suspect' && verdict.repeats === 1) {
        // Too far to be drift: the phone or the board moved (DESIGN §4.4.1(b)).
        // Never acted on — a reposition is the operator's to resolve — and said
        // once per episode rather than every ten seconds.
        setBanner(
          'warn',
          'The board no longer sits where the frame says — press look again, or drag ' +
            'the handles back onto it.'
        );
      }
    } catch (cause) {
      redetect = noteRedetectMiss(redetect, performance.now());
      trial.note('framing-redetect', { outcome: 'error', error: String(cause) });
    } finally {
      activity = { kind: 'idle' };
    }
  }

  /**
   * Re-anchor the tracker on the corrected quad.
   *
   * Calibration is deliberately untouched: this is the same physical board and the
   * same labels — `rot` is carried over, and the corrected quad is expressed in the
   * operator's own corner convention (`detected - R0`), so nothing the bank was
   * keyed to has changed except that the crop is now centred on the board again.
   * The banner is a short "here is what just happened", not a nag: there is nothing
   * for the operator to do.
   */
  function adoptFraming(quad: CornerQuad, frame: FrameSize): void {
    if (camera.kind !== 'live') return;
    // A lens switch or a rotation while the request was in flight re-expressed
    // every corner: this quad is in the old frame's pixels and means nothing here.
    if (camera.frame.width !== frame.width || camera.frame.height !== frame.height) return;
    setFraming({ quad, rot: camera.framing.rot, origin: 'detected' });
    saveFraming();
    anchorTracker();
    setBanner('ok', 'The frame had drifted — re-locked onto the board.');
  }

  // ---- calibrate ----------------------------------------------------------------

  function wait(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  interface CalibrationShot {
    /** Full-resolution pixels, ready to encode. */
    readonly canvas: HTMLCanvasElement;
    /** The quad as it stood at the same tick those pixels were drawn. */
    readonly corners: CornerQuad;
    readonly score: number;
  }

  /**
   * The sharpest frame of a short burst, with the corners that were true when it was
   * taken.
   *
   * One frame decides the piece bank for the rest of the session, and this page is
   * held in a hand — an instantaneous grab lands on whichever 1/30 s the thumb fell
   * on, and a good half of those are smeared. So it watches for about a second and a
   * half and keeps the best of ten (`$lib/edge/sharpness`).
   *
   * **Two full-resolution canvases, never ten.** One holds the best so far, the other
   * takes the next candidate, and they swap when the candidate wins; ten 1920x1080
   * canvases would be 80 MB of backing store on a phone.
   *
   * **Each candidate carries its OWN corners**, read beside the `drawImage` that
   * fixed its pixels. `frameTick` keeps tracking right through a calibration — only
   * forwarding pauses — so the quad follows the board while this runs, and pairing
   * the winning pixels with the quad as it stood at the end would describe a board
   * that has since moved. This is what makes hand shake geometrically harmless
   * rather than merely blurry.
   */
  async function bestCalibrationShot(): Promise<CalibrationShot | null> {
    let best = document.createElement('canvas');
    let spare = document.createElement('canvas');
    const scorer = document.createElement('canvas');
    const scoreCtx = scorer.getContext('2d', { willReadFrequently: true });
    let gray: Float32Array | null = null;
    let winner: { corners: CornerQuad; score: number } | null = null;

    for (let i = 0; i < CALIBRATE_SAMPLES; i++) {
      if (i > 0) await wait(CALIBRATE_SAMPLE_MS);
      if (camera.kind !== 'live' || video === undefined || scoreCtx === null) break;
      const { frame, framing } = camera;
      const corners = orderedCorners(framing.quad, framing.rot);

      // Full resolution, deliberately NOT the downscaled stream frame: calibration
      // reads one exemplar per square and every pixel of a piece is evidence.
      spare.width = frame.width;
      spare.height = frame.height;
      spare.getContext('2d')?.drawImage(video, 0, 0);

      // Scored on the BOARD's crop rather than the whole frame: a sharp bookshelf
      // behind a smeared board would otherwise decide which candidate wins.
      //
      // And scored FROM `spare`, not from the video a second time: the two reads
      // would be of different instants, so the ranking would be of frames that are
      // not the ones on offer. This way the number describes the very pixels that
      // would be posted.
      const box = cropBox(corners, frame);
      const k = Math.min(1, CALIBRATE_SCORE_LONG_SIDE / Math.max(box.w, box.h));
      const w = Math.max(3, Math.round(box.w * k));
      const h = Math.max(3, Math.round(box.h * k));
      scorer.width = w;
      scorer.height = h;
      scoreCtx.drawImage(spare, box.x, box.y, box.w, box.h, 0, 0, w, h);
      const { data } = scoreCtx.getImageData(0, 0, w, h);
      // Reused across candidates, and safely — unlike the tracker's pyramid and the
      // motion meter's thumbnail, `tenengrad` reads this buffer and retains nothing.
      if (gray === null || gray.length !== w * h) gray = new Float32Array(w * h);
      const score = tenengrad({ data: toGray(data, gray), width: w, height: h });

      if (winner === null || score > winner.score) {
        winner = { corners, score };
        [best, spare] = [spare, best];
      }
    }
    return winner === null ? null : { canvas: best, ...winner };
  }

  async function runCalibrate(isAuto = false): Promise<void> {
    if (camera.kind !== 'live' || video === undefined) return;
    // Exactly one thing owns the camera and the quad (`Activity`), and the pill being
    // enabled is not proof: an automatic detection can start between the paint and
    // the tap. `detectCorners` ends with an unconditional `activity = idle`, so a
    // calibration begun underneath one would have that fire mid-burst — resuming the
    // forwarder into a half-rebuilt bank. The disabled state says why; this enforces it.
    if (activity.kind !== 'idle') return;
    if (portrait) return;
    // The automatic path earned its trigger (the reading looks like a start
    // position at rest) — a confirm() there would put the tap back in.
    if (
      !isAuto &&
      !confirm(
        'Calibrate from the live camera?\n\n' +
          'The board must CURRENTLY show the STARTING position, with the pieces at rest. ' +
          'The node adds that one frame’s exemplars to its piece bank.'
      )
    ) {
      return;
    }
    activity = { kind: 'calibrating' };
    try {
      setBanner('warn', 'Hold steady — capturing…');
      const shot = await bestCalibrationShot();
      if (shot === null) {
        setBanner('err', 'the camera stopped before a frame could be captured');
        return;
      }
      const jpeg = await toBlob(shot.canvas, CALIBRATE_QUALITY);
      if (jpeg === null) {
        setBanner('err', 'the browser produced no image to calibrate from');
        return;
      }

      const result = await postCalibrate(jpeg, { corners: shot.corners });
      if (!result.ok) {
        trial.note('calibrate-refused', { status: result.error.status, message: result.error.message });
        // A named cause gets the one actionable sentence (the node's full
        // paragraph is in the trial log); anything else is surfaced verbatim.
        if (result.error.rotatedDeg !== undefined) {
          setBanner(
            'err',
            'the board is rotated in the frame — turn the phone so the board sits ' +
              'upright, then calibrate again.',
            true
          );
          return;
        }
        // Nobody asked for an automatic attempt, so its refusal does not shout —
        // the trial log has it, and the cooldown schedules the retry.
        if (isAuto) return;
        setBanner(
          'err',
          result.error.status === 0
            ? `${result.error.message}\nIs the inference node reachable from the viewer?`
            : result.error.message
        );
        return;
      }

      const reply = result.value;
      calibrations++;
      trial.note('calibrated', {
        classes: reply.classes_calibrated,
        exemplars: reply.exemplars,
        label_rotation: reply.label_rotation
      });
      let text =
        `${isAuto ? 'Calibrated automatically from the start position, ' : 'Calibrated '}` +
        `${reply.classes_calibrated}/13 classes — bank now holds ${reply.exemplars} exemplars`;
      // The node resolved the true labelling from the pieces; fold it into the
      // page's own (a relabelling — no point moves) so the streamed corners agree
      // and nobody has to press anything. Sign pinned by the node's
      // test_calibrate_label_rotation_equals_the_corner_label_shift.
      if (reply.label_rotation && camera.kind === 'live') {
        setFraming({
          ...camera.framing,
          rot: absorbCalibrateRotation(camera.framing.rot, reply.label_rotation)
        });
        saveFraming();
        labelsNeedCheck = false;
        text +=
          `\nlabels aligned from the pieces (turned ${reply.label_rotation * 90}°) — ` +
          'a8 is set; use rotate labels in settings if the board reads wrong.';
      }
      setBanner('ok', text);
    } catch (cause) {
      setBanner('err', `${String(cause)}\nIs the inference node reachable from the viewer?`);
    } finally {
      activity = { kind: 'idle' };
    }
  }

  // ---- the controls -------------------------------------------------------------

  function rotateLabels(): void {
    if (camera.kind !== 'live') return;
    setFraming({ ...camera.framing, rot: nextRotation(camera.framing.rot) });
    saveFraming();
    // Pressing this IS the acknowledgement of the "check the labels" warning.
    labelsNeedCheck = false;
    clearBanner();
  }

  /**
   * The sheet's framing control closes it first. Its entire result is on the picture
   * — which corner now says a8 — and a sheet covering the bottom of the screen while
   * that changes is a control that appears to do nothing.
   */
  function rotateFromSheet(): void {
    sheetOpen = false;
    rotateLabels();
  }

  function resetCrop(): void {
    if (camera.kind !== 'live') return;
    setFraming(unframed(camera.frame)); // -> not framed: the page goes back to looking
    auto = AUTO_DETECT_START;
    redetect = REDETECT_START;
    saveFraming();
    anchorTracker();
  }

  /**
   * The crop belongs to the board, not to the page: switching board id swaps in
   * that board's saved quad (and remembers the id itself for the next visit).
   */
  function changeBoard(next: string): void {
    boardId = next.trim() || 'phone';
    ingest.forgetAcks();
    ackedSeq = null;
    writeLocal(BOARD_KEY, boardId);
    if (camera.kind !== 'live') return;
    const frame = camera.frame;
    setFraming(parseStoredFraming(readLocal(cornerKey(boardId)), frame) ?? unframed(frame));
    auto = AUTO_DETECT_START;
    redetect = REDETECT_START;
    saveFraming();
    anchorTracker();
  }

  function toggleTracking(): void {
    tracking = !tracking;
    writeLocal(TRACKING_KEY, tracking ? '1' : '0');
    if (tracking) anchorTracker();
    else trackingState = 'off';
  }

  function togglePanel(): void {
    panel = { ...panel, shown: !panel.shown };
    savePanel();
  }

  function savePanel(): void {
    writeLocal(PANEL_KEY, JSON.stringify(panel));
  }

  // ---- corner handles -----------------------------------------------------------

  function onGrab(corner: CornerIndex): void {
    // A request already owns the quad: letting a drag start here would have the
    // reply overwrite the operator's fingers a second later.
    if (activity.kind !== 'idle' && activity.kind !== 'dragging') return;
    activity = { kind: 'dragging', corner };
  }

  function onMove(corner: CornerIndex, point: Point): void {
    if (camera.kind !== 'live') return;
    const current = camera.framing.quad;
    const quad: CornerQuad = [
      corner === 0 ? point : current[0],
      corner === 1 ? point : current[1],
      corner === 2 ? point : current[2],
      corner === 3 ? point : current[3]
    ];
    setFraming({ ...camera.framing, quad });
  }

  function onRelease(): void {
    activity = { kind: 'idle' };
    if (camera.kind !== 'live') return;
    // A fresh corner is a fresh convention — where this operator's finger puts the
    // corner line is exactly what the drift baseline measures — so it starts over.
    redetect = REDETECT_START;
    // The operator has framed it: automatic detection stands down.
    setFraming({ ...camera.framing, origin: 'operator' });
    saveFraming();
    anchorTracker(); // the operator has spoken: tracking continues from here
  }

  function onCancel(): void {
    activity = { kind: 'idle' };
  }

  // ---- capture bring-up ---------------------------------------------------------

  function stopTracks(): void {
    // iOS hands out one camera at a time: the outgoing tracks must be stopped
    // before the next request, or it fails with the device busy.
    for (const track of stream?.getTracks() ?? []) track.stop();
    stream = null;
  }

  /**
   * All camera opens go through one queue. `openCameraNow` stops the outgoing
   * tracks and then AWAITS `getUserMedia`; two of these interleaved (a lens
   * switch racing the stall watchdog, say) would each stop what the other was
   * mid-way through starting, and the loser's stream — assigned last or never —
   * keeps its tracks live with nothing pointing at them: a camera held claimed
   * for the rest of the session.
   */
  let cameraOpen: Promise<void> = Promise.resolve();
  function openCamera(deviceId: string | null): Promise<void> {
    const next = cameraOpen.then(() => openCameraNow(deviceId));
    cameraOpen = next.catch(() => {}); // a refused lens must not wedge the queue
    return next;
  }

  async function openCameraNow(deviceId: string | null): Promise<void> {
    stopTracks();
    // Forget the old frame size, so the incoming one reads as a first sighting
    // rather than as the phone having been rotated mid-stream.
    lastFrame = null;
    const next = await navigator.mediaDevices.getUserMedia({
      video:
        deviceId !== null
          ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
          : // `ideal`, not `exact`: a laptop with only a front camera should still
            // get a preview rather than an OverconstrainedError.
            {
              facingMode: { ideal: 'environment' },
              width: { ideal: 1920 },
              height: { ideal: 1080 }
            },
      audio: false
    });
    stream = next;
    // The track tells us when it dies — listen, but only while it is the CURRENT
    // stream: a replaced lens's tracks end too, and that ending is history.
    const track = next.getVideoTracks()[0];
    track?.addEventListener('ended', () => {
      if (stream !== next) return;
      trial.note('camera-track-ended');
      setBanner('warn', 'The camera stopped — restarting it…');
      void recoverCamera();
    });
    // iOS mutes the track when the page loses the camera (backgrounding, a call).
    // Noted for the trial log; recovery waits for `unmute`/visibility/the watchdog.
    track?.addEventListener('mute', () => {
      if (stream === next) trial.note('camera-track-mute');
    });
    track?.addEventListener('unmute', () => {
      if (stream !== next) return;
      trial.note('camera-track-unmute');
      void video?.play().catch(() => {});
    });
    if (video !== undefined) {
      video.srcObject = next;
      await video.play().catch(() => {}); // autoplay policy: muted+playsinline is enough
    }
    tracker = null; // a new lens is a new frame size and a new view: nothing carries over
  }

  function currentDeviceId(): string | null {
    return stream?.getVideoTracks()[0]?.getSettings().deviceId ?? null;
  }

  async function refreshLenses(): Promise<readonly Lens[]> {
    let devices: MediaDeviceInfo[] = [];
    try {
      devices = await navigator.mediaDevices.enumerateDevices();
    } catch {
      devices = [];
    }
    const found = devices
      .filter((device) => device.kind === 'videoinput')
      .map((device, i) => ({ id: device.deviceId, label: device.label || `camera ${i + 1}` }));
    lenses = found;
    lensId = currentDeviceId();
    return found;
  }

  async function switchLens(id: string, label: string): Promise<void> {
    saveFraming(); // the crop belongs to the board, and is restored at the new resolution
    try {
      await openCamera(id);
      writeLocal(LENS_KEY, label);
      setBanner('ok', `Switched to ${label}. Check the corners still sit on the board.`);
    } catch (cause) {
      const name = cause instanceof Error ? cause.name : String(cause);
      setBanner('err', `That lens refused to open (${name}); staying on the last one.`);
      await openCamera(null).catch(() => {});
    }
    await refreshLenses();
  }

  async function startCamera(): Promise<void> {
    if (!window.isSecureContext) {
      camera = { kind: 'blocked', reason: { kind: 'insecure' } };
      return;
    }
    if (navigator.mediaDevices?.getUserMedia === undefined) {
      camera = { kind: 'blocked', reason: { kind: 'unsupported' } };
      return;
    }
    camera = { kind: 'requesting' };
    try {
      await openCamera(null);
    } catch (cause) {
      const name = cause instanceof Error ? cause.name : 'Error';
      const detail = cause instanceof Error ? cause.message : String(cause);
      trial.note('camera-blocked', { name, detail });
      camera = { kind: 'blocked', reason: { kind: 'refused', name, detail } };
      return;
    }
    // Labels only exist once permission has been granted, so the remembered lens
    // can only be honoured on a second pass — after a camera, any camera, is open.
    const available = await refreshLenses();
    const saved = readLocal(LENS_KEY);
    const wanted = saved === null ? undefined : available.find((lens) => lens.label === saved);
    if (wanted !== undefined && wanted.id !== currentDeviceId()) {
      try {
        await openCamera(wanted.id);
      } catch {
        /* gone: stay where we are */
      }
      await refreshLenses();
    }
    navigator.mediaDevices.addEventListener('devicechange', () => void refreshLenses());
  }

  /**
   * Corner state needs the real frame size, which only exists once metadata has
   * loaded — and so does the tracker, which is sized from it.
   */
  function onLoadedMetadata(): void {
    if (video === undefined) return;
    const frame: FrameSize = { width: video.videoWidth, height: video.videoHeight };
    if (frame.width === 0 || frame.height === 0) return;
    trial.note('camera-live', { frame });
    lastFrame = frame;
    camera = {
      kind: 'live',
      frame,
      framing: parseStoredFraming(readLocal(cornerKey(boardId)), frame) ?? unframed(frame),
      liveAt: performance.now() // autoexposure and autofocus start settling now
    };
    auto = AUTO_DETECT_START;
    redetect = REDETECT_START;
    // Anchored explicitly, not just built: `resize` can fire before
    // `loadedmetadata` and build the tracker while there is no framing yet, and
    // `ensureTracker` only re-anchors when it actually constructs one.
    ensureTracker(frame);
    anchorTracker();
    startLoop();
  }

  /**
   * Rotating the phone can swap videoWidth/videoHeight under a LIVE track — iOS
   * Safari rotates it, Android Chrome usually does not, so a landscape-first page
   * has to handle both. Corners are stored as FRACTIONS of the frame, so the same
   * fraction is a different physical point once the aspect flips, and the crop
   * starts clamping against the new dimensions. Rescale, rebuild the tracker at the
   * new size, and say the frame changed: the quad will need checking, and only the
   * operator can do that.
   */
  function onVideoResize(): void {
    if (video === undefined) return;
    const frame: FrameSize = { width: video.videoWidth, height: video.videoHeight };
    if (frame.width === 0 || frame.height === 0) return;
    const previous = lastFrame;
    if (previous !== null && previous.width === frame.width && previous.height === frame.height) {
      return;
    }
    lastFrame = frame;
    if (camera.kind !== 'live') return;
    if (previous !== null) {
      trial.note('frame-resized', { from: previous, to: frame });
      camera = {
        ...camera,
        frame,
        framing: { ...camera.framing, quad: rebaseQuad(camera.framing.quad, previous, frame) }
      };
      setBanner('warn', 'The camera frame changed shape — check the corners still sit on the board.');
      saveFraming();
      // R0 is a per-corner offset in PIXELS of a frame that no longer exists.
      redetect = REDETECT_START;
    } else {
      camera = { ...camera, frame };
    }
    tracker = null; // a new frame size is a new tracker, re-anchored on the rescaled quad
    ensureTracker(frame);
  }

  // ---- lifecycle ----------------------------------------------------------------

  onMount(() => {
    port = location.port || (location.protocol === 'https:' ? '443' : '80');
    boardId = readLocal(BOARD_KEY) ?? 'phone';
    tracking = (readLocal(TRACKING_KEY) ?? '1') !== '0';
    panel = parseStoredPanel(readLocal(PANEL_KEY));

    motionCanvas = document.createElement('canvas');
    motionCanvas.width = MOTION_SIDE;
    motionCanvas.height = MOTION_SIDE;
    motionCtx = motionCanvas.getContext('2d', { willReadFrequently: true });
    trackCanvas = document.createElement('canvas');
    trackCtx = trackCanvas.getContext('2d', { willReadFrequently: true });
    sendCanvas = document.createElement('canvas');
    sendCtx = sendCanvas.getContext('2d');

    readScreenOrientation();
    screen.orientation?.addEventListener('change', readScreenOrientation);

    positions.start(TICK_MS);
    statusTimer = setInterval(paintStatus, TICK_MS);
    trial.start();
    trial.note('page-open', { ua: navigator.userAgent });
    void startCamera();

    return () => {
      screen.orientation?.removeEventListener('change', readScreenOrientation);
      if (frameTimer !== null) clearInterval(frameTimer);
      if (statusTimer !== null) clearInterval(statusTimer);
      if (bannerTimer !== null) clearTimeout(bannerTimer);
      positions.stop();
      ingest.stop();
      heartbeat.stop();
      releaseWakeLock();
      stopTracks();
      trial.stop();
    };
  });

  /**
   * Coming back from the background is the common way a phone camera dies: the
   * screen lock is gone (reacquire it), the video element may be paused (play is
   * free), and on iOS the track may have been ended outright (reopen). The mere
   * `mute` case is left to `unmute`/the watchdog — muted-while-hidden is normal.
   */
  function onVisibilityChange(): void {
    trial.note('visibility', { state: document.visibilityState });
    if (document.visibilityState !== 'visible') return;
    // The watchdog restarts its grace period HERE, not in the loop: iOS suspends
    // `setInterval` outright while backgrounded, so the loop's own hidden-reset
    // may never have run — the first ticks after return would then read a 30 s
    // "stall" and escalate straight to a reopen before `play()` had a chance.
    frameClock = freshFrameClock(performance.now());
    if (streaming && wakeLock === null) void acquireWakeLock();
    if (camera.kind !== 'live') return;
    const track = stream?.getVideoTracks()[0];
    if (track === undefined || track.readyState === 'ended') {
      trial.note('camera-dead-on-return');
      void recoverCamera();
    } else {
      void video?.play().catch(() => {});
    }
  }

  function readScreenOrientation(): void {
    screenPortrait = screen.orientation?.type.startsWith('portrait') ?? false;
  }

  /** A bfcache restore resurrects the page with every stream in it dead. */
  function onPageShow(event: PageTransitionEvent): void {
    if (!event.persisted) return;
    trial.note('bfcache-restore');
    frameClock = freshFrameClock(performance.now());
    void recoverCamera();
  }

  /** The most interesting notes are the ones just before the page dies. */
  function onPageHide(): void {
    trial.note('pagehide');
    trial.flushBeacon();
  }
</script>

<svelte:head>
  <title>board-reader · camera</title>
  <!-- No viewport override here: `app.html` now ships `viewport-fit=cover` for every
       route, which is what makes the `env(safe-area-inset-*)` non-zero — and there
       are only two places left that read them, the stage (`Viewfinder.svelte`) and
       the sheet, which is its own fixed element. Emitting a second, identical tag
       only invited the two to drift. -->
</svelte:head>

<svelte:document onvisibilitychange={onVisibilityChange} ondragstart={(e) => e.preventDefault()} />
<svelte:window onpageshow={onPageShow} onpagehide={onPageHide} />

{#snippet chrome()}
  <!-- Everything the page has to SAY lives down here, in one stack. The top of the
       picture is the part being aimed at a board, and status floating over it was the
       operator's complaint. One container rather than three floating boxes because
       the three have to be placed against each other — the chips must not end up
       under the hint pill on a narrow landscape screen — and a flex row states that
       once instead of three sets of offsets that agree by luck.

       It is rendered INSIDE the frame (`Viewfinder.svelte`), which is the whole
       reason it no longer dodges anything: the frame is what the rail leaves over, so
       "left: 0; right: 0" here already stops at the buttons. It used to inset itself
       by a `--cluster-w` the rail also claimed to obey — two numbers that had to
       agree about a layout neither could see. -->
  <div class="bottom-chrome">
    <div class="chips">
      <span class="chip {connTone}">{connDetail}</span>
      <span class="chip {trackChip.tone}">{trackChip.text}</span>
    </div>
    <div class="messages">
      <Banner {banner} />
      <HintBar {hint} onLookAgain={() => void detectCorners()} />
    </div>
  </div>
  {#if portrait}
    <div class="portrait-block" role="alert">
      <div class="glyph" aria-hidden="true">📱</div>
      <div class="head">turn the phone sideways</div>
      <div class="sub">
        {streaming ? 'stream paused — ' : ''}pieces are read standing up, and a sideways
        board cannot be. {streaming
          ? 'sending resumes in landscape.'
          : 'detect, calibrate and start unlock in landscape.'}
      </div>
    </div>
  {/if}
{/snippet}

{#snippet controls()}
  <Rail
    positionShown={panel.shown}
    detectDisabled={camera.kind !== 'live' || activity.kind !== 'idle' || portrait}
    rotateDisabled={camera.kind !== 'live'}
    rotateAttention={labelsNeedCheck}
    calibrateDisabled={calibrateBlock !== null}
    calibrateCount={calibrations}
    calibrateTitle={calibrateBlock ??
      'rebuild the piece bank from the live camera (the board must show the starting position)'}
    {streaming}
    streamDisabled={portrait}
    onMore={() => (sheetOpen = !sheetOpen)}
    onPosition={togglePanel}
    onDetect={() => void detectCorners()}
    onRotate={rotateLabels}
    onCalibrate={() => void runCalibrate()}
    onStream={toggleStreaming}
  />
{/snippet}

<div class="camera">
  {#if camera.kind === 'blocked'}
    <BlockedNotice reason={camera.reason} {port} />
  {:else}
    <Viewfinder
      bind:video
      {camera}
      tracking={trackingState}
      {dragging}
      {onGrab}
      {onMove}
      {onRelease}
      {onCancel}
      {onLoadedMetadata}
      onFrameResize={onVideoResize}
      {chrome}
      {controls}
    />
  {/if}

  <!-- A sibling of the stage on purpose: chessground sizes itself from this
       element's box and re-measures through its own observer, so the box has to
       stay real even when the stage is gone. -->
  <PositionPanel
    bind:panel
    {placement}
    {shapes}
    status={panelStatus}
    onCommit={savePanel}
  />

  <SettingsSheet
    open={sheetOpen}
    {boardId}
    {tracking}
    rotateDisabled={camera.kind !== 'live'}
    {lenses}
    {lensId}
    sent={sentCount}
    acked={ackedSeq}
    dropped={droppedCount}
    {fps}
    {stability}
    {bursting}
    placement={board?.event.placement ?? '—'}
    onClose={() => (sheetOpen = false)}
    onBoardId={changeBoard}
    onToggleTracking={toggleTracking}
    onResetCrop={resetCrop}
    onRotate={rotateFromSheet}
    onLens={(id, label) => void switchLens(id, label)}
  />
</div>

<style>
  /* Nothing on this page scrolls, and nothing on it is text to select. */
  :global(html),
  :global(body) {
    height: 100%;
    overflow: hidden;
    -webkit-user-select: none;
    user-select: none;
    -webkit-touch-callout: none;
    background: #000;
    -webkit-tap-highlight-color: transparent;
  }

  .portrait-block {
    position: absolute;
    inset: 0;
    z-index: 30;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    text-align: center;
    padding: 1.5rem;
    background: rgba(0, 0, 0, 0.84);
    color: var(--fg);
  }
  .portrait-block .glyph {
    font-size: 2.2rem;
  }
  .portrait-block .head {
    font-weight: 600;
    font-size: 1.05rem;
  }
  .portrait-block .sub {
    color: var(--muted);
    font-size: 0.85rem;
    max-width: 34ch;
  }

  .camera {
    --bg: #14171c;
    --panel: #1b1f26;
    --panel-2: #232833;
    --line: #272b33;
    --fg: #e6e6e6;
    --muted: #9aa3ad;
    --dim: #7d8590;
    --accent: #2a6ef0;
    --accent-soft: #6aa8ff;
    --ok: #97e0a6;
    --warn: #ffcf33;
    --bad: #ff8f8f;
    color: var(--fg);
  }

  /* The button, input and chip base rules were global in the hand-written page and
     every component below still expects them; `:global()` under `.camera` keeps
     them page-scoped instead of leaking into the other two routes. */
  .camera :global(button) {
    background: var(--accent);
    color: #fff;
    border: 0;
    border-radius: 9px;
    padding: 10px 14px;
    font: inherit;
    font-size: 15px;
    font-weight: 600;
    min-height: 48px;
    touch-action: manipulation;
    cursor: pointer;
  }

  .camera :global(button.secondary) {
    background: #333842;
  }

  .camera :global(button:disabled) {
    opacity: 0.4;
    cursor: default;
  }

  /* The one bottom strip: chips on the left, messages centred in what is left of it,
     along the bottom edge of the FRAME. It reserves nothing and dodges nothing — the
     frame's edges are already the room the rail left over, in either orientation, so
     `left: 0; right: 0; bottom: 0` is the whole placement.

     Inert throughout, so nothing here can intercept a corner handle underneath — only
     the hint's "look again" opts back in. */
  .bottom-chrome {
    position: absolute;
    z-index: 7;
    pointer-events: none;
    left: 0;
    right: 0;
    bottom: 10px;
    display: flex;
    align-items: flex-end;
    gap: 10px;
    padding: 0 12px;
  }

  /* Portrait has width to spare and height to lose, so the strip stacks instead:
     `column-reverse` keeps the chips at the bottom (nearest the button row) with the
     messages above them, each still aligned the way it is in landscape. */
  @media (orientation: portrait) {
    .bottom-chrome {
      flex-direction: column-reverse;
      align-items: stretch;
      gap: 8px;
    }
  }

  /* A column, not a row: two chips side by side are ~230px, which on a landscape
     phone leaves the sentence beside them nothing to be centred in. Portrait puts
     them back in a row — there the constraint is vertical, not horizontal. */
  .chips {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }

  @media (orientation: portrait) {
    .chips {
      flex-direction: row;
      flex-wrap: wrap;
    }
  }

  /* Banner above hint, both centred in whatever the chips leave. `min-width: 0` so a
     long sentence wraps instead of pushing the chips off the screen. */
  .messages {
    flex: 1 1 auto;
    min-width: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }

  .chip {
    font-size: 12px;
    padding: 3px 9px;
    border-radius: 10px;
    white-space: nowrap;
    background: rgba(35, 40, 51, 0.85);
    color: var(--muted);
  }

  .chip.up {
    background: #1f3a24;
    color: var(--ok);
  }

  .chip.warn {
    background: #3a2f18;
    color: var(--warn);
  }

  .chip.bad {
    background: #3a2020;
    color: var(--bad);
  }
</style>
