<script lang="ts">
  /**
   * /camera — the phone as a DUMB CAMERA (DESIGN D11): it films, crops and streams,
   * and that is all. There is nothing to press. The control UI on the laptop
   * (`/control`) chooses the lens, the resolution, the crop rectangle and whether to
   * stream; the node pushes that config down this page's control socket
   * (`CameraLink`) and the page applies it.
   *
   *  - the FRAME loop (`CAPTURE_FPS`): the stall watchdog, the motion meter on the
   *    configured crop, and — while streaming — the adaptive forwarder deciding which
   *    frames are cropped, encoded and sent on `/ws/ingest` (DESIGN D4).
   *  - the REPORT: what the camera is (lenses, delivered frame, stream state), sent
   *    on the control socket whenever it changes — an `$effect`, not a timer.
   *  - the PREVIEW (`PREVIEW_MS`): a small JPEG of the WHOLE frame, so the control UI
   *    can show the field of view and draw the crop on it — sent only while a
   *    control page watches (the node says `previews` on/off).
   *
   * Same-origin and relative throughout (`$lib/api`), relayed by the viewer:
   * `getUserMedia` needs a secure context, and one origin means one certificate.
   *
   * PRERENDERED: no browser API at module scope or during SSR; everything is inside
   * `onMount` or a handler.
   */
  import { onMount } from 'svelte';

  import {
    AdaptiveForwarder,
    CAPTURE_FPS,
    MotionMeter,
    MOTION_DEFAULTS,
    MOTION_OVERSAMPLE,
    sampleThumbnail
  } from '$lib/edge/motion';
  import type { ForwardDecision } from '$lib/edge/motion';
  import { IngestClient, type IngestState } from '$lib/edge/ingest';
  import { CameraLink } from '$lib/edge/cameralink';
  import { tenengrad, toGray } from '$lib/edge/sharpness';
  import { TrialLog } from '$lib/edge/triallog';
  import { Heartbeat } from '$lib/edge/heartbeat';
  import { DEFAULT_CAMERA_CONFIG } from '$lib/protocol';
  import type { CameraConfig, CameraDevice, CameraReport } from '$lib/protocol/generated';

  import BlockedNotice from '$lib/components/camera/BlockedNotice.svelte';
  import {
    DEFAULT_BOARD_ID,
    JPEG_QUALITY,
    PREVIEW_LONG_SIDE,
    PREVIEW_MS,
    PREVIEW_QUALITY,
    SEND_LONG_SIDE,
    SHARPNESS_LONG_SIDE,
    TICK_MS,
    boardIdFrom,
    cropBox,
    fitLongSide,
    freshFrameClock,
    nextCapture,
    noteFrameClock,
    ratePerSecond,
    videoConstraints,
    withinRateWindow,
    type CameraState,
    type Capture,
    type PixelBox
  } from '$lib/components/camera/state';

  // ---- the state ---------------------------------------------------------------

  let camera = $state.raw<CameraState>({ kind: 'idle' });
  let config = $state.raw<CameraConfig>(DEFAULT_CAMERA_CONFIG);
  let lenses = $state.raw<readonly CameraDevice[]>([]);
  let lensLabel = $state<string | null>(null);
  let boardId = $state(DEFAULT_BOARD_ID);
  /** Whether the control socket is open — the phone's link to its config. */
  let nodeReachable = $state<boolean | null>(null);
  let ingestState = $state<IngestState>('idle');
  let fps = $state(0);
  /** The one thing worth saying right now, if anything (a refused frame, a stall). */
  let message = $state<string | null>(null);
  let port = $state('8080');
  let video = $state<HTMLVideoElement | undefined>();

  // ---- the plumbing, deliberately NOT reactive ---------------------------------

  const meter = new MotionMeter();
  const forwarder = new AdaptiveForwarder(meter);
  const trial = new TrialLog();
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
      ingestState = status;
      trial.note('ingest-status', { status, detail });
    },
    // Said once per distinct reason: at the burst rate a per-frame message would
    // rewrite the status line twelve times a second.
    onFrameRefused: (reason) => {
      if (reason === lastRefusal) return;
      lastRefusal = reason;
      trial.note('frame-refused', { reason });
      message = `the node refused a frame: ${reason}`;
    }
  });

  const MOTION_SIDE = MOTION_DEFAULTS.downscale;
  /** The canvas the crop is drawn on before `sampleThumbnail` area-averages it. */
  const MOTION_DRAW_SIDE = MOTION_SIDE * MOTION_OVERSAMPLE;

  let stream: MediaStream | null = null;
  /** What the open camera was asked for — a config is applied once, not every poll. */
  let capture: Capture = { device: null, resolution: DEFAULT_CAMERA_CONFIG.resolution };
  let streaming = false;
  let motionCtx: CanvasRenderingContext2D | null = null;
  /** The box the meter's last thumbnail came from; `null` after a camera (re)open. */
  let meteredBox: PixelBox | null = null;
  let sharpCanvas: HTMLCanvasElement | null = null;
  let sharpCtx: CanvasRenderingContext2D | null = null;
  let sendCanvas: HTMLCanvasElement | null = null;
  let sendCtx: CanvasRenderingContext2D | null = null;
  let previewCanvas: HTMLCanvasElement | null = null;
  let previewCtx: CanvasRenderingContext2D | null = null;
  let wakeLock: WakeLockSentinel | null = null;
  let frameClock = freshFrameClock(0);
  let encoding = false;
  let previewing = false;
  /** Created on mount: the board comes from `?board=`, which prerendering cannot read. */
  let link: CameraLink | null = null;
  let stamps: number[] = [];

  // ---- derived ------------------------------------------------------------------

  const frame = $derived(camera.kind === 'live' ? camera.frame : null);
  const portrait = $derived(frame !== null && frame.height > frame.width);
  const crop = $derived(config.crop);
  const status = $derived(
    [
      `board ${boardId}`,
      nodeReachable === null ? 'node…' : nodeReachable ? 'node connected' : 'node unreachable',
      ingestState === 'streaming'
        ? `streaming ${fps.toFixed(1)} fps`
        : ingestState === 'idle'
          ? 'not streaming'
          : ingestState,
      ...(frame === null ? [] : [`${frame.width}×${frame.height}`]),
      ...(lensLabel === null ? [] : [lensLabel])
    ].join(' · ')
  );
  $effect(() => {
    trial.note('portrait', { portrait });
  });
  /** What the camera is now. */
  const report = $derived<CameraReport>({
    devices: lenses,
    device: lensLabel,
    frame,
    ingest: ingestState
  });
  // Sent whenever any of it changes (the link drops repeats). `report` is read
  // outside the call: `link?.report(...)` would skip reading it while `link` is
  // null, and the effect would then track nothing and never run again.
  $effect(() => {
    const now = report;
    link?.report(now);
  });

  function toBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob | null> {
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
  }

  // ---- the frame loop -----------------------------------------------------------

  function currentBox(): PixelBox | null {
    return frame === null ? null : cropBox(config.crop, frame);
  }

  function grabThumbnail(box: PixelBox): Float32Array | null {
    if (motionCtx === null || video === undefined) return null;
    // Motion is measured on the CROP: the rest of the frame (resting hands,
    // background, exposure hunting) is not the board and would only dilute it.
    // Drawn oversampled and area-averaged in JS: a canvas downscale straight to
    // MOTION_SIDE keeps the sensor noise and buries a hand under it (MOTION_OVERSAMPLE).
    motionCtx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, MOTION_DRAW_SIDE, MOTION_DRAW_SIDE);
    // A fresh array per frame: `MotionMeter` RETAINS what it is given as the next
    // frame's reference, so a reused buffer would read as perfectly still.
    return sampleThumbnail(motionCtx.getImageData(0, 0, MOTION_DRAW_SIDE, MOTION_DRAW_SIDE), MOTION_SIDE);
  }

  /** A new crop or a reopened camera is a new picture, not motion in the old one. */
  function rebaseMeterOn(box: PixelBox): void {
    const same =
      meteredBox !== null &&
      meteredBox.x === box.x &&
      meteredBox.y === box.y &&
      meteredBox.w === box.w &&
      meteredBox.h === box.h;
    if (!same) meter.rebase();
    meteredBox = box;
  }

  function frameTick(): void {
    if (camera.kind !== 'live') return;
    const captureTs = Date.now() / 1000;
    watchFrameClock();
    const box = currentBox();
    if (box === null) return;
    const thumbnail = grabThumbnail(box);
    if (thumbnail === null) return;
    rebaseMeterOn(box);
    // The meter sees EVERY sampled frame, streaming or not — dropping frames
    // upstream would make stability lie.
    const decision = streaming
      ? forwarder.decide(thumbnail, { boardId, captureTs })
      : (meter.update(thumbnail), null);
    if (decision === null || !ingest.connected) return;
    // A frame still in the encoder, or a link already behind, means another frame
    // would only make the reading staler (DESIGN D3/D4).
    if (encoding || ingest.behind) {
      ingest.noteDropped();
      return;
    }
    encoding = true;
    void sendFrame(decision, captureTs, box).finally(() => {
      encoding = false;
    });
  }

  function sharpnessOf(box: PixelBox): number | null {
    if (sharpCanvas === null || sharpCtx === null || video === undefined) return null;
    const { w, h } = fitLongSide(box, SHARPNESS_LONG_SIDE);
    sharpCanvas.width = w;
    sharpCanvas.height = h;
    sharpCtx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, w, h);
    const { data } = sharpCtx.getImageData(0, 0, w, h);
    return tenengrad({ data: toGray(data, new Float32Array(w * h)), width: w, height: h });
  }

  async function sendFrame(decision: ForwardDecision, captureTs: number, box: PixelBox): Promise<void> {
    if (sendCanvas === null || sendCtx === null || video === undefined) return;
    const { w, h } = fitLongSide(box, SEND_LONG_SIDE);
    sendCanvas.width = w;
    sendCanvas.height = h;
    sendCtx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, w, h);
    const sharpness = sharpnessOf(box);
    const jpeg = await toBlob(sendCanvas, JPEG_QUALITY);
    if (jpeg === null) return;
    const outcome = ingest.sendFrame({
      boardId,
      captureTs,
      stability: decision.stability,
      ...(sharpness === null ? {} : { sharpness }),
      burstId: decision.burstId,
      jpeg
    });
    if (outcome === 'sent') stamps.push(performance.now());
  }

  function paintStatus(): void {
    stamps = withinRateWindow(stamps, performance.now());
    fps = ratePerSecond(stamps, streaming);
  }

  // ---- the control channel ------------------------------------------------------

  function connectControl(id: string): CameraLink {
    return new CameraLink({
      boardId: id,
      onConfig: applyConfig,
      onPreviews: (on) => {
        trial.note('previews', { on });
        if (on) void postPreview();
      },
      onLink: (up, detail) => {
        if (nodeReachable !== up) trial.note('control-link', { up, detail });
        nodeReachable = up;
      },
      onRefused: (reason) => {
        trial.note('control-refused', { reason });
        message = `the node refused a message: ${reason}`;
      }
    });
  }

  /** The config the node pushed; kept as it is while the link is down. */
  function applyConfig(next: CameraConfig): void {
    if (JSON.stringify(next) !== JSON.stringify(config)) trial.note('config', { ...next });
    config = next;
    setStreaming(next.streaming);
    reconcileCapture();
  }

  /**
   * Open the lens and resolution the config names, if the camera is not already on
   * them. Called on every config, when the camera first goes live and whenever the
   * lens list changes, because a config is pushed once: a lens label the phone could
   * not resolve yet must be tried again once it can.
   */
  function reconcileCapture(): void {
    if (camera.kind !== 'live') return;
    const wanted = nextCapture(capture, config, lenses);
    if (wanted !== null) void switchCapture(wanted);
  }

  async function postPreview(): Promise<void> {
    if (link === null || !link.previewsWanted) return;
    if (previewing || document.hidden || camera.kind !== 'live') return;
    if (previewCanvas === null || previewCtx === null || video === undefined) return;
    previewing = true;
    try {
      const { w, h } = fitLongSide({ w: camera.frame.width, h: camera.frame.height }, PREVIEW_LONG_SIDE);
      previewCanvas.width = w;
      previewCanvas.height = h;
      previewCtx.drawImage(video, 0, 0, w, h);
      const jpeg = await toBlob(previewCanvas, PREVIEW_QUALITY);
      if (jpeg !== null) link.sendPreview(jpeg);
    } finally {
      previewing = false;
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

  function setStreaming(on: boolean): void {
    if (on === streaming) return;
    streaming = on;
    if (on) {
      forwarder.reset();
      stamps = [];
      lastRefusal = null;
      ingest.start();
      heartbeat.start();
      void acquireWakeLock();
      trial.note('streaming-on');
    } else {
      ingest.stop();
      heartbeat.stop();
      void wakeLock?.release().catch(() => {});
      wakeLock = null;
      trial.note('streaming-off', { sent: ingest.sent, acked: ingest.acked, dropped: ingest.dropped });
    }
  }

  // ---- the camera ---------------------------------------------------------------

  function stopTracks(): void {
    // iOS hands out one camera at a time: stop the outgoing tracks first.
    for (const track of stream?.getTracks() ?? []) track.stop();
    stream = null;
  }

  /**
   * All camera opens go through one queue: two `getUserMedia` calls interleaved (a
   * lens switch racing the stall watchdog) would each stop what the other was
   * starting, and the loser's tracks would stay claimed for the session.
   */
  let cameraOpen: Promise<void> = Promise.resolve();
  function openCamera(target: Capture): Promise<void> {
    const next = cameraOpen.then(() => openCameraNow(target));
    cameraOpen = next.catch(() => {}); // a refused lens must not wedge the queue
    return next;
  }

  async function openCameraNow(target: Capture): Promise<void> {
    stopTracks();
    const deviceId =
      target.device === null ? null : (lenses.find((lens) => lens.label === target.device)?.id ?? null);
    const next = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints(target.resolution, deviceId),
      audio: false
    });
    stream = next;
    meteredBox = null;
    const track = next.getVideoTracks()[0];
    track?.addEventListener('ended', () => {
      if (stream !== next) return; // a replaced lens's tracks end too
      trial.note('camera-track-ended');
      message = 'the camera stopped — restarting it…';
      void recoverCamera();
    });
    track?.addEventListener('unmute', () => {
      if (stream === next) void video?.play().catch(() => {});
    });
    if (video !== undefined) {
      video.srcObject = next;
      await video.play().catch(() => {}); // muted + playsinline satisfies autoplay
    }
    await refreshLenses();
  }

  async function refreshLenses(): Promise<void> {
    let devices: MediaDeviceInfo[] = [];
    try {
      devices = await navigator.mediaDevices.enumerateDevices();
    } catch {
      devices = [];
    }
    lenses = devices
      .filter((device) => device.kind === 'videoinput')
      .map((device, i) => ({ id: device.deviceId, label: device.label || `camera ${i + 1}` }));
    const openId = stream?.getVideoTracks()[0]?.getSettings().deviceId;
    lensLabel = lenses.find((lens) => lens.id === openId)?.label ?? null;
    reconcileCapture();
  }

  /** Recorded BEFORE the open, so a lens that refuses is not asked for again every poll. */
  async function switchCapture(target: Capture): Promise<void> {
    capture = target;
    trial.note('capture', { ...target });
    try {
      await openCamera(target);
      message = null;
    } catch (cause) {
      const name = cause instanceof Error ? cause.name : String(cause);
      message = `the lens ${target.device ?? 'default'} at ${target.resolution} refused to open (${name})`;
      await openCamera({ device: null, resolution: target.resolution }).catch(() => {});
    }
  }

  let recovering = false;
  async function recoverCamera(): Promise<void> {
    if (recovering || camera.kind !== 'live') return;
    recovering = true;
    try {
      await openCamera(capture);
      trial.note('camera-recovered');
      message = null;
    } catch {
      try {
        await openCamera({ device: null, resolution: capture.resolution });
      } catch (cause) {
        trial.note('camera-recover-failed', { name: cause instanceof Error ? cause.name : String(cause) });
        message = 'the camera did not come back — reload the page';
      }
    } finally {
      recovering = false;
    }
  }

  function watchFrameClock(): void {
    if (video === undefined) return;
    const step = noteFrameClock(frameClock, video.currentTime, performance.now(), document.hidden);
    frameClock = step.clock;
    if (step.action === 'play') {
      trial.note('camera-stall-play');
      void video.play().catch(() => {});
    } else if (step.action === 'reopen') {
      trial.note('camera-stall-reopen', { attempt: step.clock.reopens });
      message = 'the camera stalled — restarting it…';
      void recoverCamera();
    }
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
      await openCamera(capture);
    } catch (cause) {
      const name = cause instanceof Error ? cause.name : 'Error';
      const detail = cause instanceof Error ? cause.message : String(cause);
      trial.note('camera-blocked', { name, detail });
      camera = { kind: 'blocked', reason: { kind: 'refused', name, detail } };
      return;
    }
    navigator.mediaDevices.addEventListener('devicechange', () => void refreshLenses());
  }

  /** The frame size exists only once metadata has loaded, and changes on rotation. */
  function onFrameSize(): void {
    if (video === undefined || video.videoWidth === 0 || video.videoHeight === 0) return;
    const size = { width: video.videoWidth, height: video.videoHeight };
    if (camera.kind === 'live' && camera.frame.width === size.width && camera.frame.height === size.height) {
      return;
    }
    trial.note('camera-live', { frame: size });
    const first = camera.kind !== 'live';
    camera = { kind: 'live', frame: size };
    // A config that arrived during bring-up is applied now the camera is live.
    if (first) reconcileCapture();
  }

  // ---- lifecycle ----------------------------------------------------------------

  function canvas2d(): [HTMLCanvasElement, CanvasRenderingContext2D | null] {
    const element = document.createElement('canvas');
    return [element, element.getContext('2d', { willReadFrequently: true })];
  }

  onMount(() => {
    port = location.port || (location.protocol === 'https:' ? '443' : '80');
    boardId = boardIdFrom(location.search);
    const [motionCanvas, motion] = canvas2d();
    motionCanvas.width = MOTION_DRAW_SIDE;
    motionCanvas.height = MOTION_DRAW_SIDE;
    motionCtx = motion;
    [sharpCanvas, sharpCtx] = canvas2d();
    [previewCanvas, previewCtx] = canvas2d();
    sendCanvas = document.createElement('canvas');
    sendCtx = sendCanvas.getContext('2d');

    trial.start();
    trial.note('page-open', { ua: navigator.userAgent, board: boardId });
    void startCamera();
    const timers = [
      setInterval(frameTick, Math.round(1000 / CAPTURE_FPS)),
      setInterval(paintStatus, TICK_MS),
      setInterval(() => void postPreview(), PREVIEW_MS)
    ];
    link = connectControl(boardId);
    link.report(report);
    link.start();

    return () => {
      for (const timer of timers) clearInterval(timer);
      link?.stop();
      link = null;
      setStreaming(false);
      stopTracks();
      trial.stop();
    };
  });

  /**
   * Coming back from the background is the common way a phone camera dies: the
   * screen lock is gone, the video may be paused, and on iOS the track may have
   * ended outright.
   */
  function onVisibilityChange(): void {
    trial.note('visibility', { state: document.visibilityState });
    if (document.visibilityState !== 'visible') return;
    // iOS suspends `setInterval` while backgrounded, so the watchdog's grace
    // period restarts here or the first tick back reads a 30 s "stall".
    frameClock = freshFrameClock(performance.now());
    if (streaming && wakeLock === null) void acquireWakeLock();
    const track = stream?.getVideoTracks()[0];
    if (camera.kind === 'live' && (track === undefined || track.readyState === 'ended')) {
      trial.note('camera-dead-on-return');
      void recoverCamera();
    } else {
      void video?.play().catch(() => {});
    }
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
</svelte:head>

<svelte:document onvisibilitychange={onVisibilityChange} />
<svelte:window onpageshow={onPageShow} onpagehide={onPageHide} />

<div class="camera">
  {#if camera.kind === 'blocked'}
    <BlockedNotice reason={camera.reason} {port} />
  {:else}
    <!-- The picture is sized to the frame's own aspect ratio, so the crop outline
         can be placed in percent of it with no second coordinate space. -->
    <div
      class="picture"
      style:--aspect={frame === null ? 16 / 9 : frame.width / frame.height}
    >
      <video
        bind:this={video}
        autoplay
        muted
        playsinline
        onloadedmetadata={onFrameSize}
        onresize={onFrameSize}
      ></video>
      {#if crop !== null}
        <div
          class="crop"
          data-testid="crop-outline"
          style:left="{crop.x * 100}%"
          style:top="{crop.y * 100}%"
          style:width="{crop.w * 100}%"
          style:height="{crop.h * 100}%"
        ></div>
      {/if}
    </div>
    <div class="status" data-testid="camera-status">
      <span class:up={ingestState === 'streaming'}>{status}</span>
      {#if portrait}
        <span class="warn">portrait: turn the phone sideways</span>
      {/if}
      {#if message !== null}
        <span class="warn">{message}</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* Nothing on this page scrolls, and nothing on it is text to select. */
  :global(html),
  :global(body) {
    height: 100%;
    overflow: hidden;
    -webkit-user-select: none;
    user-select: none;
    background: #000;
  }

  .camera {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    color: #e6e6e6;
  }

  /* As big as fits, exactly the frame's aspect ratio. */
  .picture {
    position: relative;
    overflow: hidden;
    aspect-ratio: var(--aspect);
    width: min(100vw, calc(100dvh * var(--aspect)));
  }

  video {
    display: block;
    width: 100%;
    height: 100%;
  }

  /* The part of the frame that is sent; the rest is dimmed. */
  .crop {
    position: absolute;
    border: 2px solid #6aa8ff;
    box-shadow: 0 0 0 100vmax rgba(0, 0, 0, 0.45);
    pointer-events: none;
  }

  .status {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 4px 12px;
    justify-content: center;
    padding: 6px 12px calc(6px + env(safe-area-inset-bottom));
    font-size: 13px;
    background: rgba(20, 23, 28, 0.75);
    color: #9aa3ad;
  }

  .status .up {
    color: #97e0a6;
  }

  .status .warn {
    color: #ffcf33;
  }
</style>
