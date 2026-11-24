<script lang="ts">
  /**
   * `/snapshot` — read one photo, all the way down.
   *
   * This is the **pipeline inspector**: the diagnostic surface for the reading algorithm.
   * One photo goes in and the page shows every stage that turned it into a position —
   * capture, read, repair, output — each with its own state, headline and full
   * evidence. The recognizer finds the board and its orientation in the whole photo,
   * so there are no corners to set and the read is always in the chess frame (rank 8
   * on top).
   *
   * Two invariants the whole page is built around:
   *
   *  - **One image.** The photo is downscaled once and *that* image is what is posted
   *    to `/estimate` and what the capture stage describes.
   *  - **The node's read is never mutated.** `prior.placement ?? placement` plus
   *    `square_dist` is the baseline; hand corrections are a separate layer over it. So
   *    "reset corrections" restores exactly what the node served, and the ground-truth
   *    label is derived rather than a second copy of the board.
   *
   * Prerendered: nothing below touches a browser API at module scope or during SSR —
   * the canvases, observers and document listeners all live in `$effect`.
   */
  import { onMount } from 'svelte';
  import { estimate, listUploads, saveLabel, uploadPhotoUrl } from '#lib/api.ts';
  import Board from '#lib/board/Board.svelte';
  import { squareKey } from '#lib/board/fen.ts';
  import ConfidenceMap from '#lib/components/snapshot/ConfidenceMap.svelte';
  import CorrectionsPanel from '#lib/components/snapshot/CorrectionsPanel.svelte';
  import GroundTruthForm from '#lib/components/snapshot/GroundTruthForm.svelte';
  import HistoryList from '#lib/components/snapshot/HistoryList.svelte';
  import KvTable from '#lib/components/snapshot/KvTable.svelte';
  import PipelineStep from '#lib/components/snapshot/PipelineStep.svelte';
  import SquareInspector from '#lib/components/snapshot/SquareInspector.svelte';
  import {
    READS_CAP,
    historyNote,
    historyRows,
    type SessionRead
  } from '#lib/components/snapshot/history.ts';
  import {
    MAX_SOURCE_PX,
    POST_QUALITY,
    THUMB_QUALITY,
    captureRows,
    captureSummary,
    kb,
    meanLuma,
    type CaptureFacts,
    type SourceImage
  } from '#lib/components/snapshot/photo.ts';
  import {
    READ_STAGES,
    STAGES,
    initialPipeline,
    resetFrom,
    withStage,
    withStages,
    type Pipeline,
    type StageId
  } from '#lib/components/snapshot/pipeline.ts';
  import {
    NO_CORRECTIONS,
    argmaxClass,
    assignmentFor,
    baselineGrid,
    classOfKey,
    confidenceOf,
    confidenceSummary,
    correctedGrid,
    cycleTarget,
    fmt,
    fullFen,
    indexOfSquare,
    labelState,
    lichessUrls,
    placementOfGrid,
    rawEventJson,
    scoresAt,
    shownClassAt,
    weakestSquares,
    withCorrection,
    type Corrections,
    type SideToMove
  } from '#lib/components/snapshot/reading.ts';
  import { materialTable, readTable, repairTable } from '#lib/components/snapshot/tables.ts';
  import {
    EMPTY_PLACEMENT,
    GRID,
    LOW_CONFIDENCE,
    WARN_LOW_LIGHT,
    type ClassName,
    type EstimateResponse,
    type UploadRecord
  } from '#lib/protocol/index.ts';
  import type { DrawShape } from '@lichess-org/chessground/draw';

  // ---- state ---------------------------------------------------------------
  // The photo lives in an offscreen canvas — the posted JPEG and the capture stage
  // both read it, and both mean its pixels.
  // (`$state<T>(null)`, never `let x: T | null = $state(null)`: with the annotation on the
  // variable, TypeScript's flow analysis narrows every top-level `$derived` that reads it
  // to the initial `null`.)
  let source: HTMLCanvasElement | null = null;

  let image = $state<SourceImage | null>(null);
  let facts = $state<CaptureFacts | null>(null);
  let pipeline: Pipeline = $state(initialPipeline());
  /** Which stage bodies are expanded. Everything starts collapsed. */
  let stageOpen: Record<StageId, boolean> = $state({
    capture: false,
    read: false,
    repair: false,
    output: false
  });

  let reading = $state<EstimateResponse | null>(null);
  let corrections = $state<Corrections>(NO_CORRECTIONS);
  let selected = $state<{ readonly r: number; readonly c: number } | null>(null);
  let savedLabel = $state<string | null>(null);
  let submittingLabel = $state(false);
  let stm: SideToMove = $state('w');

  let runState = $state('idle');
  let busy = $state<string | null>(null);
  let failure = $state('');
  let banner = $state<{ readonly kind: 'info' | 'warn'; readonly text: string } | null>(null);
  let bannerTimer: ReturnType<typeof setTimeout> | null = null;

  let estimateMs = $state<number | null>(null);

  let reads = $state<readonly SessionRead[]>([]);
  let uploads = $state<readonly UploadRecord[]>([]);
  let retention = $state<boolean | null>(null);
  let activeReadId = $state<number | null>(null);
  let readSeq = 0;

  let boardColumn: HTMLDivElement | undefined = $state();
  let actionBar: HTMLDivElement | undefined = $state();
  let boardSize = $state(360);
  let barHeight = $state(64);

  // ---- derived -------------------------------------------------------------

  const dist = $derived(reading?.event.square_dist ?? null);
  const baseGrid = $derived(reading === null ? null : baselineGrid(reading));
  const shownGrid = $derived(baseGrid === null ? null : correctedGrid(baseGrid, corrections));
  const shownPlacement = $derived(shownGrid === null ? null : placementOfGrid(shownGrid));
  /**
   * Everything the inspector needs, or nothing at all. Bundling it into one value is
   * what makes "an inspected square with no distribution behind it" unrepresentable —
   * the panel cannot be rendered without the scores it is a view of.
   */
  const inspection = $derived.by(() => {
    const at = selected;
    const grid = baseGrid;
    if (at === null || grid === null || dist === null) return null;
    const scores = scoresAt(dist, at.r, at.c);
    if (scores === null) return null;
    return {
      square: squareKey(at.r, at.c),
      scores,
      shown: shownClassAt(grid, corrections, at.r, at.c)
    };
  });
  const links = $derived(shownPlacement === null ? null : lichessUrls(shownPlacement, stm));

  const rows = $derived(historyRows(reads, uploads));

  const boardShapes = $derived.by((): DrawShape[] => {
    const shapes: DrawShape[] = [];
    if (baseGrid === null) return shapes;
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const square = squareKey(r, c);
        if (corrections.has(square)) {
          shapes.push({ orig: square, brush: 'blue' });
        } else if (confidenceOf(scoresAt(dist, r, c)) < LOW_CONFIDENCE) {
          shapes.push({ orig: square, brush: 'red' });
        }
      }
    }
    // Paints last, so the inspected square sits on top of a blue "corrected" ring.
    if (selected !== null) shapes.push({ orig: squareKey(selected.r, selected.c), brush: 'green' });
    return shapes;
  });

  const lowLight = $derived(reading?.warnings.includes(WARN_LOW_LIGHT) ?? false);

  // ---- stage tables ---------------------------------------------------------
  // Each one is a pure function of the read (`./tables`); this is only the wiring.

  const captureTable = $derived(image === null || facts === null ? [] : captureRows(image, facts));
  const readRows = $derived(readTable(reading, estimateMs));
  const repairRows = $derived(repairTable(reading?.prior ?? null));
  const materialRows = $derived(shownGrid === null ? [] : materialTable(shownGrid));

  const lowChips = $derived.by(() => {
    const grid = baseGrid;
    if (dist === null || grid === null) return [];
    return weakestSquares(dist).map((entry) => {
      const cls = shownClassAt(grid, corrections, entry.r, entry.c);
      return { ...entry, symbol: cls === 'empty' ? '.' : cls };
    });
  });

  // ---- the source canvas ----------------------------------------------------

  function sourceCanvas(): HTMLCanvasElement {
    if (source === null) source = document.createElement('canvas');
    return source;
  }

  function context(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
    const ctx = canvas.getContext('2d');
    // A 2d context is not optional on any browser this page runs in; a null here is a
    // dead page, not a condition to degrade around.
    if (ctx === null) throw new Error('this browser gave us no 2d canvas context');
    return ctx;
  }

  function canvasJpeg(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error('the browser could not encode this photo'))),
        'image/jpeg',
        quality
      );
    });
  }

  function decodeImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error('the photo could not be decoded'));
      element.src = url;
    });
  }

  /**
   * Draw a decoded photo into the source canvas at the posted resolution, and measure
   * everything the capture stage reports.
   */
  async function adoptPhoto(
    element: HTMLImageElement,
    describe: (scale: number) => CaptureFacts
  ): Promise<void> {
    const canvas = sourceCanvas();
    const scale = Math.min(1, MAX_SOURCE_PX / Math.max(element.naturalWidth, element.naturalHeight));
    canvas.width = Math.round(element.naturalWidth * scale);
    canvas.height = Math.round(element.naturalHeight * scale);
    const ctx = context(canvas);
    ctx.drawImage(element, 0, 0, canvas.width, canvas.height);
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const bytes = (await canvasJpeg(canvas, POST_QUALITY)).size;
    image = {
      width: canvas.width,
      height: canvas.height,
      dataUrl: canvas.toDataURL('image/jpeg', THUMB_QUALITY),
      bytes,
      meanLuma: meanLuma({ data: pixels.data, width: canvas.width, height: canvas.height })
    };
    facts = describe(scale);

    // A new photo invalidates every stage and everything read off the old one.
    reading = null;
    corrections = NO_CORRECTIONS;
    selected = null;
    savedLabel = null;
    activeReadId = null;
    estimateMs = null;
    hideBanner();
    pipeline = withStage(resetFrom(pipeline, 'capture'), 'capture', 'ok', captureSummary(image));
  }

  // ---- 2-4. estimate + render ----------------------------------------------

  async function runRead(): Promise<void> {
    if (image === null) return;
    busy = 'estimating…';
    failure = '';
    runState = 'reading';
    pipeline = withStages(pipeline, READ_STAGES, 'run', 'estimating…');
    const started = performance.now();
    try {
      const blob = await canvasJpeg(sourceCanvas(), POST_QUALITY);
      const result = await estimate(blob, 'snapshot');
      estimateMs = performance.now() - started;
      if (!result.ok) {
        failure = `estimate failed: ${result.error.message}`;
        pipeline = withStages(pipeline, READ_STAGES, 'fail', 'read failed');
        runState = 'failed';
        return;
      }
      applyReading(result.value, null);
      runState = 'done';
      // This read may have added a photo to the node's retained set.
      await refreshUploads();
    } catch (cause) {
      failure = `${String(cause)}\nIs the inference node reachable from the viewer?`;
      pipeline = withStages(pipeline, READ_STAGES, 'fail', 'read failed');
      runState = 'failed';
    } finally {
      busy = null;
    }
  }

  function applyReading(response: EstimateResponse, restored: SessionRead | null): void {
    reading = response;
    // Point the correction sync at the right entry BEFORE anything renders: a fresh read
    // has no entry yet (`pushRead` makes it), a restored one owns its own.
    activeReadId = restored?.id ?? null;
    selected = null;
    savedLabel = null;

    // A new read is a fresh baseline; a restored one brings back the corrections that
    // were made against it, so switching between reads never loses hand work.
    corrections = restored?.corrections ?? NO_CORRECTIONS;

    const summary = confidenceSummary(response.event.square_dist);
    pipeline = withStage(
      pipeline,
      'read',
      summary.low ? 'warn' : 'ok',
      `mean per-square confidence ${fmt(summary.mean, 3)} · ${summary.low} square(s) below ` +
        `${LOW_CONFIDENCE} · position confidence ${fmt(response.event.confidence, 3)}`
    );

    const prior = response.prior;
    pipeline = withStage(
      pipeline,
      'repair',
      prior === null ? 'idle' : prior.flips.length ? 'warn' : 'ok',
      prior === null
        ? 'no repair reported by the server'
        : prior.flips.length
          ? `${prior.flips.length} square(s) relabelled to reach a possible board` +
            (Object.keys(prior.residual).length ? ' — some impossibility remains' : '')
          : 'the read was already a possible board — nothing changed'
    );
    restateOutput();
    hideBanner();

    if (restored === null) pushRead(response);
    else markRestored(restored);
  }

  /** The output stage restates itself on every change to the board as shown. */
  function restateOutput(): void {
    if (shownPlacement === null) return;
    pipeline = withStage(
      pipeline,
      'output',
      corrections.size ? 'warn' : 'ok',
      `${shownPlacement}${corrections.size ? ` · ${corrections.size} hand correction(s)` : ''}`
    );
  }

  // ---- corrections ----------------------------------------------------------

  function setCorrections(next: Corrections): void {
    corrections = next;
    restateOutput();
    // Corrections belong to the read they were made against, so leaving a history entry
    // and coming back preserves them.
    reads = reads.map((read) => (read.id === activeReadId ? { ...read, corrections: next } : read));
  }

  function assign(cls: ClassName): void {
    if (selected === null) return;
    const scores = scoresAt(dist, selected.r, selected.c);
    if (scores === null) return;
    setCorrections(
      withCorrection(corrections, squareKey(selected.r, selected.c), assignmentFor(scores, cls))
    );
  }

  function cycle(r: number, c: number): void {
    const scores = scoresAt(dist, r, c);
    if (scores === null) return;
    const square = squareKey(r, c);
    setCorrections(withCorrection(corrections, square, cycleTarget(scores, corrections.get(square))));
  }

  function resetCorrections(): void {
    setCorrections(NO_CORRECTIONS);
  }

  /** Opening a square never changes it; clicking it again toggles through its top-3. */
  function selectSquare(r: number, c: number): void {
    if (dist === null) return;
    selected = { r: Math.min(7, Math.max(0, r)), c: Math.min(7, Math.max(0, c)) };
  }

  function pickSquare(r: number, c: number): void {
    if (selected !== null && selected.r === r && selected.c === c) cycle(r, c);
    else selectSquare(r, c);
  }

  function stepSquare(delta: number): void {
    if (selected === null) return;
    const index = (selected.r * GRID + selected.c + delta + 64) % 64;
    selectSquare(Math.floor(index / GRID), index % GRID);
  }

  function nextLowConfidence(): void {
    const low = weakestSquares(dist);
    if (low.length === 0) return;
    const current = selected === null ? -1 : selected.r * GRID + selected.c;
    const next = low.find((entry) => entry.r * GRID + entry.c > current) ?? low[0];
    if (next !== undefined) selectSquare(next.r, next.c);
  }

  /**
   * Tap = tap, not touch-capture. A plain click listener never touches `touch-action` or
   * calls `preventDefault`, so a swipe that starts on the board still scrolls the page;
   * only an actual (non-dragged) tap fires a click at all. chessground stays `viewOnly`,
   * which is also why the square has to be worked out from the rect rather than from its
   * `select` event — `viewOnly` binds no interaction handlers at all.
   */
  function boardClick(event: MouseEvent): void {
    if (dist === null) return;
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return;
    pickSquare(
      Math.min(7, Math.max(0, Math.floor((y / rect.height) * GRID))),
      Math.min(7, Math.max(0, Math.floor((x / rect.width) * GRID)))
    );
  }

  // ---- ground truth ---------------------------------------------------------

  const uploadId = $derived(reading?.upload ?? null);
  const groundTruth = $derived(labelState(uploadId, shownPlacement ?? '', savedLabel));

  async function saveGroundTruth(): Promise<void> {
    if (uploadId === null || shownPlacement === null) return;
    submittingLabel = true;
    try {
      const result = await saveLabel(uploadId, {
        placement: shownPlacement,
        corrections: corrections.size
      });
      if (!result.ok) {
        failure = `could not save the label: ${result.error.message}`;
        return;
      }
      savedLabel = shownPlacement;
      showBanner(
        'info',
        corrections.size
          ? `Labelling saved with ${corrections.size} correction(s) — this board can now score the reader.`
          : 'Labelling saved — the model read this board exactly right.'
      );
      // The node now reports this photo as labelled.
      await refreshUploads();
    } finally {
      submittingLabel = false;
    }
  }

  // ---- history --------------------------------------------------------------

  function pushRead(response: EstimateResponse): void {
    if (image === null || facts === null) return;
    const id = ++readSeq;
    reads = [
      {
        id,
        at: new Date(),
        response,
        corrections,
        image,
        facts,
        estimateMs
      },
      ...reads
    ].slice(0, READS_CAP);
    activeReadId = id;
  }

  function markRestored(read: SessionRead): void {
    activeReadId = read.id;
  }

  async function refreshUploads(): Promise<void> {
    const result = await listUploads();
    if (!result.ok) {
      // Degrade to the session list — never break the page over a missing endpoint.
      uploads = [];
      retention = null;
      return;
    }
    uploads = result.value.uploads;
    retention = result.value.retention;
  }

  /**
   * Put the whole page back into the state that read produced: capture from the stored
   * photo, the rest from the stored event — so the evidence on screen always belongs to
   * one and the same read, never a mix of two.
   */
  async function restoreRead(id: number): Promise<void> {
    const read = reads.find((entry) => entry.id === id);
    if (read === undefined || (read.id === activeReadId && reading === read.response)) return;
    failure = '';
    const element = await decodeImage(read.image.dataUrl);
    const canvas = sourceCanvas();
    canvas.width = read.image.width;
    canvas.height = read.image.height;
    context(canvas).drawImage(element, 0, 0);
    image = read.image;
    facts = read.facts;
    estimateMs = read.estimateMs;
    pipeline = withStage(pipeline, 'capture', 'ok', captureSummary(read.image));
    applyReading(read.response, read);
    runState = 'restored';
  }

  /**
   * A photo the node holds but this session never read: there is no event and no
   * distribution to restore, so the pipeline is rebuilt honestly — capture from THIS
   * image, then a real read, which lands in the session list like any other. Nothing
   * is faked.
   */
  async function loadUpload(upload: UploadRecord): Promise<void> {
    failure = '';
    runState = 'loading photo';
    busy = 'loading photo…';
    let objectUrl: string | null = null;
    try {
      // Fetched as a blob rather than assigned straight to `img.src`: this page reads the
      // canvas back (luma, thumbnail), and only a same-origin (blob:) image leaves it
      // untainted.
      const response = await fetch(uploadPhotoUrl(upload.id), { cache: 'no-store' });
      if (!response.ok) throw new Error(`photo ${response.status}`);
      objectUrl = URL.createObjectURL(await response.blob());
      const element = await decodeImage(objectUrl);
      const at = upload.ts === null ? null : new Date(upload.ts * 1000).toLocaleString();
      await adoptPhoto(element, (k) => ({
        origin: `retained upload ${upload.id} on the inference node`,
        retainedAt: at,
        originalWidth: element.naturalWidth,
        originalHeight: element.naturalHeight,
        scale: k
      }));
      await runRead();
    } catch (cause) {
      failure = `${String(cause)}\nIs the inference node reachable from the viewer?`;
      runState = 'failed';
    } finally {
      if (objectUrl !== null) URL.revokeObjectURL(objectUrl);
      busy = null;
    }
  }

  // ---- capture input --------------------------------------------------------

  async function onFile(event: Event & { currentTarget: HTMLInputElement }): Promise<void> {
    const file = event.currentTarget.files?.[0];
    if (file === undefined) return;
    failure = '';
    const objectUrl = URL.createObjectURL(file);
    try {
      const element = await decodeImage(objectUrl);
      await adoptPhoto(element, (scale) => ({
        origin: `${file.name || '(camera)'} · ${kb(file.size)}`,
        retainedAt: null,
        originalWidth: element.naturalWidth,
        originalHeight: element.naturalHeight,
        scale
      }));
    } catch (cause) {
      failure = String(cause);
      return;
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
    await runRead();
  }

  // ---- banners --------------------------------------------------------------

  function showBanner(kind: 'info' | 'warn', text: string): void {
    banner = { kind, text };
    if (bannerTimer !== null) clearTimeout(bannerTimer);
    // Non-blocking, self-dismissing: a banner never stands between the operator and the
    // board it is talking about.
    bannerTimer = setTimeout(() => (banner = null), 6000);
  }

  function hideBanner(): void {
    if (bannerTimer !== null) clearTimeout(bannerTimer);
    banner = null;
  }

  async function copy(text: string | null): Promise<void> {
    if (!text) return;
    try {
      await navigator.clipboard?.writeText(text);
    } catch {
      /* a refused clipboard is not worth an error banner — the text is on screen */
    }
  }

  // ---- browser-only wiring --------------------------------------------------

  // The node's retained photos are listed before the first read.
  onMount(() => {
    void refreshUploads();
  });

  $effect(() => {
    const column = boardColumn;
    if (!column) return;
    // chessground measures the box it is handed, so the board gets an explicit px size
    // and heals itself from there via its own ResizeObserver.
    const measure = () => (boardSize = Math.max(160, Math.min(460, column.clientWidth)));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(column);
    return () => observer.disconnect();
  });

  $effect(() => {
    const bar = actionBar;
    if (!bar) return;
    // The fixed phone action bar overlaps the bottom of the page and the inspector sheet
    // docks directly above it; publish its real height instead of hard-coding one.
    const measure = () => (barHeight = Math.round(bar.getBoundingClientRect().height));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(bar);
    return () => observer.disconnect();
  });

  $effect(() => {
    /**
     * Clicking outside the board and the inspector closes it.
     *
     * The check uses the event's dispatch-time `composedPath()`, never a live
     * `element.contains(event.target)`: an inner handler may replace the clicked node,
     * and a detached target makes `contains()` answer false — which used to close the
     * panel on every direct pick. Containers that must not close it mark themselves with
     * `data-keeps-inspector`.
     */
    const onDocumentClick = (event: MouseEvent) => {
      if (selected === null) return;
      const kept = event
        .composedPath()
        .some((node) => node instanceof HTMLElement && node.dataset['keepsInspector'] !== undefined);
      if (!kept) selected = null;
    };
    document.addEventListener('click', onDocumentClick);
    return () => document.removeEventListener('click', onDocumentClick);
  });

  $effect(() => {
    /** The diagnostic fast path on a desktop. FEN letters, so case carries colour. */
    const onKeyDown = (event: KeyboardEvent) => {
      if (selected === null || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === 'Escape') {
        selected = null;
        return;
      }
      const move: Readonly<Record<string, readonly [number, number]>> = {
        ArrowLeft: [0, -1],
        ArrowRight: [0, 1],
        ArrowUp: [-1, 0],
        ArrowDown: [1, 0]
      };
      const step = move[event.key];
      if (step !== undefined) {
        event.preventDefault();
        selectSquare(selected.r + step[0], selected.c + step[1]);
        return;
      }
      const cls = classOfKey(event.key);
      if (cls !== null) {
        event.preventDefault();
        assign(cls);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  });
</script>

<svelte:head>
  <title>board-reader · pipeline inspector</title>
</svelte:head>

<div class="page" style:--bar-h="{barHeight}px">
  <header>
    <h1>board-reader · pipeline inspector</h1>
    <span class="spacer"></span>
    <span class="run-state">{runState}</span>
  </header>

  <main>
    <label class="capture-btn" class:compact={image !== null} for="file">
      {image === null ? 'Take / choose photo' : 'New photo'}
    </label>
    <!-- Deliberately NO capture attribute: with it, iOS Safari skips the chooser and
         forces the camera, making the photo library unreachable. Bare, both iOS and
         Android present the native sheet offering camera AND camera roll. -->
    <input
      type="file"
      id="file"
      accept="image/*"
      class="visually-hidden"
      onchange={onFile}
    />
    <p class="hint">
      Photograph the board — the reader finds it in the whole photo, so there is nothing to
      frame. Every stage below records what it saw.
    </p>

    {#if banner !== null}
      <div class="banner {banner.kind}">{banner.text}</div>
    {/if}
    {#if lowLight}
      <div class="banner warn">Photo is very dark — reads degrade; add light if you can.</div>
    {/if}

    <div class="out-main">
      {#if image !== null}
        <div class="photo-col">
          <div class="col-label">Photo as posted</div>
          <div class="photo-frame">
            <img src={image.dataUrl} alt="the board as posted to the reader" />
          </div>
        </div>
      {/if}

      {#if reading !== null}
        <div class="board-col" bind:this={boardColumn}>
          <div class="col-label">Position read</div>
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <div
            class="board-wrap"
            style:--board-size="{boardSize}px"
            data-keeps-inspector
            onclick={boardClick}
          >
            <Board placement={shownPlacement ?? EMPTY_PLACEMENT} shapes={boardShapes} />
          </div>
          <div class="board-legend">
            <span><i style="background:#e25555"></i> confidence &lt; {LOW_CONFIDENCE}</span>
            <span><i style="background:#6aa8ff"></i> corrected by hand</span>
            <span><i style="background:#7ac77a"></i> inspecting</span>
          </div>
          {#if reading !== null}
            <p class="hint">
              Click a square to inspect it; click it again to toggle through its most likely
              readings.
            </p>
            <GroundTruthForm
              state={groundTruth}
              corrections={corrections.size}
              submitting={submittingLabel}
              onSave={saveGroundTruth}
            />
          {/if}
        </div>
      {/if}

      {#if inspection !== null}
        <div class="inspector-col">
          <SquareInspector
            square={inspection.square}
            scores={inspection.scores}
            shown={inspection.shown}
            corrected={corrections.has(inspection.square)}
            onAssign={assign}
            onRevert={() => setCorrections(withCorrection(corrections, inspection.square, null))}
            onClose={() => (selected = null)}
            onStep={stepSquare}
            onNextLow={nextLowConfidence}
          />
        </div>
      {/if}
    </div>

    <div class="action-bar" bind:this={actionBar}>
      <button type="button" disabled={image === null || busy !== null} onclick={runRead}>
        Read position
      </button>
      {#if busy !== null}<span class="busy">{busy}</span>{/if}
    </div>

    <div class="steps">
      {#each STAGES as stage (stage.id)}
        <PipelineStep {stage} status={pipeline[stage.id]} bind:open={stageOpen[stage.id]}>
          {#if stage.id === 'capture'}
            <p class="note">
              The photo is downscaled to a {MAX_SOURCE_PX} px long side in the browser; that
              downscaled image is exactly what is POSTed to <code>/estimate</code>, so this
              stage describes the very pixels the server reads.
            </p>
            <KvTable rows={captureTable} />
          {:else if stage.id === 'read'}
            <p class="note">
              A 13-class distribution per square, read from the whole photo and always in the
              chess frame (rank 8 on top), stateless and rule-free — an illegal-looking
              board is still reported as <code>ok</code>. Click any cell to open it in the
              inspector.
            </p>
            <KvTable rows={readRows} />
            {#if dist !== null && baseGrid !== null}
              <ConfidenceMap {dist} base={baseGrid} {corrections} {selected} onPick={pickSquare} />
              <div class="note">Least confident squares</div>
              <div class="chips" data-keeps-inspector>
                {#each lowChips as chip (chip.square)}
                  <button
                    class="chip"
                    type="button"
                    onclick={() => selectSquare(chip.r, chip.c)}
                  >
                    {chip.square}
                    {chip.symbol}
                    {fmt(chip.p, 2)}
                  </button>
                {/each}
              </div>
            {/if}
            {#if (reading?.warnings.length ?? 0) > 0}
              <div class="note warn">Server warnings: {reading?.warnings.join(', ')}</div>
            {/if}
          {:else if stage.id === 'repair'}
            <p class="note">
              Some reads describe a board that cannot exist — two white kings, a pawn on the
              back rank, more pieces than a side can own. The server decodes the nearest
              possible board out of the same per-square distribution and returns it alongside
              the raw read; the distribution itself is never rewritten. Every square it
              relabelled is listed here, cheapest first, so the repair is auditable rather than
              silent. The board above shows the repaired position.
            </p>
            <KvTable rows={repairRows} />
          {:else}
            <code class="fen">{shownPlacement ?? ''}</code>
            <div class="row">
              <button
                class="text-btn"
                type="button"
                onclick={() => copy(shownPlacement === null ? null : fullFen(shownPlacement, stm))}
              >
                Copy FEN
              </button>
              <button
                class="text-btn"
                type="button"
                onclick={() => copy(reading === null ? null : rawEventJson(reading))}
              >
                Copy raw event
              </button>
              {#if corrections.size}
                <button class="text-btn" type="button" onclick={resetCorrections}>
                  Reset corrections
                </button>
              {/if}
            </div>
            {#if reading !== null}
              <div class="stm">
                to move:
                <label><input type="radio" bind:group={stm} value="w" /> White</label>
                <label><input type="radio" bind:group={stm} value="b" /> Black</label>
              </div>
              {#if links !== null}
                <div class="links">
                  <a href={links.analysis} target="_blank" rel="noopener">
                    Open in Lichess analysis
                  </a>
                  <a class="secondary" href={links.editor} target="_blank" rel="noopener">
                    Lichess editor
                  </a>
                </div>
              {/if}
              <div class="note">Corrections</div>
              <CorrectionsPanel
                {corrections}
                readAt={(square) => {
                  const at = indexOfSquare(square);
                  const scores = scoresAt(dist, at.r, at.c);
                  return scores === null ? 'empty' : argmaxClass(scores);
                }}
                onRevert={(square) => setCorrections(withCorrection(corrections, square, null))}
              />
              <KvTable rows={materialRows} />
            {/if}
            <div class="note">Raw event</div>
            <pre class="raw">{reading === null ? '' : rawEventJson(reading)}</pre>
          {/if}
        </PipelineStep>
      {/each}
    </div>

    <HistoryList
      {rows}
      note={historyNote(retention)}
      activeId={activeReadId}
      clearable={reads.length > 0}
      onRestore={(id) => void restoreRead(id)}
      onLoadUpload={(upload) => void loadUpload(upload)}
      onClear={() => {
        // This session's entries only — the node-held ones are not ours to drop, and they
        // reappear on the next refresh.
        reads = [];
        activeReadId = null;
      }}
    />

    {#if failure}
      <div class="err">{failure}</div>
    {/if}
  </main>
</div>

<style>
  /* The snapshot page is a DIAGNOSTIC INTERFACE for the reading algorithm, not a wizard.
     One column, in pipeline order: capture → the stages → the position they produced →
     the log of earlier reads. The only thing that ever sits side-by-side is the piece
     switcher next to the board, and only when there is room for both.
     No emoji anywhere: state is text + colour. */
  .page {
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
    background: var(--bg);
    color: var(--fg);
    -webkit-tap-highlight-color: transparent;
    -webkit-user-select: none;
    user-select: none;
    -webkit-touch-callout: none;
    min-height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }

  h1 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
  }

  .spacer {
    flex: 1;
  }

  .run-state {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--dim);
  }

  main {
    padding: 14px 16px calc(var(--bar-h) + 96px);
    max-width: 1240px;
    margin: 0 auto;
  }

  @media (min-width: 760px) {
    main {
      padding-bottom: 40px;
    }
  }

  .hint {
    color: var(--muted);
    font-size: 13px;
    margin: 6px 0;
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
    border: 0;
  }

  /* Big primary capture control: a <label> styled as a button wrapping a visually-hidden
     file input, so the native chooser still fires but the look is ours. */
  .capture-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent);
    color: #fff;
    font-weight: 700;
    font-size: 17px;
    min-height: 60px;
    border-radius: 12px;
    cursor: pointer;
    touch-action: manipulation;
  }

  /* Shrinks once a photo is in hand — the big CTA has done its job. */
  .capture-btn.compact {
    min-height: 44px;
    font-size: 14px;
    border-radius: 9px;
  }

  /* Input and output, side by side and the same size: the photo left of the board it
     produced, with the piece switcher after them when the window has room. The photo
     never leaves the screen — it is what every stage is talking about. */
  .out-main {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-start;
    margin-top: 14px;
  }

  .photo-col,
  .board-col {
    flex: 1 1 340px;
    max-width: 440px;
    min-width: 0;
  }

  .inspector-col {
    flex: 1 1 320px;
    min-width: 0;
  }

  .col-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--dim);
    margin-bottom: 6px;
  }

  /* A square frame, so the photo block matches the board block exactly whatever the
     photo's aspect ratio. The frame's size is definite (width + aspect-ratio), so the
     photo can fit itself with object-fit and needs no measuring. */
  .photo-frame {
    width: 100%;
    aspect-ratio: 1;
    background: #0f1216;
    border: 1px solid var(--line);
    border-radius: 6px;
    overflow: hidden;
  }

  .photo-frame img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  /* viewOnly means no piece dragging, so touch may pan — a thumb landing on the board
     must still scroll the page. */
  .board-wrap {
    width: 100%;
    touch-action: manipulation;
  }

  .board-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 12px;
    color: var(--muted);
    margin: 8px 0;
  }

  .board-legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .board-legend i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }

  /* Banners: non-blocking, near the board. Text only — state is carried by colour. */
  .banner {
    font-size: 13px;
    font-weight: 600;
    padding: 9px 12px;
    border-radius: 8px;
    margin-bottom: 10px;
  }

  .banner.info {
    background: #1f3a24;
    color: var(--ok);
  }

  .banner.warn {
    background: #2a2410;
    color: var(--warn);
    border: 1px solid #4a3f10;
  }

  .err {
    color: var(--bad);
    font-size: 13px;
    white-space: pre-wrap;
    margin-top: 12px;
  }

  .action-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 12px 0;
  }

  .action-bar button {
    flex: 1 1 auto;
  }

  .busy {
    font-size: 13px;
    color: var(--muted);
    white-space: nowrap;
  }

  @media (max-width: 759px) {
    /* Phone: the run controls stay reachable without scrolling out of the pipeline. */
    .action-bar {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 30;
      margin: 0;
      padding: 8px 16px calc(8px + env(safe-area-inset-bottom));
      background: var(--panel);
      border-top: 1px solid var(--line);
      box-shadow: 0 -6px 20px rgba(0, 0, 0, 0.4);
    }
  }

  button {
    background: var(--accent);
    color: #fff;
    border: 0;
    border-radius: 9px;
    padding: 10px 16px;
    font: inherit;
    font-size: 15px;
    font-weight: 600;
    min-height: 48px;
    touch-action: manipulation;
    cursor: pointer;
  }

  button:disabled {
    opacity: 0.4;
    cursor: default;
  }

  /* Occasional side actions, deliberately NOT the >=48px treatment. */
  .text-btn {
    background: none;
    border: 0;
    color: var(--accent-soft);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    min-height: auto;
    padding: 4px 0;
    text-decoration: underline;
    touch-action: manipulation;
  }

  .steps {
    margin-top: 18px;
  }

  .note {
    font-size: 12.5px;
    color: var(--muted);
    margin: 8px 0;
  }

  .note.warn {
    color: var(--warn);
  }

  .row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin: 10px 0;
  }

  .fen {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: #c8cdd4;
    word-break: break-all;
    -webkit-user-select: text;
    user-select: text;
    background: #0f1216;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 8px;
    margin: 8px 0;
    display: block;
  }

  .raw {
    background: #0f1216;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    line-height: 1.4;
    max-height: 320px;
    overflow: auto;
    -webkit-user-select: text;
    user-select: text;
    margin: 8px 0;
  }

  .stm {
    margin: 10px 0;
    font-size: 13px;
    color: var(--muted);
  }

  .stm label {
    margin-right: 16px;
    padding: 6px 0;
    display: inline-block;
    touch-action: manipulation;
    color: var(--fg);
  }

  .links a {
    display: inline-flex;
    align-items: center;
    min-height: 44px;
    background: #1f3a24;
    color: var(--ok);
    text-decoration: none;
    padding: 8px 14px;
    border-radius: 9px;
    margin: 6px 8px 0 0;
    font-weight: 600;
    font-size: 14px;
    touch-action: manipulation;
  }

  .links a.secondary {
    background: #333842;
    color: #c8cdd4;
  }

  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
  }

  .chip {
    font-family: ui-monospace, monospace;
    font-size: 12px;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 4px 8px;
    color: var(--fg);
    min-height: 32px;
    font-weight: 500;
    touch-action: manipulation;
  }
</style>
