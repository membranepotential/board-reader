/**
 * The state of one `/snapshot` visit: the photo, the pipeline stages that describe it,
 * the board read off it (`BoardState`), and the log of earlier reads.
 *
 * One invariant: **one image.** The photo is downscaled once and *that* image is what is
 * posted to `/estimate` and what the capture stage describes.
 *
 * Created by the page in its script, so the instance lives and dies with the page; the
 * canvas is only touched from event handlers, never during SSR.
 */
import { estimate, listUploads, saveLabel, uploadPhotoUrl } from '#lib/api.ts';
import { BoardState } from '#lib/components/snapshot/board-state.svelte.ts';
import {
  canvasJpeg,
  decodeImage,
  drawSource,
  restoreSource
} from '#lib/components/snapshot/canvas.ts';
import {
  READS_CAP,
  type SessionRead
} from '#lib/components/snapshot/history.ts';
import {
  POST_QUALITY,
  captureSummary,
  kb,
  type CaptureFacts,
  type SourceImage
} from '#lib/components/snapshot/photo.ts';
import {
  READ_STAGES,
  initialPipeline,
  resetFrom,
  withStage,
  withStages,
  type Pipeline,
  type StageId
} from '#lib/components/snapshot/pipeline.ts';
import {
  NO_CORRECTIONS,
  confidenceSummary,
  fmt,
  labelState,
  type SideToMove
} from '#lib/components/snapshot/reading.ts';
import {
  LOW_CONFIDENCE,
  type EstimateResponse,
  type UploadRecord
} from '#lib/protocol/index.ts';

export interface Banner {
  readonly kind: 'info' | 'warn';
  readonly text: string;
}

export class SnapshotSession {
  image = $state<SourceImage | null>(null);
  facts = $state<CaptureFacts | null>(null);
  pipeline: Pipeline = $state(initialPipeline());
  /** Which stage bodies are expanded. Everything starts collapsed. */
  stageOpen: Record<StageId, boolean> = $state({ capture: false, read: false, output: false });

  /** The read position, its corrections and the inspected square. */
  readonly board = new BoardState((next) => {
    this.#restateOutput();
    // Corrections belong to the read they were made against, so leaving a history entry
    // and coming back preserves them.
    this.reads = this.reads.map((read) =>
      read.id === this.activeReadId ? { ...read, corrections: next } : read
    );
  });
  savedLabel = $state<string | null>(null);
  submittingLabel = $state(false);
  stm: SideToMove = $state('w');

  runState = $state('idle');
  busy = $state<string | null>(null);
  failure = $state('');
  banner = $state<Banner | null>(null);
  estimateMs = $state<number | null>(null);

  reads = $state<readonly SessionRead[]>([]);
  uploads = $state<readonly UploadRecord[]>([]);
  retention = $state<boolean | null>(null);
  activeReadId = $state<number | null>(null);

  /** The photo lives in an offscreen canvas: the posted JPEG and the capture stage both read it. */
  #source: HTMLCanvasElement | null = null;
  #bannerTimer: ReturnType<typeof setTimeout> | null = null;
  #readSeq = 0;

  readonly groundTruth = $derived(
    labelState(this.board.reading?.upload ?? null, this.board.shownPlacement ?? '', this.savedLabel)
  );

  // ---- the photo, and the read of it ----------------------------------------

  #sourceCanvas(): HTMLCanvasElement {
    if (this.#source === null) this.#source = document.createElement('canvas');
    return this.#source;
  }

  /** Draw a decoded photo into the source canvas; a new photo invalidates every stage. */
  async #adoptPhoto(
    element: HTMLImageElement,
    describe: (scale: number) => CaptureFacts
  ): Promise<void> {
    const { image, scale } = await drawSource(this.#sourceCanvas(), element);
    this.image = image;
    this.facts = describe(scale);

    this.board.clear();
    this.savedLabel = null;
    this.activeReadId = null;
    this.estimateMs = null;
    this.hideBanner();
    this.pipeline = withStage(resetFrom(this.pipeline, 'capture'), 'capture', 'ok', captureSummary(image));
  }

  /** The user chose a photo: decode it, describe it, read it. */
  async loadFile(file: File): Promise<void> {
    this.failure = '';
    const objectUrl = URL.createObjectURL(file);
    try {
      const element = await decodeImage(objectUrl);
      await this.#adoptPhoto(element, (scale) => ({
        origin: `${file.name || '(camera)'} · ${kb(file.size)}`,
        retainedAt: null,
        originalWidth: element.naturalWidth,
        originalHeight: element.naturalHeight,
        scale
      }));
    } catch (cause) {
      this.failure = String(cause);
      return;
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
    await this.runRead();
  }

  async runRead(): Promise<void> {
    if (this.image === null) return;
    this.busy = 'estimating…';
    this.failure = '';
    this.runState = 'reading';
    this.pipeline = withStages(this.pipeline, READ_STAGES, 'run', 'estimating…');
    const started = performance.now();
    try {
      const blob = await canvasJpeg(this.#sourceCanvas(), POST_QUALITY);
      const result = await estimate(blob, 'snapshot');
      this.estimateMs = performance.now() - started;
      if (!result.ok) {
        this.failure = `estimate failed: ${result.error.message}`;
        this.pipeline = withStages(this.pipeline, READ_STAGES, 'fail', 'read failed');
        this.runState = 'failed';
        return;
      }
      this.#applyReading(result.value, null);
      this.runState = 'done';
      // This read may have added a photo to the node's retained set.
      await this.refreshUploads();
    } catch (cause) {
      this.failure = `${String(cause)}\nIs the inference node reachable from the viewer?`;
      this.pipeline = withStages(this.pipeline, READ_STAGES, 'fail', 'read failed');
      this.runState = 'failed';
    } finally {
      this.busy = null;
    }
  }

  #applyReading(response: EstimateResponse, restored: SessionRead | null): void {
    // Point the correction sync at the right entry BEFORE anything renders: a fresh read
    // has no entry yet (`#pushRead` makes it), a restored one owns its own.
    this.activeReadId = restored?.id ?? null;
    this.savedLabel = null;

    // A new read is a fresh baseline; a restored one brings back the corrections that
    // were made against it, so switching between reads never loses hand work.
    this.board.load(response, restored?.corrections ?? NO_CORRECTIONS);

    const summary = confidenceSummary(response.event.square_dist);
    this.pipeline = withStage(
      this.pipeline,
      'read',
      summary.low ? 'warn' : 'ok',
      `mean per-square confidence ${fmt(summary.mean, 3)} · ${summary.low} square(s) below ` +
        `${LOW_CONFIDENCE} · position confidence ${fmt(response.event.confidence, 3)}`
    );

    this.#restateOutput();
    this.hideBanner();

    if (restored === null) this.#pushRead(response);
    else this.activeReadId = restored.id;
  }

  /** The output stage restates itself on every change to the board as shown. */
  #restateOutput(): void {
    const placement = this.board.shownPlacement;
    if (placement === null) return;
    const n = this.board.corrections.size;
    this.pipeline = withStage(
      this.pipeline,
      'output',
      n ? 'warn' : 'ok',
      `${placement}${n ? ` · ${n} hand correction(s)` : ''}`
    );
  }

  // ---- ground truth ----------------------------------------------------------

  async saveGroundTruth(): Promise<void> {
    const uploadId = this.board.reading?.upload ?? null;
    const placement = this.board.shownPlacement;
    if (uploadId === null || placement === null) return;
    this.submittingLabel = true;
    try {
      const n = this.board.corrections.size;
      const result = await saveLabel(uploadId, { placement, corrections: n });
      if (!result.ok) {
        this.failure = `could not save the label: ${result.error.message}`;
        return;
      }
      this.savedLabel = placement;
      this.showBanner(
        'info',
        n
          ? `Labelling saved with ${n} correction(s) — this board can now score the reader.`
          : 'Labelling saved — the model read this board exactly right.'
      );
      // The node now reports this photo as labelled.
      await this.refreshUploads();
    } finally {
      this.submittingLabel = false;
    }
  }

  // ---- history ----------------------------------------------------------------

  #pushRead(response: EstimateResponse): void {
    if (this.image === null || this.facts === null) return;
    const id = ++this.#readSeq;
    this.reads = [
      {
        id,
        at: new Date(),
        response,
        corrections: this.board.corrections,
        image: this.image,
        facts: this.facts,
        estimateMs: this.estimateMs
      },
      ...this.reads
    ].slice(0, READS_CAP);
    this.activeReadId = id;
  }

  async refreshUploads(): Promise<void> {
    const result = await listUploads();
    if (!result.ok) {
      // Degrade to the session list — never break the page over a missing endpoint.
      this.uploads = [];
      this.retention = null;
      return;
    }
    this.uploads = result.value.uploads;
    this.retention = result.value.retention;
  }

  /**
   * Put the whole page back into the state that read produced: capture from the stored
   * photo, the rest from the stored event — so the evidence on screen always belongs to
   * one and the same read, never a mix of two.
   */
  async restoreRead(id: number): Promise<void> {
    const read = this.reads.find((entry) => entry.id === id);
    if (read === undefined || (read.id === this.activeReadId && this.board.reading === read.response)) return;
    this.failure = '';
    await restoreSource(this.#sourceCanvas(), read.image);
    this.image = read.image;
    this.facts = read.facts;
    this.estimateMs = read.estimateMs;
    this.pipeline = withStage(this.pipeline, 'capture', 'ok', captureSummary(read.image));
    this.#applyReading(read.response, read);
    this.runState = 'restored';
  }

  /**
   * A photo the node holds but this session never read: there is no event and no
   * distribution to restore, so the pipeline is rebuilt honestly — capture from THIS
   * image, then a real read, which lands in the session list like any other. Nothing
   * is faked.
   */
  async loadUpload(upload: UploadRecord): Promise<void> {
    this.failure = '';
    this.runState = 'loading photo';
    this.busy = 'loading photo…';
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
      await this.#adoptPhoto(element, (k) => ({
        origin: `retained upload ${upload.id} on the inference node`,
        retainedAt: at,
        originalWidth: element.naturalWidth,
        originalHeight: element.naturalHeight,
        scale: k
      }));
      await this.runRead();
    } catch (cause) {
      this.failure = `${String(cause)}\nIs the inference node reachable from the viewer?`;
      this.runState = 'failed';
    } finally {
      if (objectUrl !== null) URL.revokeObjectURL(objectUrl);
      this.busy = null;
    }
  }

  /** This session's entries only — the node-held ones are not ours to drop. */
  clearReads(): void {
    this.reads = [];
    this.activeReadId = null;
  }

  // ---- banners -----------------------------------------------------------------

  showBanner(kind: Banner['kind'], text: string): void {
    this.banner = { kind, text };
    if (this.#bannerTimer !== null) clearTimeout(this.#bannerTimer);
    // Non-blocking, self-dismissing: a banner never stands between the operator and the
    // board it is talking about.
    this.#bannerTimer = setTimeout(() => (this.banner = null), 6000);
  }

  hideBanner(): void {
    if (this.#bannerTimer !== null) clearTimeout(this.#bannerTimer);
    this.banner = null;
  }
}
