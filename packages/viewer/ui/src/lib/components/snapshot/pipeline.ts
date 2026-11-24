/**
 * The pipeline, as data.
 *
 * `/snapshot` is a DIAGNOSTIC INTERFACE for the reading algorithm, not a wizard: it
 * renders one read as an ordered list of stages, each of which always exists, always
 * carries a state and a headline, and holds its full evidence behind a collapsed
 * body. Nothing is gated behind an opt-in checkbox — diagnostics are the point.
 *
 * The hand-written page drove those stages by writing `data-state` attributes and
 * `textContent` into fixed `<section>`s (`setStep`/`resetSteps`/`openStep`).
 * Here the same model is a value: one immutable record from stage id to
 * `{state, summary}`, updated by returning a new record. That is what lets a history
 * entry restore a whole pipeline by assignment, and what makes "which stages did a
 * failed read fail?" a thing the type system can answer.
 */

/**
 * Four stages: the recognizer reads the whole photo, so there is no detection,
 * rectification or orientation step left for this page to show.
 */
export const STAGE_ORDER = ['capture', 'read', 'repair', 'output'] as const;

export type StageId = (typeof STAGE_ORDER)[number];

/**
 * What a stage is currently saying about itself. `warn` is not a failure: it is the
 * stage's own "look at this" — a dim photo, a hand-corrected square, a board the
 * server had to repair.
 */
export type StageState = 'idle' | 'run' | 'ok' | 'warn' | 'fail';

/** The word shown in the stage's chip. State is text + colour; no emoji anywhere. */
export const CHIP_LABEL: Readonly<Record<StageState, string>> = {
  idle: 'idle',
  run: 'running',
  ok: 'ok',
  warn: 'check',
  fail: 'failed'
};

export interface StageStatus {
  readonly state: StageState;
  /** One line, always present — the stage's headline, shown in the summary row. */
  readonly summary: string;
}

export interface StageMeta {
  readonly id: StageId;
  /** 1-based, as shown. */
  readonly index: number;
  readonly title: string;
  /** What the stage says before anything has happened at all. */
  readonly initialSummary: string;
}

export const STAGES: readonly StageMeta[] = [
  { id: 'capture', index: 1, title: 'Capture', initialSummary: 'no photo yet' },
  { id: 'read', index: 2, title: 'Read', initialSummary: 'waiting for a photo' },
  { id: 'repair', index: 3, title: 'Impossible-board repair', initialSummary: 'waiting for a read' },
  { id: 'output', index: 4, title: 'Position output', initialSummary: 'waiting for a read' }
];

export type Pipeline = Readonly<Record<StageId, StageStatus>>;

/** Every stage idle, each with the line it carries before the first photo. */
export function initialPipeline(): Pipeline {
  const stages = {} as Record<StageId, StageStatus>;
  for (const stage of STAGES) {
    stages[stage.id] = { state: 'idle', summary: stage.initialSummary };
  }
  return stages;
}

/** One stage restated. `summary` omitted keeps the line the stage already had. */
export function withStage(
  pipeline: Pipeline,
  id: StageId,
  state: StageState,
  summary?: string
): Pipeline {
  const current = pipeline[id];
  return { ...pipeline, [id]: { state, summary: summary ?? current.summary } };
}

/** The same state and line applied to several stages — a read starting, or failing. */
export function withStages(
  pipeline: Pipeline,
  ids: readonly StageId[],
  state: StageState,
  summary: string
): Pipeline {
  let next = pipeline;
  for (const id of ids) next = withStage(next, id, state, summary);
  return next;
}

/**
 * `id` and everything downstream of it back to idle — a new photo invalidates every
 * stage from capture on, and nothing upstream of the change is touched.
 */
export function resetFrom(pipeline: Pipeline, id: StageId): Pipeline {
  const start = STAGE_ORDER.indexOf(id);
  let next = pipeline;
  for (const stage of STAGE_ORDER.slice(start)) {
    next = withStage(next, stage, 'idle', 'waiting');
  }
  return next;
}

/** The three stages a `/estimate` call owns, in order. */
export const READ_STAGES: readonly StageId[] = ['read', 'repair', 'output'];
