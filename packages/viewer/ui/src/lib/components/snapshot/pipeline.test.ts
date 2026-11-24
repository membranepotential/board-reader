import { describe, expect, it } from 'vitest';
import {
  READ_STAGES,
  STAGES,
  STAGE_ORDER,
  initialPipeline,
  resetFrom,
  withStage,
  withStages
} from './pipeline';

describe('the pipeline record', () => {
  it('describes every stage exactly once, in display order', () => {
    expect(STAGES.map((s) => s.id)).toEqual([...STAGE_ORDER]);
    expect(STAGES.map((s) => s.index)).toEqual([1, 2, 3, 4]);
  });

  it('starts idle with each stage carrying its own opening line', () => {
    const pipeline = initialPipeline();
    for (const stage of STAGES) {
      expect(pipeline[stage.id]).toEqual({ state: 'idle', summary: stage.initialSummary });
    }
  });

  it('never mutates: restating a stage leaves the previous record untouched', () => {
    const before = initialPipeline();
    const after = withStage(before, 'read', 'ok', 'mean confidence 0.9');
    expect(before['read']).toEqual({ state: 'idle', summary: 'waiting for a photo' });
    expect(after['read']).toEqual({ state: 'ok', summary: 'mean confidence 0.9' });
    expect(after['capture']).toBe(before['capture']);
  });

  it('keeps the existing line when only the state changes', () => {
    const pipeline = withStage(initialPipeline(), 'repair', 'warn');
    expect(pipeline['repair']).toEqual({ state: 'warn', summary: 'waiting for a read' });
  });

  it('restates a whole read at once', () => {
    const pipeline = withStages(initialPipeline(), READ_STAGES, 'run', 'estimating…');
    for (const id of READ_STAGES) expect(pipeline[id].state).toBe('run');
    expect(pipeline['capture'].state).toBe('idle');
  });

  it('resets a stage and everything downstream of it, and nothing upstream', () => {
    let pipeline = initialPipeline();
    for (const id of STAGE_ORDER) pipeline = withStage(pipeline, id, 'ok', 'done');
    const reset = resetFrom(pipeline, 'read');
    expect(reset['capture']).toEqual({ state: 'ok', summary: 'done' });
    for (const id of STAGE_ORDER.slice(1)) {
      expect(reset[id]).toEqual({ state: 'idle', summary: 'waiting' });
    }
  });
});
