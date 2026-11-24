import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import * as wire from './generated';

/**
 * The samples `tests/test_protocol_ts.py` runs through pydantic, run through the
 * generated validators: each must be accepted or refused as it says, on both
 * sides, so the two ends agree on the edge cases the schema alone cannot show
 * (strict scalars, the `x-check` rules, base64, the union discriminators).
 */
const Sample = z.strictObject({
  note: z.string(),
  message: z.string(),
  valid: z.boolean(),
  value: z.unknown()
});
const samples = z
  .array(Sample)
  .parse(JSON.parse(readFileSync(new URL('./samples.json', import.meta.url), 'utf8')));

function validator(name: string): z.ZodType {
  const found: unknown = (wire as Record<string, unknown>)[name];
  if (!(found instanceof z.ZodType)) throw new Error(`no generated validator ${name}`);
  return found;
}

describe('the generated validators, on the samples pydantic judges too', () => {
  it.each(samples.map((s) => [`${s.message}: ${s.note}`, s] as const))('%s', (_label, sample) => {
    expect(validator(sample.message).safeParse(sample.value).success).toBe(sample.valid);
  });
});
