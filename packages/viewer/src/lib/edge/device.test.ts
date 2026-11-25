import { describe, expect, it, vi } from 'vitest';
import { CameraEntry } from '#lib/protocol/generated.ts';
import { DEVICE_ID_KEY, deviceIdentity, randomDeviceId, shortDeviceId, type KeyValueStore } from './device';

function memory(seed: Record<string, string> = {}): KeyValueStore & { data: Record<string, string> } {
  const data = { ...seed };
  return { data, getItem: (key) => data[key] ?? null, setItem: (key, value) => void (data[key] = value) };
}

const ids = (...made: string[]) => () => made.shift() ?? 'exhausted-id';

describe('the device id', () => {
  it('is made once and kept', () => {
    const store = memory();
    expect(deviceIdentity(() => store, ids('first-device-id'))).toEqual({ id: 'first-device-id', kept: true });
    expect(store.data[DEVICE_ID_KEY]).toBe('first-device-id');
    expect(deviceIdentity(() => store, ids('second-device-id'))).toEqual({ id: 'first-device-id', kept: true });
  });

  it('replaces a kept value the node would refuse', () => {
    const store = memory({ [DEVICE_ID_KEY]: 'not/an id' });
    expect(deviceIdentity(() => store, ids('fresh-device-id')).id).toBe('fresh-device-id');
  });

  it('lives only as long as the page where storage throws', () => {
    const blocked = () => {
      throw new DOMException('denied', 'SecurityError');
    };
    expect(deviceIdentity(blocked, ids('session-device-id'))).toEqual({ id: 'session-device-id', kept: false });
    const full: KeyValueStore = {
      getItem: () => null,
      setItem: () => {
        throw new DOMException('quota', 'QuotaExceededError');
      }
    };
    expect(deviceIdentity(() => full, ids('session-device-id')).kept).toBe(false);
  });

  it('is random and one the node accepts, with or without randomUUID', () => {
    const made = randomDeviceId();
    expect(CameraEntry.shape.device_id.safeParse(made).success).toBe(true);
    expect(randomDeviceId()).not.toBe(made);
    vi.stubGlobal('crypto', { getRandomValues: (a: Uint8Array) => a.fill(171) });
    expect(randomDeviceId()).toBe('ab'.repeat(16));
    vi.unstubAllGlobals();
  });

  it('reads as its first eight characters', () => {
    expect(shortDeviceId('kitchen-phone-id')).toBe('kitchen-');
  });
});
