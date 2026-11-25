/**
 * The phone's own id, so the node gives the same phone the same board id again.
 *
 * A random id (`crypto.randomUUID`) made on the first visit and kept in
 * `localStorage`. Where storage is unavailable (a private window, blocked site
 * data, an accessor that throws), the id lives only as long as the page, and the
 * page says so: a reload then makes a new camera with a new board id.
 */

import { CameraEntry } from '#lib/protocol/generated.ts';

export const DEVICE_ID_KEY = 'board-reader.device-id';

export interface DeviceIdentity {
  readonly id: string;
  /** Whether the id survives a reload (it is in `localStorage`). */
  readonly kept: boolean;
}

/** Just the part of `Storage` this needs, so a test can pass a plain object. */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** The node's rule for a device id (`board_reader_core.control.DeviceId`). */
const DEVICE_ID = CameraEntry.shape.device_id;

/**
 * This browser's device id: the kept one, else a new one kept from now on.
 * `store` is a thunk because merely reading `window.localStorage` throws where
 * site data is blocked.
 */
export function deviceIdentity(store: () => KeyValueStore, newId: () => string): DeviceIdentity {
  try {
    const storage = store();
    const kept = storage.getItem(DEVICE_ID_KEY);
    if (kept !== null && DEVICE_ID.safeParse(kept).success) return { id: kept, kept: true };
    const id = newId();
    storage.setItem(DEVICE_ID_KEY, id);
    return { id, kept: storage.getItem(DEVICE_ID_KEY) === id };
  } catch {
    return { id: newId(), kept: false };
  }
}

/**
 * A new random device id. `crypto.randomUUID` exists only in a secure context, and
 * a phone on plain http still loads the page (to be told why its camera is
 * blocked), so outside one the id is 16 random bytes in hex.
 */
export function randomDeviceId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** How a device id reads in a list: its first eight characters. */
export function shortDeviceId(id: string): string {
  return id.slice(0, 8);
}
