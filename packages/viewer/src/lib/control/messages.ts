/**
 * The camera control channel's upward messages (`board_reader_core.control`).
 *
 Three sockets, relayed same-origin by the viewer under `/ws`:
 *
 *  - the phone's, `cameraSocket(deviceId)`: it sends `report` (text) and, only while
 *    the node asks, preview JPEGs (binary); it hears a `PhoneDown`.
 *  - the camera list's, `CAMERAS_SOCKET`: it sends `rename` and `forget`; it hears a
 *    `CamerasDown`.
 *  - a control page's, one per board, `controlSocket(id)`: it sends `configure`; it
 *    hears a `ControlDown`.
 *
 * Every text message is a JSON object whose `type` names it. The message types and
 * the validators for what comes down are generated from the Python models
 * (`#lib/protocol/generated`) and read with `parseWire`.
 */

import type {
  CameraConfig,
  CameraReport,
  ConfigureMessage,
  ForgetMessage,
  RenameMessage,
  ReportMessage
} from '#lib/protocol/generated.ts';

export function reportMessage(report: CameraReport): string {
  const message: ReportMessage = { type: 'report', report };
  return JSON.stringify(message);
}

export function configureMessage(config: CameraConfig): string {
  const message: ConfigureMessage = { type: 'configure', config };
  return JSON.stringify(message);
}

/** `null` clears the name: the camera is then shown by its board id. */
export function renameMessage(deviceId: string, name: string | null): string {
  const message: RenameMessage = { type: 'rename', device_id: deviceId, name };
  return JSON.stringify(message);
}

export function forgetMessage(deviceId: string): string {
  const message: ForgetMessage = { type: 'forget', device_id: deviceId };
  return JSON.stringify(message);
}

/** A JPEG the node pushed as base64 (`preview`, `frame`), as an `<img src>`. */
export function jpegSrc(jpeg: string): string {
  return `data:image/jpeg;base64,${jpeg}`;
}
