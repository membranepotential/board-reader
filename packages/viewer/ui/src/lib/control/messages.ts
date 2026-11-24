/**
 * The camera control channel's upward messages (`board_reader_core.control`).
 *
 * Two sockets per board, relayed same-origin by the viewer under `/ws`:
 *
 *  - the phone's, `cameraSocket(id)`: it sends `report` (text) and, only while the
 *    node asks, preview JPEGs (binary); it hears a `PhoneDown`.
 *  - a control page's, `controlSocket(id)`: it sends `configure`; it hears a
 *    `ControlDown`.
 *
 * Every text message is a JSON object whose `type` names it. The message types and
 * the validators for what comes down are generated from the Python models
 * (`$lib/protocol/generated`) and read with `parseWire`.
 */

import type {
  CameraConfig,
  CameraReport,
  ConfigureMessage,
  ReportMessage
} from '$lib/protocol/generated';

export function reportMessage(report: CameraReport): string {
  const message: ReportMessage = { type: 'report', report };
  return JSON.stringify(message);
}

export function configureMessage(config: CameraConfig): string {
  const message: ConfigureMessage = { type: 'configure', config };
  return JSON.stringify(message);
}

/** A JPEG the node pushed as base64 (`preview`, `frame`), as an `<img src>`. */
export function jpegSrc(jpeg: string): string {
  return `data:image/jpeg;base64,${jpeg}`;
}
