/**
 * The one success-or-reason type the library uses at its boundaries.
 *
 * Everything that can fail for a reason the *operator* needs to read — a socket
 * frame that is not a position event, a node that refused a photo — returns
 * one of these rather than throwing. Two reasons, in order:
 *
 *  - the pages surface the node's own wording verbatim (`{"error": ...}` bodies
 *    say why the node refused; a generic "request
 *    failed" would throw that away), and
 *  - a live viewer must never lose a board to an exception raised inside a render
 *    path. Exceptions stay for programming errors — an index off the board, a grid
 *    that is not 8 long — which are bugs here, not conditions out on the wire.
 */
export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

export function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}
