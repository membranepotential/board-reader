/** What the control page says about the way the board lies in the camera's image. */

import type { BoardDirection } from '$lib/protocol/generated';

type WhiteSide = BoardDirection['white_side'];

const SIDE_COPY: Record<WhiteSide, string> = {
  bottom: 'at the bottom',
  left: 'on the left',
  top: 'at the top',
  right: 'on the right'
};

/** `White sits on the left`: the side as the operator sees it on the preview. */
export function whiteSideCopy(side: WhiteSide): string {
  return `White sits ${SIDE_COPY[side]}`;
}

export const CAMERA_MOVED_COPY =
  'The camera moved: the board has turned since its direction settled. Redraw the crop around ' +
  'the board and save it; a new crop, lens or resolution settles the direction anew.';
