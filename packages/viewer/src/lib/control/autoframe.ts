/** What /control says about Auto-frame (`StateMessage.autoframe`), as an operator reads it. */

import type { StateMessage } from '#lib/protocol/generated.ts';

type AutoFrameStatus = StateMessage['autoframe'];

export const AUTOFRAME_COPY: { readonly [K in AutoFrameStatus]: string } = {
  idle: 'waiting for the camera',
  locating: 'looking for the board…',
  framed: 'on: the node set the crop',
  'not-visible': 'on, but the board is not visible in the picture',
  'no-board': 'no board found yet; trying again',
  manual: 'off: the crop was set by hand'
};
