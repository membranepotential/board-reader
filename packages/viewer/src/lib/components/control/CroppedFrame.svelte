<script lang="ts">
  /**
   * The latest frame the node received for a board: the camera's picture after the
   * crop, so what the recognizer reads. The control socket pushes it (`frame`) as
   * frames arrive, so nothing here polls.
   *
   * Sized like `CropEditor`'s picture: the box takes the frame's own aspect ratio
   * and the image fills it at `inset: 0`, so the box IS the picture, with no
   * letterbox inside it.
   */
  import type { FrameMessage } from '#lib/protocol/generated.ts';
  import { jpegSrc } from '#lib/control/messages.ts';

  interface Props {
    frame: FrameMessage | null;
    boardId: string;
  }

  const { frame, boardId }: Props = $props();
</script>

{#if frame !== null}
  <div class="picture" style:--ar={frame.width / frame.height}>
    <img
      class="received"
      src={jpegSrc(frame.jpeg)}
      alt="the latest frame the node received for board {boardId}"
    />
  </div>
{:else}
  <div class="placeholder">no frame received yet: start streaming to see what the node reads</div>
{/if}

<style>
  .picture {
    position: relative;
    aspect-ratio: var(--ar);
    width: min(100%, calc(55vh * var(--ar)));
    background: #0f1216;
    border: 1px solid #272b33;
    border-radius: 6px;
    overflow: hidden;
  }
  img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
  }
  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 16 / 9;
    border: 1px dashed #2f3540;
    border-radius: 6px;
    color: #7d8590;
    font-size: 13px;
    text-align: center;
    padding: 12px;
  }
</style>
