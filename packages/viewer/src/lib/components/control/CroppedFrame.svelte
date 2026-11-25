<script lang="ts">
  /**
   * The camera's picture after the crop: what the recognizer reads.
   *
   * While the camera streams, it is the latest frame the node received for the
   * board (`frame`, pushed by the control socket as frames arrive). While it does
   * not, no frame arrives, so the picture would go stale; the phone still sends its
   * small whole-frame previews, so the view crops the latest one to the config's
   * crop in CSS instead. That follows a new crop at once, including one Auto-frame
   * sets.
   *
   * Sized like `CropEditor`'s picture: the box takes the shown picture's own aspect
   * ratio and the image sits inside it, so the box IS the picture, with no letterbox.
   * For a cropped preview the box takes the crop's aspect ratio (in pixels), and the
   * whole preview inside it is scaled by 1/w and 1/h and shifted by -x and -y, so only
   * the crop's region shows.
   */
  import type { CropRect, FrameMessage, PreviewMessage } from '#lib/protocol/generated.ts';
  import { WHOLE_FRAME } from '#lib/control/crop.ts';
  import { jpegSrc } from '#lib/control/messages.ts';

  interface Props {
    /** The latest frame the node received, or `null` before the first. */
    frame: FrameMessage | null;
    /** The phone's latest whole-frame preview. */
    preview: PreviewMessage | null;
    /** The config's crop; `null` sends the whole frame. */
    crop: CropRect | null;
    streaming: boolean;
    /** The camera's name, for the pictures' alt text. */
    name: string;
  }

  const { frame, preview, crop, streaming, name }: Props = $props();

  type View =
    | { readonly kind: 'received'; readonly frame: FrameMessage }
    | { readonly kind: 'preview'; readonly preview: PreviewMessage; readonly crop: CropRect }
    | { readonly kind: 'none' };

  const view = $derived.by((): View => {
    if (streaming && frame !== null) return { kind: 'received', frame };
    if (preview !== null) return { kind: 'preview', preview, crop: crop ?? WHOLE_FRAME };
    if (frame !== null) return { kind: 'received', frame };
    return { kind: 'none' };
  });
</script>

{#if view.kind === 'received'}
  <div class="picture" style:--ar={view.frame.width / view.frame.height}>
    <img
      class="received"
      src={jpegSrc(view.frame.jpeg)}
      alt="the latest frame the node received from camera {name}"
    />
  </div>
{:else if view.kind === 'preview'}
  {@const { preview: shot, crop: c } = view}
  <div class="picture" data-testid="cropped-preview" style:--ar={(c.w * shot.width) / (c.h * shot.height)}>
    <img
      class="preview-crop"
      src={jpegSrc(shot.jpeg)}
      alt="the preview of camera {name}, cropped as its stream would be"
      style:width="{100 / c.w}%"
      style:height="{100 / c.h}%"
      style:left="{(-c.x / c.w) * 100}%"
      style:top="{(-c.y / c.h) * 100}%"
    />
  </div>
{:else}
  <div class="placeholder">no picture from this camera yet</div>
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
    display: block;
  }
  img.received {
    inset: 0;
    width: 100%;
    height: 100%;
  }
  img.preview-crop {
    max-width: none;
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
