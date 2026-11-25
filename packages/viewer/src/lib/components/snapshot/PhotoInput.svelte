<script lang="ts">
  /**
   * The big capture control: a `<label>` styled as a button wrapping a visually-hidden
   * file input, so the native chooser still fires but the look is ours.
   */
  interface Props {
    /** Once a photo is in hand the call to action shrinks. */
    hasPhoto: boolean;
    onFile: (file: File) => void;
  }

  const { hasPhoto, onFile }: Props = $props();

  function onChange(event: Event & { currentTarget: HTMLInputElement }): void {
    const file = event.currentTarget.files?.[0];
    if (file !== undefined) onFile(file);
  }
</script>

<label class="capture-btn" class:compact={hasPhoto} for="file">
  {hasPhoto ? 'New photo' : 'Take / choose photo'}
</label>
<!-- Deliberately NO capture attribute: with it, iOS Safari skips the chooser and
     forces the camera, making the photo library unreachable. Bare, both iOS and
     Android present the native sheet offering camera AND camera roll. -->
<input type="file" id="file" accept="image/*" class="visually-hidden" onchange={onChange} />
<p class="hint">
  Photograph the board — the reader finds it in the whole photo, so there is nothing to
  frame. Every stage below records what it saw.
</p>

<style>
  .hint {
    color: var(--muted);
    font-size: 13px;
    margin: 6px 0;
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
    border: 0;
  }

  .capture-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent);
    color: #fff;
    font-weight: 700;
    font-size: 17px;
    min-height: 60px;
    border-radius: 12px;
    cursor: pointer;
    touch-action: manipulation;
  }

  /* Shrinks once a photo is in hand — the big CTA has done its job. */
  .capture-btn.compact {
    min-height: 44px;
    font-size: 14px;
    border-radius: 9px;
  }
</style>
