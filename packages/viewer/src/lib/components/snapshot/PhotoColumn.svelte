<script lang="ts">
  /** The photo as posted: it never leaves the screen — it is what every stage talks about. */
  interface Props {
    dataUrl: string;
  }

  const { dataUrl }: Props = $props();
</script>

<div class="photo-col">
  <div class="col-label">Photo as posted</div>
  <div class="photo-frame">
    <img src={dataUrl} alt="the board as posted to the reader" />
  </div>
</div>

<style>
  .photo-col {
    flex: 1 1 340px;
    max-width: 440px;
    min-width: 0;
  }

  .col-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--dim);
    margin-bottom: 6px;
  }

  /* A square frame, so the photo block matches the board block exactly whatever the
     photo's aspect ratio. The frame's size is definite (width + aspect-ratio), so the
     photo can fit itself with object-fit and needs no measuring. */
  .photo-frame {
    width: 100%;
    aspect-ratio: 1;
    background: #0f1216;
    border: 1px solid var(--line);
    border-radius: 6px;
    overflow: hidden;
  }

  .photo-frame img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
</style>
