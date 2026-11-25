<script lang="ts">
  /**
   * A camera's name, edited in place on its card. Enter or Save sends it; Escape puts
   * the node's name back. An empty name, or the board id itself, clears the name.
   *
   * The field shows what the operator types until the node's name CHANGES (the
   * rename went through, or another page renamed it). `known` is derived from the
   * name string alone, so the list the node pushes about once a second with the same
   * name does not touch it, and `value`, a writable `$derived` over `known`, keeps
   * the typing; the first /control reset a half-typed name on that push. A refused
   * name stays in the field to be corrected; the refusal shows on the page.
   */
  interface Props {
    name: string | null;
    boardId: string;
    /** Send the rename; `false` when the node is not connected. */
    onrename: (name: string | null) => boolean;
  }

  const { name, boardId, onrename }: Props = $props();

  const known = $derived(name ?? '');
  let value = $derived(known);

  /** What the typed value names: `null` clears the name. */
  const wanted = $derived.by(() => {
    const typed = value.trim();
    return typed === '' || typed === boardId ? null : typed;
  });
  const dirty = $derived(wanted !== name);

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    if (dirty) onrename(wanted);
  }

  function onkeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') value = known;
  }
</script>

<form class="name" onsubmit={submit}>
  <label>
    name
    <input bind:value placeholder={boardId} maxlength="32" autocomplete="off" {onkeydown} />
  </label>
  <button type="submit" disabled={!dirty}>Save name</button>
</form>

<style>
  form {
    display: flex;
    align-items: flex-end;
    gap: 8px;
  }
  label {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
    color: #9aa3ad;
  }
  input {
    font: inherit;
    font-size: 13px;
    background: #232833;
    color: #e6e6e6;
    border: 1px solid #2f3540;
    border-radius: 6px;
    padding: 5px 8px;
  }
  button {
    font: inherit;
    font-size: 13px;
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid #2f3540;
    background: #232833;
    color: #e6e6e6;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
