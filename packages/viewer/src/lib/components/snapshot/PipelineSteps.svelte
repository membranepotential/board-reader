<script lang="ts">
  /** The three pipeline stages in order, each an expanding section with its own body. */
  import CaptureStage from '#lib/components/snapshot/CaptureStage.svelte';
  import OutputStage from '#lib/components/snapshot/OutputStage.svelte';
  import PipelineStep from '#lib/components/snapshot/PipelineStep.svelte';
  import ReadStage from '#lib/components/snapshot/ReadStage.svelte';
  import type { SnapshotSession } from '#lib/components/snapshot/session.svelte.ts';
  import { STAGES } from '#lib/components/snapshot/pipeline.ts';

  interface Props {
    session: SnapshotSession;
  }

  const { session }: Props = $props();
</script>

<div class="steps">
  {#each STAGES as stage (stage.id)}
    <PipelineStep
      {stage}
      status={session.pipeline[stage.id]}
      bind:open={session.stageOpen[stage.id]}
    >
      {#if stage.id === 'capture'}
        <CaptureStage image={session.image} facts={session.facts} />
      {:else if stage.id === 'read'}
        <ReadStage board={session.board} estimateMs={session.estimateMs} />
      {:else}
        <OutputStage board={session.board} bind:stm={session.stm} />
      {/if}
    </PipelineStep>
  {/each}
</div>

<style>
  .steps {
    margin-top: 18px;
  }
</style>
