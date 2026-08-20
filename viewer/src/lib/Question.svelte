<script>
  import { Badge, Card } from 'flowbite-svelte'
  import Content from './Content.svelte'
  import { parseNote } from './notes.js'

  export let q
  export let refToUrl = new Map()

  $: noteDetail = q?.note ? parseNote(q.note) : null

  const sectionLabels = { reading_writing: 'Reading & Writing', math: 'Math' }

  $: srcBadge =
    {
      key: { label: 'Official key', color: 'green' },
      key_x: { label: 'Official key missing (x)', color: 'purple' },
      key_disputed: { label: 'Differs from official key', color: 'amber' },
      model_solved: { label: 'Model answer', color: 'blue' }
    }[q?.answer_source] ?? { label: q?.answer_source ?? 'Unknown', color: 'gray' }

  const statusBadge = {
    disputed: { label: 'Disputed answer', color: 'red' },
    unresolved: { label: 'Unresolved', color: 'amber' },
    rewritten: { label: 'Explanation rewritten', color: 'indigo' }
  }
</script>

{#if q}
  <Card class="mb-4 w-full! max-w-none">
    <div class="mb-3 flex flex-wrap items-center gap-2">
      <Badge color="gray">{sectionLabels[q.section] ?? q.section} · M{q.module}</Badge>
      <Badge color="gray">Question {q.number}</Badge>
      {#if q.page != null}<Badge color="light">Page {q.page}</Badge>{/if}
      <Badge color={srcBadge.color}>{srcBadge.label}</Badge>
      {#if statusBadge[q.status]}
        <Badge color={statusBadge[q.status].color}>{statusBadge[q.status].label}</Badge>
      {/if}
    </div>

    {#if q.background}
      <div class="mb-2 text-gray-600">
        <Content text={q.background} {refToUrl} />
      </div>
    {/if}

    <p class="mb-2 font-medium">
      <Content text={q.question} {refToUrl} />
    </p>

    {#if q.options && q.options.length}
      <ol class="space-y-1">
        {#each q.options as opt, i (i)}
          {@const isAnswer = opt.code === q.answer}
          {@const isKey = q.key_answer != null && opt.code === q.key_answer}
          <li
            class="flex items-start gap-2 rounded-lg border px-3 py-2 text-sm
            {isAnswer ? 'border-green-300 bg-green-50 text-green-900' : 'border-gray-200 bg-white'}"
          >
            <span class="font-semibold">{opt.code}.</span>
            <span class="flex-1"><Content text={opt.label} {refToUrl} /></span>
            {#if isAnswer}<Badge color="green">Answer</Badge>{/if}
            {#if isAnswer && !isKey && q.key_answer != null}
              <Badge color="amber">Official key = {q.key_answer}</Badge>
            {/if}
          </li>
        {/each}
      </ol>
    {:else}
      <div class="mb-2 rounded-lg bg-gray-100 px-3 py-2">
        <span class="font-semibold text-gray-500">Answer (free response):</span>
        <span class="font-medium">{q.answer}</span>
      </div>
    {/if}

    {#if q.explanations}
      <details class="mt-3">
        <summary class="cursor-pointer select-none text-sm font-medium text-blue-600">
          Explanation
        </summary>
        <div class="mt-2 rounded-lg bg-gray-50 p-3 text-sm text-gray-700">
          <Content text={q.explanations} {refToUrl} />
        </div>
      </details>
    {/if}

    {#if noteDetail}
      <div class="mt-3 flex flex-wrap items-center gap-2 border-t border-gray-200 pt-2 text-xs">
        {#if noteDetail.kind === 'corrected'}
          <Badge color="red">Answer changed from {noteDetail.from} to {noteDetail.to}</Badge>
          {#if noteDetail.explanationRegenerated}
            <Badge color="light">Explanation rewritten</Badge>
          {/if}
        {:else if noteDetail.kind === 'disputed'}
          <Badge color="amber">Differs from official key (official key: {noteDetail.keyAnswer})</Badge>
        {:else if noteDetail.kind === 'unresolved'}
          <Badge color="amber">No verified correct answer</Badge>
        {:else if noteDetail.kind === 'rewritten'}
          <Badge color="indigo">Explanation rewritten</Badge>
        {:else}
          <span class="text-gray-400">Note: {noteDetail.text}</span>
        {/if}
      </div>
    {/if}
  </Card>
{/if}
