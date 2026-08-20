<script>
  import { Badge, Card } from 'flowbite-svelte'
  import { onDestroy, onMount } from 'svelte'
  import Question from './Question.svelte'

  export let paper

  $: meta = paper?.paper?.meta
  $: supp = meta?.supplement
  $: perModule = paper?.paper?.completeness?.per_module ?? {}

  const filters = [
    { value: 'all', label: 'All questions' },
    { value: 'rw1', label: 'Reading & Writing · M1' },
    { value: 'rw2', label: 'Reading & Writing · M2' },
    { value: 'math1', label: 'Math · M1' },
    { value: 'math2', label: 'Math · M2' }
  ]
  let filter = 'all'
  let search = ''

  $: list = (paper?.questions ?? []).filter((q) => {
    const sectionMatch = {
      all: true,
      rw1: q.section === 'reading_writing' && q.module === 1,
      rw2: q.section === 'reading_writing' && q.module === 2,
      math1: q.section === 'math' && q.module === 1,
      math2: q.section === 'math' && q.module === 2
    }[filter]
    if (!sectionMatch) return false
    if (search) {
      const hay =
        `${q.question} ${q.background} ${(q.options || []).map((o) => o.label).join(' ')}`
          .toLowerCase()
      if (!hay.includes(search.toLowerCase())) return false
    }
    return true
  })

  const itemLabel = (key) =>
    (
      {
        'reading_writing:1': 'RW · M1',
        'reading_writing:2': 'RW · M2',
        'math:1': 'Math · M1',
        'math:2': 'Math · M2'
      }[key] ?? key
    )

  const groupLabel = (key) =>
    (
      {
        'reading_writing:1': 'Reading & Writing · Module 1',
        'reading_writing:2': 'Reading & Writing · Module 2',
        'math:1': 'Math · Module 1',
        'math:2': 'Math · Module 2'
      }[key] ?? key
    )

  // groups of questions by (section, module), in canonical order
  $: grouped = (() => {
    const order = []
    const map = new Map()
    for (const q of list) {
      const key = `${q.section}:${q.module}`
      if (!map.has(key)) {
        map.set(key, [])
        order.push(key)
      }
      map.get(key).push(q)
    }
    return order.map((key) => ({ key, questions: map.get(key) }))
  })()

  // Parse the ISO timestamp and render it in the user's system locale/timezone.
  const formatCurated = (iso) => {
    if (!iso) return ''
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return iso
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  $: title =
    `${meta?.year ?? '?'}-${meta?.month ?? '?'} ${meta?.area ?? ''}${meta?.code ? ` ${meta.code}` : ''} Paper`

  // --- floating navigation ------------------------------------------------
  const MODULE_ORDER = ['reading_writing:1', 'reading_writing:2', 'math:1', 'math:2']
  $: presentKeys = MODULE_ORDER.filter((k) =>
    (paper?.questions ?? []).some((q) => `${q.section}:${q.module}` === k)
  )

  let jumpModule = MODULE_ORDER[0]
  let active = null

  let currentId = null
  $: currentModule = currentId ? currentId.replace(/:\d+$/, '') : null
  $: if (currentModule) jumpModule = currentModule
  $: currentNumber = currentId ? Number(currentId.split(':').pop()) : null

  $: jumpNums = jumpModule
    ? (paper?.questions ?? [])
        .filter((q) => `${q.section}:${q.module}` === jumpModule)
        .map((q) => q.number)
        .sort((a, b) => a - b)
    : []

  function scrollToSection(key) {
    document
      .querySelector(`[data-section="${key}"]`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  function scrollToQuestion(moduleKey, number) {
    // Scroll to the invisible anchor placed before the card, so the landing
    // position is stable regardless of how tall the card's content is.
    document
      .querySelector(`[data-anchor="${moduleKey}:${number}"]`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  let io
  let headerObserver
  let qObserver
  let headerEl
  let navVisible = false

  onMount(() => {
    const headers = [...document.querySelectorAll('[data-section]')]
    io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) active = e.target.getAttribute('data-section')
        }
      },
      { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
    )
    headers.forEach((h) => io.observe(h))

    const qitems = [...document.querySelectorAll('[data-qid]')]
    qObserver = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) currentId = e.target.getAttribute('data-qid')
        }
      },
      { rootMargin: '-35% 0px -55% 0px', threshold: 0 }
    )
    qitems.forEach((el) => qObserver.observe(el))

    if (headerEl) {
      headerObserver = new IntersectionObserver(([entry]) => {
        navVisible = !entry.isIntersecting
      }, { rootMargin: '0px' })
      headerObserver.observe(headerEl)
    }
  })
  onDestroy(() => {
    io?.disconnect()
    qObserver?.disconnect()
    headerObserver?.disconnect()
  })
</script>

<div class="mx-auto w-full max-w-3xl px-8 py-6">
  <!-- left: navigate to sections (shows once the header scrolls out of view) -->
  <aside
    class="fixed left-2 top-16 z-20 hidden lg:block transition-all duration-300"
    class:pointer-events-none={!navVisible}
    class:opacity-0={!navVisible}
    class:-translate-x-3={!navVisible}
  >
    <nav class="w-44 max-h-[80vh] overflow-y-auto rounded-xl border border-gray-200 bg-white p-3 shadow-lg">
      <div class="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-gray-400">Sections</div>
      <ul class="space-y-1">
        {#each presentKeys as key}
          <li>
            <button
              type="button"
              class:bg-blue-50={active === key}
              class:font-semibold={active === key}
              class:text-blue-600={active === key}
              on:click={() => scrollToSection(key)}
              class="w-full rounded-lg px-2 py-1.5 text-left text-sm text-gray-600 transition hover:bg-gray-100"
            >
              {itemLabel(key)}
            </button>
          </li>
        {/each}
      </ul>
    </nav>
  </aside>

  <!-- right: navigate to a specific question -->
  <aside
    class="fixed right-2 top-16 z-20 hidden lg:block transition-all duration-300"
    class:pointer-events-none={!navVisible}
    class:opacity-0={!navVisible}
    class:translate-x-3={!navVisible}
  >
    <div class="w-44 max-h-[80vh] overflow-y-auto rounded-xl border border-gray-200 bg-white p-3 shadow-lg">
      <div class="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-gray-400">Questions</div>
      <select
        bind:value={jumpModule}
        class="mb-2 w-full rounded-lg border border-gray-300 bg-white px-2 py-1.5 text-sm outline-none focus:border-blue-400"
      >
        {#each presentKeys as key}<option value={key}>{itemLabel(key)}</option>{/each}
      </select>
      <div class="grid grid-cols-4 gap-1">
        {#each jumpNums as n}
          {@const isCurrent = currentModule === jumpModule && currentNumber === n}
          <button
            type="button"
            on:click={() => scrollToQuestion(jumpModule, n)}
            class:bg-blue-600={isCurrent}
            class:text-white={isCurrent}
            class:border-blue-300={isCurrent}
            class:hover:border-blue-300={!isCurrent}
            class:hover:text-blue-600={!isCurrent}
            class="rounded border border-gray-200 py-1 text-xs text-gray-600 transition"
          >
            {n}
          </button>
        {/each}
      </div>
    </div>
  </aside>

  <div bind:this={headerEl} data-meta-header>
    <Card class="mb-6 w-full! max-w-none">
    <div class="flex flex-wrap items-center gap-2">
      <h2 class="mr-2 text-xl font-bold">{title}</h2>
      {#if meta?.organizer}<Badge color="blue">Curated by {meta.organizer}</Badge>{/if}
      {#if meta?.curated_at}<Badge color="light">{formatCurated(meta.curated_at)}</Badge>{/if}
    </div>

    {#if supp}
      <div class="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
        <div class="rounded-lg bg-gray-50 p-3">
          <div class="text-xs text-gray-500">Expected</div>
          <div class="text-lg font-semibold">{supp.expected}</div>
        </div>
        <div class="rounded-lg bg-gray-50 p-3">
          <div class="text-xs text-gray-500">Included</div>
          <div class="text-lg font-semibold">{paper.questions.length}</div>
        </div>
        <div class="rounded-lg bg-gray-50 p-3">
          <div class="text-xs text-gray-500">Recovered</div>
          <div class="text-lg font-semibold text-blue-600">{supp.recovered}</div>
        </div>
        <div class="rounded-lg bg-gray-50 p-3">
          <div class="text-xs text-gray-500">Missing</div>
          <div class="text-lg font-semibold text-amber-600">{supp.missing}</div>
        </div>
      </div>
      <div class="mt-3 flex flex-wrap gap-1.5">
        {#each Object.entries(perModule) as [key, c]}
          <Badge color="gray">
            {itemLabel(key)}: {c.kept + (c.recovered || 0)}/{c.expected}
          </Badge>
        {/each}
      </div>
    {/if}

    {#if paper.missingRefs?.length}
      <p class="mt-3 text-xs text-red-500">
        ⚠ {paper.missingRefs.length} referenced image(s) missing: {paper.missingRefs.join(', ')}
      </p>
    {/if}
  </Card>
</div>

  <div class="mb-4 flex flex-wrap items-center gap-3">
    <select
      bind:value={filter}
      class="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm outline-none focus:border-blue-400"
    >
      {#each filters as f}<option value={f.value}>{f.label}</option>{/each}
    </select>
    <input
      bind:value={search}
      placeholder="Search questions / options…"
      class="min-w-0 flex-1 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm outline-none focus:border-blue-400"
    />
    <span class="text-sm text-gray-500">{list.length} questions</span>
  </div>

  {#if grouped.length}
    {#each grouped as g}
      <div data-section={g.key} class="mt-8 flex scroll-mt-24 items-center gap-3 py-6 first:mt-0">
        <h3 class="shrink-0 text-sm font-semibold uppercase tracking-wide text-gray-600">
          {groupLabel(g.key)}
        </h3>
        <div class="h-px flex-1 bg-gray-300"></div>
        <span class="shrink-0 text-xs text-gray-400">{g.questions.length} questions</span>
      </div>
      {#each g.questions as q, qi (qi)}
        <div
          class="h-0 scroll-mt-24"
          data-anchor={`${q.section}:${q.module}:${q.number}`}
          aria-hidden="true"
        ></div>
        <div data-qid={`${q.section}:${q.module}:${q.number}`} class="scroll-mt-24">
          <Question {q} refToUrl={paper.refToUrl} />
        </div>
      {/each}
    {/each}
  {:else}
    <div class="rounded-lg bg-white p-8 text-center text-gray-400">No matching questions.</div>
  {/if}
</div>
