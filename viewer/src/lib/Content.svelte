<script>
  import { onMount, afterUpdate, tick } from 'svelte'
  import renderMathInElement from 'katex/contrib/auto-render'
  import { marked } from 'marked'
  import { splitInline, extractTables, cleanContent, restoreDollars } from './blocks.js'

  export let text = ''
  export let refToUrl = new Map()

  marked.setOptions({ gfm: true, breaks: true })

  $: cleaned = cleanContent(text)
  $: prepped = extractTables(cleaned)
  $: rawChunks = splitInline(prepped.text)
  $: chunks = rawChunks.map((c) =>
    c.type === 'img' ? c : { ...c, html: marked.parse(c.text) }
  )

  let root

  async function renderMath() {
    await tick()
    if (!root) return
    renderMathInElement(root, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false
    })
    restoreDollars(root)
  }

  onMount(renderMath)
  afterUpdate(renderMath)
  $: if (root && cleaned) {
    // reactive trigger when text changes
    renderMath()
  }
</script>

<div bind:this={root} class="break-words">
  {#if prepped.tables.length}
    <div class="my-3 space-y-3">
      {#each prepped.tables as table, ti (ti)}
        <div class="overflow-x-auto rounded-lg border border-gray-200">
          <table class="w-full text-sm">
            {#if table.header}
              <thead>
                <tr class="bg-gray-100 text-left text-gray-700">
              {#each table.header as h}
                  <th class="px-3 py-2 font-semibold">{@html marked.parseInline(h)}</th>
                {/each}
                </tr>
              </thead>
            {/if}
            <tbody>
              {#each table.rows as row, ri (ri)}
                <tr class="border-t border-gray-200 odd:bg-white even:bg-gray-50">
                  {#each row as cell}
                    <td class="px-3 py-2">{@html marked.parseInline(cell)}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/each}
    </div>
  {/if}

  <div class="prose prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2">
    {#each chunks as c, i (i)}
      {#if c.type === 'img'}
        {#if refToUrl.get(c.ref)}
          <img
            src={refToUrl.get(c.ref)}
            alt={c.ref}
            class="my-2 max-w-full rounded-lg border border-gray-200"
            loading="lazy"
          />
        {:else}
          <span class="text-xs font-medium text-red-500">[Missing image: {c.ref}]</span>
        {/if}
      {:else}
        <div class="markdown-chunk">{@html c.html}</div>
      {/if}
    {/each}
  </div>
</div>

<style>
  :global(.markdown-chunk p) {
    margin: 0.5em 0;
  }
  :global(.markdown-chunk ul) {
    list-style: disc;
    padding-left: 1.5em;
    margin: 0.5em 0;
  }
  :global(.markdown-chunk ol) {
    list-style: decimal;
    padding-left: 1.5em;
    margin: 0.5em 0;
  }
  :global(.markdown-chunk blockquote) {
    border-left: 2px solid #e5e7eb;
    padding-left: 0.75em;
    color: #6b7280;
    margin: 0.5em 0;
  }
  :global(.markdown-chunk code) {
    background: #f3f4f6;
    padding: 0.1em 0.3em;
    border-radius: 0.25em;
    font-size: 0.9em;
  }
  :global(.markdown-chunk pre) {
    background: #f3f4f6;
    padding: 0.75em;
    border-radius: 0.5em;
    overflow-x: auto;
    margin: 0.5em 0;
  }
  :global(.markdown-chunk pre code) {
    background: none;
    padding: 0;
  }
</style>
