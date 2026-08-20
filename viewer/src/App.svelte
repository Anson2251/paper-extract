<script>
  import UploadZone from './lib/UploadZone.svelte'
  import Paper from './lib/Paper.svelte'
  import { Button, Spinner } from 'flowbite-svelte'
  import { loadPaperFromZip, releasePaper } from './lib/parseZip.js'

  let loaded = null
  let loading = false
  let error = ''

  async function handleLoad(event) {
    const [file] = event.detail.files
    if (!file) return
    error = ''
    loading = true
    try {
      const buf = await file.arrayBuffer()
      loaded = await loadPaperFromZip(buf)
    } catch (e) {
      error = e?.message ?? String(e)
    } finally {
      loading = false
    }
  }

  function reset() {
    if (loaded) releasePaper(loaded)
    loaded = null
    error = ''
  }
</script>

<main class="min-h-screen bg-gray-100 text-gray-900">
  {#if loading}
    <div class="flex min-h-screen items-center justify-center gap-2 text-gray-500">
      <Spinner class="h-6 w-6" /> Loading paper…
    </div>
  {:else if loaded}
    <header class="sticky top-0 z-10 border-b border-gray-200 bg-white/90 backdrop-blur">
      <div class="mx-auto flex w-full max-w-3xl items-center justify-between px-4 py-3">
        <span class="font-semibold">Paper Viewer</span>
        <Button size="sm" color="light" on:click={reset}>Open another paper</Button>
      </div>
    </header>
    <Paper paper={loaded} />
  {:else}
    <UploadZone {error} on:load={handleLoad} />
  {/if}
</main>
