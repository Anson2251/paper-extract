<script>
  import { createEventDispatcher } from 'svelte'
  import { UploadSolid as UploadIcon } from 'flowbite-svelte-icons'

  const dispatch = createEventDispatcher()

  export let error = ''

  let dragging = false

  function emit(files) {
    if (files && files.length) dispatch('load', { files: [...files] })
  }
  function onSelect(e) {
    emit(e.currentTarget.files)
  }
  function onDrop(e) {
    e.preventDefault()
    dragging = false
    emit(e.dataTransfer.files)
  }
  function onDragOver(e) {
    e.preventDefault()
    dragging = true
  }
  function onDragLeave() {
    dragging = false
  }
</script>

<div class="mx-auto flex min-h-screen w-full max-w-2xl flex-col justify-center px-4">
  <header class="mb-6 text-center">
    <h1 class="text-3xl font-bold text-gray-900">Paper Viewer</h1>
    <p class="mt-2 text-gray-500">Select a paper bundle zip (containing paper.json and imgs/) to start browsing</p>
  </header>

  <label
    for="paper-zip"
    on:drop|preventDefault={onDrop}
    on:dragover|preventDefault={onDragOver}
    on:dragleave={onDragLeave}
    class:border-blue-500={dragging}
    class="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-gray-300 bg-white px-6 py-16 text-center transition hover:border-blue-400"
  >
    <UploadIcon class="h-12 w-12 text-blue-500" />
    <span class="text-lg font-medium text-gray-700">
      {dragging ? 'Release to select' : 'Drag a zip here, or click to choose a file'}
    </span>
    <span class="text-sm text-gray-400">Drag & drop or click to select, one .zip at a time</span>
  </label>

  <input
    id="paper-zip"
    type="file"
    accept=".zip,application/zip"
    class="hidden"
    on:change={onSelect}
  />

  {#if error}
    <p class="mt-4 rounded-lg bg-red-50 px-4 py-3 text-center text-sm text-red-600">
      {error}
    </p>
  {/if}
</div>
