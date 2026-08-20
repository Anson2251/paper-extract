import JSZip from 'jszip'

/** Thrown when the uploaded file is not a usable paper bundle. */
export class PaperLoadError extends Error {}

/**
 * Read a paper bundle from a zip. The pipeline produces one directory per paper
 * containing `paper.json` and an `imgs/` folder, but we accept any zip that
 * contains a `paper.json` anywhere, and match `imgs/<file>` references by name.
 *
 * Returns `{ paper, questions, refToUrl, missingRefs }`:
 *  - `paper`: the parsed paper.json object
 *  - `questions`: paper.questions (array)
 *  - `refToUrl`: Map of `imgs/<file>` -> object URL (blob)
 *  - `missingRefs`: refs referenced by questions but absent from the zip
 */
export async function loadPaperFromZip(arrayBuffer) {
  let zip
  try {
    zip = await JSZip.loadAsync(arrayBuffer)
  } catch {
    throw new PaperLoadError('Not a valid zip file')
  }

  const entries = Object.values(zip.files).filter((f) => !f.dir)
  const jsonEntries = entries
    .filter((e) => /(^|\/)paper\.json$/.test(e.name))
    .sort((a, b) => a.name.length - b.name.length)

  if (!jsonEntries.length) throw new PaperLoadError('paper.json not found in zip')

  let paper
  try {
    paper = JSON.parse(await jsonEntries[0].async('string'))
  } catch {
    throw new PaperLoadError('Could not parse paper.json')
  }

  const questions = Array.isArray(paper.questions) ? paper.questions : []

  // Build the image ref -> object URL map from the zip's imgs/ entries.
  const refToUrl = new Map()
  for (const entry of entries) {
    const m = entry.name.match(/imgs\/([^/]+)$/)
    if (!m) continue
    const blob = await entry.async('blob')
    refToUrl.set(`imgs/${m[1]}`, URL.createObjectURL(blob))
  }

  // Which refs do the questions actually reference?
  const used = new Set()
  for (const q of questions) {
    for (const field of ['background', 'question', 'explanations']) {
      for (const m of String(q?.[field] ?? '').matchAll(/\[image:\s*([^\]]+)\]/g)) {
        used.add(m[1].trim())
      }
    }
    for (const o of q?.options ?? []) {
      for (const m of String(o?.label ?? '').matchAll(/\[image:\s*([^\]]+)\]/g)) {
        used.add(m[1].trim())
      }
    }
  }

  const missingRefs = [...used].filter((r) => !refToUrl.has(r))

  return { paper, questions, refToUrl, missingRefs }
}

/** Utility to free all object URLs created while loading a paper. */
export function releasePaper({ refToUrl }) {
  if (!refToUrl) return
  refToUrl.forEach((url) => URL.revokeObjectURL(url))
}
