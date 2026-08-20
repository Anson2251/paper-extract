// Tiny content parser for the pipeline's question text:
//  - `[image: imgs/<file>.jpg]`  -> inline image token
//  - `[equation]` markers        -> stripped
//  - `\$`  (escaped dollar)      -> treated as a literal currency dollar
//  - markdown pipe tables        -> extracted to real <table> data

const IMAGE_RE = /\[image:\s*([^\]]+)\]/g

// Placeholder for a literal dollar sign. The pipeline escapes currency as
// `\$` so it isn't parsed as a math delimiter; we swap it out before KaTeX runs
// and restore it afterwards, otherwise `\$15` would render with a stray
// backslash or get swallowed by math auto-rendering.
export const ESCAPED_DOLLAR = '\uE000'

/**
 * Normalise raw question text before rendering: drop leftover `[equation]`
 * markers and swap escaped currency dollars (`\$`) for a placeholder so KaTeX
 * won't read them as math delimiters (restore via `restoreDollars`).
 */
export function cleanContent(text) {
  return String(text ?? '')
    .replace(/\[equation\]/g, '')
    .replace(/\\\$/g, ESCAPED_DOLLAR)
}

/**
 * Turn the escaped-dollar placeholder back into a literal `$` in a text node.
 * Run this after KaTeX has rendered, on the same container.
 */
export function restoreDollars(container) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let node
  const nodes = []
  while ((node = walker.nextNode())) nodes.push(node)
  for (const n of nodes) {
    if (n.nodeValue.includes(ESCAPED_DOLLAR)) {
      n.nodeValue = n.nodeValue.split(ESCAPED_DOLLAR).join('$')
    }
  }
}

/** Split a string into text tokens and inline image tokens. */
export function splitInline(text) {
  const out = []
  let last = 0
  for (const m of text.matchAll(IMAGE_RE)) {
    if (m.index > last) out.push({ type: 'text', text: text.slice(last, m.index) })
    out.push({ type: 'img', ref: m[1].trim() })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ type: 'text', text: text.slice(last) })
  return out
}

const SEP_RE = /^\s*\|[\s:|+\-]+\|\s*$/ // separator row, e.g. |---|---|

/**
 * Pull markdown table blocks out of a string.
 * Returns `{ tables, text }` where `tables` is `[{ header?: [], rows: [][] }]`
 * and `text` is the original string with table lines removed.
 */
export function extractTables(text) {
  const tables = []
  const kept = []
  const lines = String(text ?? '').split('\n')
  const rowsOf = (line) => line.split('|').slice(1, -1).map((s) => s.trim())

  let i = 0
  while (i < lines.length) {
    const startsTable =
      /^\s*\|/.test(lines[i]) &&
      i + 1 < lines.length &&
      SEP_RE.test(lines[i + 1])
    if (!startsTable) {
      kept.push(lines[i])
      i += 1
      continue
    }
    const block = [lines[i]]
    let j = i + 1
    while (j < lines.length && /^\s*\|/.test(lines[j])) {
      block.push(lines[j])
      j += 1
    }
    const header = rowsOf(block[0])
    const rows = []
    for (const line of block.slice(1)) {
      if (SEP_RE.test(line)) continue // skip separator / repeated header rows
      const r = rowsOf(line)
      if (r.length) rows.push(r)
    }
    tables.push({ header, rows })
    i = j
  }
  return { tables, text: kept.join('\n') }
}
