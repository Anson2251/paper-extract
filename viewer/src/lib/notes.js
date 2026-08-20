/**
 * Parse the free-form `note` strings the pipeline attaches to questions into a
 * structured shape so they can be rendered meaningfully instead of raw text.
 *
 * Known shapes:
 *   "answer corrected {'from': 'B', 'to': 'A'}; explanation regenerated"
 *   "answer differs from official key (key: C)"
 *   "no validated correct answer"
 *   "explanation regenerated"
 *
 * Returns `{ kind, from, to, keyAnswer, explanationRegenerated, text }`:
 *   kind: 'corrected' | 'disputed' | 'unresolved' | 'rewritten' | 'plain'
 */
export function parseNote(note) {
  const s = String(note ?? '').trim()
  const out = { kind: 'plain', text: s }

  if (!s) return out

  const corrected = /'from':\s*'([^']*)'\s*,\s*'to':\s*'([^']*)'/.exec(s)
  if (corrected) {
    out.kind = 'corrected'
    out.from = corrected[1]
    out.to = corrected[2]
    out.explanationRegenerated = /explanation regenerated/i.test(s)
    return out
  }

  const disputed = /answer differs from official key\s*\(\s*key:\s*([^)]+)\)/i.exec(s)
  if (disputed) {
    out.kind = 'disputed'
    out.keyAnswer = disputed[1].trim()
    return out
  }

  if (/no validated correct answer/i.test(s)) {
    out.kind = 'unresolved'
    return out
  }

  if (/explanation regenerated/i.test(s)) {
    out.kind = 'rewritten'
    out.explanationRegenerated = true
    return out
  }

  return out
}
