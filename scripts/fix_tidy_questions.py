#!/usr/bin/env python3
"""Fix problems surfaced by validate_tidy_questions.py in tidy output.

validate_tidy_questions.py produces a "validated" report (see
scripts/validate_tidy_questions.py; by default
<papers/jsons/<year>/tidy/validated/<paper>.validated.json) that records, for
every question, whether the claimed answer is correct and whether the
explanation is reasonable. This script applies those verdicts back onto the
tidy questions and writes a corrected tidy JSON in the same schema.

Fixes applied per question when the validate report flags a problem:

  1. Answer — if the claimed answer is wrong and the validated "correct_answer"
     is known, replace "answer" with it (matched against option codes for
     multiple choice; set directly for free response).
  2. Explanation — regenerate a concise, correct explanation with the LLM (via
     litellm) whenever the answer is corrected (the old explanation argues for
     the old answer and would contradict the fix) or the explanation was judged
     unreasonable (or partial/poor) — in both cases only when a validated
     correct answer exists. Skipped in --answers-only mode.

Numbers are left as they came out of tidy: cross-page merges are handled
during tidy via the carried previous question, so renumbering is only needed
when the model produces duplicate or out-of-order numbers (opt-in via
--renumber).

The corrected questions preserve the tidy schema exactly:

    {
      "background": ..., "question": ..., "options": [...],
      "answer": ...,     "explanations": ...,
      "section": "reading_writing" | "math", "module": 1 | 2, "number": int
    }

Two files are written per input tidy file:
    <out>/<stem>.fixed.json      corrected tidy questions (drop-in for tidy)
    <out>/<stem>.changes.json    per-question fix report + summary

Usage:

    python scripts/fix_tidy_questions.py \
        --tidy papers/jsons/2024/tidy/demo-paper.json \
        --model gpt-5.6-luna

You can also point --tidy directly at an already-validated report (e.g.
.../tidy/validated/demo-paper.validated.json): it carries both the
question fields and the verdicts, so it is self-contained and no --validated
file is needed. Output stems strip any trailing .validated / .verified / .fixed
suffix, so re-running on already-processed files never double-appends.

Options:

    --tidy FILE          a tidy JSON, or an already-validated report JSON
                         (repeatable).
    --validated FILE     the corresponding validated report (repeatable). If
                         omitted, derived from each --tidy file as
                         <paper dir>/validated/<stem>.validated.json (a sibling
                         of the tidy folder) unless the --tidy file is itself an
                         already-validated report.
    --answers-only       only correct answers and renumber questions; do NOT
                         call the LLM to rewrite explanations. Offline/deterministic.
    --batch N            explanations to rewrite per LLM call (default: 1).
    --model NAME         litellm model string (default: $LITELLM_MODEL or gpt-5.6-luna).
    --output-dir DIR     where fixed files go (default: <paper dir>/fixed, a
                         sibling of the tidy folder).
    --max-retries N      LLM retries per call when JSON parsing fails (default: 2).
    --timeout SECONDS    per-call timeout in seconds (default: 300).
    --json-mode          request provider-native JSON mode (OpenAI-compatible
                         providers).
    --batch-size N       LLM calls per OpenAI Batch API job (default: 1 = one
                         synchronous call per group). With N > 1 and a native
                         OpenAI model, explanation requests go through OpenAI's
                         Batch API (50% cheaper, processed within 24h).
                         Ignored for non-OpenAI providers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import openai_batch
import paper_images

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_FLASH_MODEL = "gpt-5.6-luna"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOTS = [PROJECT_ROOT / "papers" / "jsons", PROJECT_ROOT / "papers" / "mds"]

EXPLAIN_SYSTEM_PROMPT = """\
You are an expert exam tutor. Given a question and its validated correct answer, \
write a concise, accurate explanation (1-3 sentences) of WHY that answer is \
correct, using only the question's own content. Do not question the validated \
answer. Output ONLY JSON with a single key "explanation": no markdown fences, \
no extra prose."""


# ------------------------------------------------------------- parsing helpers

def _scan_json(raw: str) -> str | None:
    """Return the first balanced top-level JSON value (array/object) in raw."""
    for start in range(len(raw)):
        if raw[start] not in "[{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
    return None


def parse_json_response(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    candidate = _scan_json(raw)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def base_stem(stem: str) -> str:
    """Strip a trailing '.validated' / '.verified' / '.fixed' suffix from a file
    stem so outputs never double-append (e.g. when the input is itself an
    already-validated or already-fixed file)."""
    return re.sub(r"\.(?:validated|verified|fixed)$", "", stem)


def _first_questions(data):
    """Return the question list of a report dict or a plain list, else None."""
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return data["questions"]
    if isinstance(data, list):
        return data
    return None


def is_validated_report(data) -> bool:
    """True if the loaded JSON is an already-validated report, i.e. its questions
    carry the verdict fields, so it is self-contained and needs no separate
    validated file."""
    qs = _first_questions(data)
    return bool(qs) and any(
        isinstance(q, dict) and ("answer_correct" in q or "correct_answer" in q)
        for q in qs
    )


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------- planning

def _numbering_has_issue(seq: list[int]) -> bool:
    """True if a module's numbers contain a duplicate or an out-of-order value."""
    seen: set[int] = set()
    prev = 0
    for n in seq:
        if n:
            if n in seen:
                return True
            seen.add(n)
            if prev and n < prev:
                return True
            prev = n
    return False


def plan_renumber(questions: list[dict]) -> dict:
    """Map (section, module) -> {question_index: new_number} for modules whose
    numbers are duplicated or out of order; renumber them 1..N in file order."""
    by: dict[tuple[str, int], list[int]] = {}
    order: list[tuple[str, int]] = []
    for qi, q in enumerate(questions):
        key = (str(q.get("section") or "unknown"), int(q.get("module") or 0))
        if key not in by:
            by[key] = []
            order.append(key)
        by[key].append(qi)
    fixes: dict = {}
    for key in order:
        indices = by[key]
        seq = [int(questions[i].get("number") or 0) for i in indices]
        if _numbering_has_issue(seq):
            fixes[key] = {qi: n + 1 for n, qi in enumerate(indices)}
    return fixes


def answer_fix(tidy_q: dict, report_q: dict) -> tuple[str | None, list[str]]:
    """Return (new_answer or None if unchanged, list of unresolved messages)."""
    claimed = str(tidy_q.get("answer") or "").strip()
    correct = str(report_q.get("correct_answer") or "").strip()
    if report_q.get("answer_correct") is not False:
        return None, []
    if not correct:
        return None, ["no validated correct answer"]
    codes = [o["code"] for o in (tidy_q.get("options") or [])]
    if codes and correct not in codes:
        return None, [f"validated answer {correct!r} not among option codes {codes}"]
    return correct, []


def should_rewrite_explanation(tidy_q: dict, report_q: dict) -> bool:
    """Rewrite the explanation when the answer is corrected — the old explanation
    argues for the old answer, so keeping it would contradict the fix — or when
    the explanation was judged unreasonable. Either way a validated correct
    answer must exist so we can explain something concrete."""
    if not str(report_q.get("correct_answer") or "").strip():
        return False
    new_answer, _ = answer_fix(tidy_q, report_q)
    if new_answer is not None:
        return True
    return report_q.get("explanation_reasonable") is False


# ------------------------------------------------------------------ llm calls

def _explain_user_prompt(tidy_q: dict, report_q: dict, idx: int) -> str:
    parts = [f"QUESTION {idx}"]
    if tidy_q.get("background"):
        parts += ["[Background]", str(tidy_q["background"])]
    parts += ["[Question]", str(tidy_q.get("question") or "")]
    options = tidy_q.get("options") or []
    if options:
        parts += ["[Options]"] + [f"{o.get('code')}. {o.get('label')}" for o in options]
    else:
        parts += ["[Free response — no options]"]
    answer = report_q.get("correct_answer") or tidy_q.get("answer")
    parts.append(f"validated CORRECT ANSWER: {answer}")
    notes = (report_q.get("notes") or "").strip()
    if notes:
        parts.append(f"Reviewer note (guidance only): {notes}")
    return "\n".join(parts)


def _explain_messages(group, *, json_mode: bool, img_index=None) -> list[dict]:
    """Build the system/user message pair for one explanation group.

    When ``img_index`` is provided (multimodal), figures referenced by the
    questions are attached so the model can see them while rewriting.
    """
    if len(group) == 1:
        (idx, tidy_q, report_q) = group[0]
        user_text = _explain_user_prompt(tidy_q, report_q, idx)
    else:
        blocks = [
            f"===== QUESTION {idx} =====\n\n"
            f"{_explain_user_prompt(tidy_q, report_q, idx)}"
            for idx, tidy_q, report_q in group
        ]
        closing = (
            'Return ONLY a JSON object: {"explanations": ["...", "..."]}, one '
            "explanation per question, in the same order."
            if json_mode
            else 'Return ONLY a JSON array of explanation strings, one per question, '
            "in the same order."
        )
        user_text = "\n\n".join(blocks) + "\n\n" + closing
    user_content = (
        paper_images.attach_images(user_text, img_index)
        if img_index is not None
        else user_text
    )
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _explain_texts(data, *, batch: int) -> list[str]:
    """Extract the per-question explanation strings from a parsed response."""
    if batch == 1:
        return [str(data.get("explanation") or "")] if isinstance(data, dict) else []
    if isinstance(data, list):
        return [str(x or "") for x in data]
    if isinstance(data, dict) and isinstance(data.get("explanations"), list):
        return [str(x or "") for x in data["explanations"]]
    return []


def llm_explanations(
    model: str,
    items: list[tuple[int, dict, dict]],
    *,
    batch: int,
    json_mode: bool,
    max_retries: int,
    timeout: int,
    batch_size: int = 1,
    poll_interval: int = openai_batch.POLL_INTERVAL,
    multimodal: bool = False,
) -> dict[int, str]:
    """Rewrite explanations; returns {question_index: new_explanation}."""
    out: dict[int, str] = {}
    groups = [items[i : i + batch] for i in range(0, len(items), batch)]
    img_index = paper_images.ImageIndex(IMAGE_ROOTS) if multimodal else None

    def consume(gi, data):
        group = groups[gi - 1]
        texts = _explain_texts(data, batch=batch)
        for n, (idx, _t, _r) in enumerate(group):
            if n < len(texts) and texts[n].strip():
                out[idx] = texts[n].strip()
        return texts

    if batch_size > 1 and openai_batch.is_openai(model):
        tasks = [
            (
                f"group-{gi}",
                _explain_messages(g, json_mode=json_mode, img_index=img_index),
                parse_json_response,
            )
            for gi, g in enumerate(groups, 1)
        ]
        if tasks:
            print(
                f"  sending {len(tasks)} explanation rewrite(s) through the OpenAI "
                f"Batch API (--batch-size {batch_size})"
            )
            results = openai_batch.batch_completions(
                model,
                tasks,
                batch_size=batch_size,
                json_mode=json_mode,
                max_retries=max_retries,
                poll_interval=poll_interval,
                log=print,
            )
            for gi, group in enumerate(groups, 1):
                texts = consume(gi, results.get(f"group-{gi}"))
                print(
                    f"  got {len(texts)} explanation(s) for questions "
                    f"{group[0][0] + 1}-{group[-1][0] + 1} ({gi}/{len(groups)})"
                )
    else:
        for gi, group in enumerate(groups, 1):
            print(
                f"  rewriting explanations {group[0][0] + 1}-{group[-1][0] + 1} "
                f"({gi}/{len(groups)}) ...",
                end="",
                flush=True,
            )
            data = _call(
                model,
                _explain_messages(group, json_mode=json_mode, img_index=img_index),
                json_mode=json_mode,
                max_retries=max_retries,
                timeout=timeout,
            )
            texts = consume(gi, data)
            print(f" {len(group)} done" if len(texts) == len(group) else f" partial ({len(texts)})")
    return out


def _call(model, messages, *, json_mode, max_retries, timeout):
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        from litellm import completion

        kwargs: dict = {"model": model, "messages": messages, "timeout": timeout}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = completion(**kwargs)
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
            continue
        data = parse_json_response(raw)
        if data is not None:
            return data
        last_error = ValueError(f"response was not usable JSON: {raw[:200]!r}")
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": "That was not valid JSON. Output ONLY the JSON described "
                "above and nothing else.",
            }
        )
    raise RuntimeError(f"failed after {max_retries + 1} attempt(s): {last_error}") from last_error


# ------------------------------------------------------------------- applying

def fix_question(tidy_q: dict, report_q: dict, *, rewrite: bool, new_explanation: str | None):
    q = dict(tidy_q)  # shallow copy; options list is shared but never mutated
    changes: dict = {"answer_fixed": None, "explanation_rewritten": False}
    unresolved: list[str] = []

    new_answer, un = answer_fix(tidy_q, report_q)
    unresolved.extend(un)
    if new_answer is not None:
        changes["answer_fixed"] = {"from": q.get("answer"), "to": new_answer}
        q["answer"] = new_answer

    if rewrite and should_rewrite_explanation(tidy_q, report_q):
        if new_explanation:
            changes["explanation_rewritten"] = True
            q["explanations"] = new_explanation
        else:
            unresolved.append("explanation needs rewrite but LLM produced none")

    if unresolved:
        changes["unresolved"] = unresolved
    return q, changes


# --------------------------------------------------------------------- output

def fix_one(
    model: str,
    tidy_path: Path,
    validated_path: Path | None,
    args,
) -> int:
    data = load_json(tidy_path)
    if is_validated_report(data):
        # Self-contained: the input IS an already-validated report (its questions
        # carry both the tidy fields and the verdicts), no separate file needed.
        report = data
        report_qs = _first_questions(data)
        tidy = [
            {
                "background": q.get("background", ""),
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "answer": q.get("answer", ""),
                "explanations": q.get("explanations", ""),
                "section": q.get("section", ""),
                "module": q.get("module", 0),
                "number": q.get("number", 0),
            }
            for q in report_qs
        ]
    else:
        tidy = data
        if not isinstance(tidy, list):
            print(f"  ! {tidy_path.name}: tidy JSON is not an array", file=sys.stderr)
            return 1
        report = load_json(validated_path)
        if not isinstance(report, dict) or not isinstance(report.get("questions"), list):
            print(
                f"  ! {validated_path.name}: validated report is not a report JSON",
                file=sys.stderr,
            )
            return 1
        report_qs = report["questions"]

    if len(report_qs) != len(tidy):
        print(
            f"  ! {tidy_path.name}: {len(tidy)} tidy questions vs "
            f"{len(report_qs)} validated questions — cannot align",
            file=sys.stderr,
        )
        return 1

    renumber = plan_renumber(tidy) if getattr(args, "renumber", False) else {}

    rewrite = not args.answers_only
    need_rewrite = [
        (i, tidy[i], report_qs[i])
        for i in range(len(tidy))
        if rewrite and should_rewrite_explanation(tidy[i], report_qs[i])
    ]
    new_expls: dict[int, str] = {}
    if need_rewrite:
        new_expls = llm_explanations(
            model,
            need_rewrite,
            batch=args.batch,
            json_mode=args.json_mode,
            max_retries=args.max_retries,
            timeout=args.timeout,
            batch_size=args.batch_size,
            multimodal=args.multimodal,
        )

    corrected: list[dict] = []
    changes_list: list[dict] = []
    stats = {"answers_fixed": 0, "explanations_rewritten": 0, "numbers_fixed": 0, "unresolved": 0}
    for i, (tidy_q, report_q) in enumerate(zip(tidy, report_qs)):
        q, changes = fix_question(
            tidy_q, report_q, rewrite=rewrite, new_explanation=new_expls.get(i)
        )
        for key, mapping in renumber.items():
            if i in mapping:
                changes["number_fixed"] = {"from": q.get("number"), "to": mapping[i]}
                q["number"] = mapping[i]
                stats["numbers_fixed"] += 1
        if changes["answer_fixed"]:
            stats["answers_fixed"] += 1
        if changes["explanation_rewritten"]:
            stats["explanations_rewritten"] += 1
        if changes.get("unresolved"):
            stats["unresolved"] += 1
        changes["idx"] = i + 1
        changes["section"] = q.get("section", "")
        changes["module"] = q.get("module", 0)
        corrected.append(q)
        changes_list.append(changes)

    out_dir = args.output_dir or (tidy_path.parent.parent / "fixed")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base_stem(tidy_path.stem)
    fixed_path = out_dir / f"{base}.fixed.json"
    fixed_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")
    changes_path = out_dir / f"{base}.changes.json"
    changes_path.write_text(
        json.dumps({"info": report.get("info"), "summary": stats, "changes": changes_list},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        rel = fixed_path.relative_to(Path.cwd())
    except ValueError:
        rel = fixed_path
    print(
        f"  -> {rel}: {len(corrected)} questions, "
        f"{stats['answers_fixed']} answers fixed, "
        f"{stats['explanations_rewritten']} explanations rewritten, "
        f"{stats['numbers_fixed']} numbers fixed, "
        f"{stats['unresolved']} unresolved"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fix problems flagged by validate_tidy_questions.py: correct answers, "
            "rewrite poor explanations (LLM), and renumber problematic modules."
        )
    )
    parser.add_argument("--tidy", action="append", required=True, metavar="FILE",
                        help="a tidy JSON, or an already-validated report JSON "
                             "(repeatable).")
    parser.add_argument("--validated", action="append", metavar="FILE",
                        help="validated report (repeatable). If omitted, derived from "
                             "each --tidy file as <tidy dir>/validated/<stem>.validated.json.")
    parser.add_argument("--answers-only", action="store_true",
                        help="correct answers only; do not call the LLM to rewrite "
                             "explanations. Offline/deterministic.")
    parser.add_argument("--renumber", action="store_true",
                        help="renumber modules whose numbers are duplicated/out of "
                             "order to 1..N (default: keep tidy numbers).")
    parser.add_argument("--batch", type=int, default=1,
                        help="explanations to rewrite per LLM call (default: 1).")
    parser.add_argument("--model", default=os.environ.get("LITELLM_MODEL", DEFAULT_FLASH_MODEL),
                        help=f"litellm model string (default: $LITELLM_MODEL or {DEFAULT_FLASH_MODEL}).")
    parser.add_argument("--output-dir", type=Path,
                        help="Where fixed files go (default: <tidy dir>/fixed).")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="LLM retries per call when JSON parsing fails (default: 2).")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-call timeout in seconds (default: 300).")
    parser.add_argument("--json-mode", action="store_true",
                        help="Request provider-native JSON response format.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="LLM calls per OpenAI Batch API job (default: 1 = "
                             "synchronous calls). With N > 1 and a native OpenAI "
                             "model, requests go through OpenAI's Batch API (50%% "
                             "cheaper, processed within 24h). Ignored for non-OpenAI "
                             "providers.")
    parser.add_argument("--multimodal", action="store_true",
                        help="Attach the figures a question references to the request "
                             "so the model can see them while rewriting explanations "
                             "(use a vision-capable model such as gpt-5.6-terra).")
    args = parser.parse_args()

    if args.batch < 1:
        parser.error("--batch must be >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    tidy_paths = [Path(p) for p in args.tidy]
    explicit = args.validated
    if explicit is not None:
        if len(explicit) != len(tidy_paths):
            parser.error("--validated count must match --tidy count (or be omitted)")
        validated: list[Path | None] = [Path(p) for p in explicit]
    else:
        validated = []
        for tp in tidy_paths:
            try:
                data = load_json(tp)
            except Exception:
                data = None
            if data is not None and is_validated_report(data):
                validated.append(None)  # self-contained: no separate file needed
            else:
                # Auto-derive the validated sibling, stripping any suffix first
                # so we don't append to an already-suffixed stem. It lives next
                # to the tidy/ folder (same level), not inside it.
                validated.append(
                    tp.parent.parent / "validated" / f"{base_stem(tp.stem)}.validated.json"
                )

    for p in tidy_paths + [v for v in validated if v is not None]:
        if not p.is_file():
            print(f"File not found: {p}", file=sys.stderr)
            return 1

    if args.answers_only:
        print("\n[answers-only mode: no LLM calls]")
    else:
        print(
            f"\n[will rewrite flagged explanations with {args.model}]"
        )
        if args.batch_size > 1:
            if openai_batch.is_openai(args.model):
                print(f"[OpenAI Batch API enabled (--batch-size {args.batch_size})]")
            else:
                print(
                    f"[--batch-size requires a native OpenAI provider; {args.model!r} "
                    "is not OpenAI — using synchronous calls]",
                    file=sys.stderr,
                )

    failures = 0
    for i, (tidy_path, validated_path) in enumerate(zip(tidy_paths, validated), 1):
        if validated_path is None:
            print(f"\n=== [{i}/{len(tidy_paths)}] {tidy_path.name} (self-contained)")
        else:
            print(f"\n=== [{i}/{len(tidy_paths)}] {tidy_path.name} + {validated_path.name}")
        try:
            failures += fix_one(args.model, tidy_path, validated_path, args)
        except Exception as exc:
            failures += 1
            print(f"  FAIL {tidy_path.name}: {exc}", file=sys.stderr)

    if failures:
        print(f"\nFinished with {failures} file failure(s).", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
