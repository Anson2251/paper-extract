#!/usr/bin/env python3
"""Validate the accuracy of tidy_questions.py output using an LLM (via litellm).

Given one or more tidy JSON files (as produced by scripts/tidy_questions.py),
this script asks the LLM to independently solve every question and judge:

  1. whether the extracted "answer" is correct, and
  2. whether the "explanations" are reasonable / accurate.

The paper identity (year / month / area / code) is parsed from each tidy
filename (e.g. "demo-paper.json"), and one validated JSON report is
written per input file with this schema:

    {
      "info": {
        "year": 2024,
        "month": 5,
        "area": "Demo",   # enum: 北美 / 亚太 (unknown regions pass through)
        "code": "X",      # A/B/C/D... ("" when the paper has no code)
        "source": "<input filename>"
      },
      "questions": [
        {
          "idx": 1,
          "section": "reading_writing",  # passed through from tidy output
          "module": 2,                    # 1 or 2 (0 if undetermined)
          "number": 15,                   # question number within its module
          "background": "...",           # same fields as the tidy output
          "question": "...",
          "options": [{"code": "A", "label": "..."}],
          "answer": "A",                 # claimed answer from tidy output
          "explanations": "...",         # claimed explanation
          "correct_answer": "B",         # LLM-validated correct answer ("" if undeterminable)
          "answer_correct": false,        # claimed answer == correct answer
          "explanation_reasonable": false,
          "explanation_accuracy": "poor",  # enum: good | partial | poor | unknown
          "notes": "..."                 # one to three sentences from the model
        },
        ...
      ],
      "summary": {
        "total": 4,
        "answer_correct": 3,
        "answer_mismatch": 1,
        "explanation_reasonable": 2,
        "explanation_unreasonable": 2,
        "explanation_accuracy": {"good": 1, "partial": 1, "poor": 2, "unknown": 0},
        "by_module": {                  # per section/module navigation breakdown
          "reading_writing:1": {"total": 2, "answer_correct": 2, "explanation_reasonable": 2},
          "math:2": {"total": 2, "answer_correct": 1, "explanation_reasonable": 0}
        },
        "numbering_issues": []          # duplicate / out-of-order question numbers
      }
    }

Usage:

    python scripts/validate_tidy_questions.py \
        --tidy papers/jsons/2024/tidy/demo-paper.json \
        --model gpt-5.6-luna

Options:

    --tidy FILE          a tidy JSON produced by tidy_questions.py (repeatable).
    --batch N            validate N questions per LLM call (default: 1 = one call
                         per question for maximum accuracy; larger N is cheaper
                         and faster but risks more mistakes).
    --model NAME         litellm model string (default: $LITELLM_MODEL or
                         gpt-5.6-luna). Any litellm provider/model works.
    --output-dir DIR     where validated JSON files go (default: <paper dir>/validated,
                         a sibling of the tidy folder).
    --max-retries N      LLM retries per call when JSON parsing fails (default: 2).
    --timeout SECONDS    per-call timeout in seconds (default: 300).
    --json-mode          request provider-native JSON mode (OpenAI-compatible
                         providers).
    --batch-size N       LLM calls per OpenAI Batch API job (default: 1 = one
                         synchronous call per group). With N > 1 and a native
                         OpenAI model, requests go through OpenAI's Batch API
                         (50% cheaper, processed within 24h). Combined with
                         --batch M it means M questions per call and N calls
                         per batch job. Ignored for non-OpenAI providers.
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

# Common region values for "area"; anything else is passed through as-is.
AREA_ENUM = ("北美", "亚太")

FILENAME_PAT = re.compile(r"^(?:20)?(\d{2})年(\d{1,2})月(.*?)([A-Za-z])?卷$")


# ---------------------------------------------------------------------- prompts

VERIFY_SYSTEM_PROMPT = """\
You are an expert exam reviewer for this paper (Reading & Writing and Math, \
including free-response questions). An extraction pipeline produced a "claimed \
answer" and a "claimed explanation" for questions it extracted from a paper; \
your job is to validate them.

For every question:
1. Solve it yourself from first principles — never trust the claimed answer.
2. Determine the truly correct answer.
3. Compare it with the claimed answer.
4. Judge whether the claimed explanation is reasonable: accurate, coherent and \
sufficient to justify the correct answer.

Rules:
- Multiple choice: "correct_answer" is one of the option codes (A-E).
- Free response (no options): "correct_answer" is the value (number or \
expression). Compare mathematically — equivalent forms (e.g. "3.43" vs \
"343/100", "2" vs "2.000") count as equal.
- If the question cannot be answered from the given text (e.g. it depends on an \
image, table or equations that are missing), set "correct_answer" to "" and \
explain the limitation in "notes".
- "answer_correct": whether the claimed answer is correct.
- "explanation_accuracy": "good" = correct and sufficient; "partial" = right \
idea but incomplete or slightly off; "poor" = wrong, confused, or \
contradicting the evidence.
- The pipeline also records a claimed "section" / "module" / "number" per \
question (e.g. "Section 1, Module 2, Q15") for navigation. This is secondary \
to answer/explanation accuracy, but if the placement is clearly wrong (wrong \
subject, or a number obviously out of sequence with the other questions), \
mention it briefly in "notes".
- Output ONLY JSON. No markdown fences, no extra prose, no extra keys."""


def _nav_label(q: dict) -> str:
    """Human-readable claimed location, e.g. 'Section 1 (reading_writing), Module 2, Q15'."""
    bits = []
    section = str(q.get("section") or "").strip()
    if section:
        num = 1 if section == "reading_writing" else 2
        bits.append(f"Section {num} ({section})")
    module = q.get("module")
    if module in (1, 2):
        bits.append(f"Module {module}")
    number = q.get("number")
    if number:
        bits.append(f"Q{number}")
    return ", ".join(bits)


def render_question(q: dict, idx: int) -> str:
    parts = [f"QUESTION {idx}"]
    if q.get("background"):
        parts += ["[Background]", str(q["background"])]
    parts += ["[Question]", str(q.get("question") or "")]
    options = q.get("options") or []
    if options:
        parts.append("[Options]")
        parts += [f"{o.get('code')}. {o.get('label')}" for o in options]
    else:
        parts.append("[Free response — no options]")
    return "\n".join(parts)


def _claimed_block(q: dict) -> str:
    nav = _nav_label(q)
    loc = f"Claimed location: {nav}\n" if nav else ""
    return (
        "===== CLAIMED EXTRACTION =====\n"
        f"{loc}"
        f"Claimed answer: {q.get('answer', '')}\n"
        f"Claimed explanation: {q.get('explanations', '')}"
    )


def single_user_prompt(q: dict, idx: int) -> str:
    return (
        f"===== VERIFY QUESTION {idx} =====\n\n"
        f"{render_question(q, idx)}\n\n"
        f"{_claimed_block(q)}\n\n"
        "Return ONLY a JSON object with exactly these keys: "
        '{"correct_answer": "", "answer_correct": true/false, '
        '"explanation_reasonable": true/false, "explanation_accuracy": '
        '"good" | "partial" | "poor", "notes": "..."}'
    )


def batch_user_prompt(group: list[dict], start_idx: int, *, json_mode: bool) -> str:
    blocks = []
    for i, q in enumerate(group, start=start_idx):
        blocks.append(
            f"===== VERIFY QUESTION {i} =====\n\n"
            f"{render_question(q, i)}\n\n"
            f"{_claimed_block(q)}"
        )
    if json_mode:
        closing = (
            'Return ONLY a JSON object: {"verdicts": [{"correct_answer": "", '
            '"answer_correct": true/false, "explanation_reasonable": true/false, '
            '"explanation_accuracy": "good" | "partial" | "poor", "notes": "..."}, ...]} '
            "with one element per question, in the same order."
        )
    else:
        closing = (
            "Return ONLY a JSON array with one object per question, in the same order: "
            '[{"correct_answer": "", "answer_correct": true/false, '
            '"explanation_reasonable": true/false, "explanation_accuracy": '
            '"good" | "partial" | "poor", "notes": "..."}, ...]'
        )
    return "\n\n".join(blocks) + "\n\n" + closing


# ----------------------------------------------------------------- data / info

def load_tidy(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a JSON array of questions, got {type(data).__name__}"
        )
    return [
        q for q in data if isinstance(q, dict) and str(q.get("question") or "").strip()
    ]


def parse_info_from_filename(name: str) -> dict:
    """Parse year/month/area/code from a paper filename in '<YY>年<M>月<area><code>卷' format.

    Unparseable names fall back to year/month 0 and empty area/code so the
    schema stays intact; 'source' always records the original filename.
    """
    base = re.sub(r"\.validated\.json$", "", name)
    base = re.sub(r"\.json$", "", base)
    m = FILENAME_PAT.match(base)
    if not m:
        return {"year": 0, "month": 0, "area": "", "code": "", "source": name}
    yy, mm, area, code = int(m.group(1)), int(m.group(2)), m.group(3), (m.group(4) or "").upper()
    year = 2000 + yy if yy < 70 else 1900 + yy
    return {"year": year, "month": mm, "area": area, "code": code, "source": name}


# ------------------------------------------------------------------ verdicts

def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "correct", "right", "对", "是", "正确", "1"}:
            return True
        if v in {"false", "no", "wrong", "incorrect", "错", "否", "不正确", "0"}:
            return False
    return None


def _codes_equivalent(a: str, b: str) -> bool:
    """Loose string comparison fallback for free-response values."""
    if a == b:
        return True
    na, nb = a.strip().lower(), b.strip().lower()
    if not na or not nb:
        return False
    return re.sub(r"[\s\.\(\)\[\]\-_/]", "", na) == re.sub(
        r"[\s\.\(\)\[\]\-_/]", "", nb
    )


def normalize_accuracy(value) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        for key in ("good", "partial", "poor"):
            if v.startswith(key):
                return key
        if v in {"合理", "准确", "好", "正确"}:
            return "good"
        if v in {"部分合理", "部分准确", "一般", "部分"}:
            return "partial"
        if v in {"不合理", "错误", "差", "不正确"}:
            return "poor"
    return "unknown"


def missing_verdict() -> dict:
    return {
        "correct_answer": "",
        "answer_correct": False,
        "explanation_reasonable": False,
        "explanation_accuracy": "unknown",
        "notes": "模型未返回该题的判定结果，请人工复核。",
    }


def normalize_verdict(item, question: dict) -> dict:
    if not isinstance(item, dict):
        return missing_verdict()
    if isinstance(item.get("verdict"), dict):
        item = item["verdict"]
    claimed = str(question.get("answer") or "").strip()
    correct = str(item.get("correct_answer") or "").strip()
    if correct.lower() in {"none", "n/a", "unknown", "无法确定", "不确定"}:
        correct = ""
    answer_correct = _as_bool(item.get("answer_correct"))
    if answer_correct is None:
        # Fallback: plain string comparison when the model omitted the boolean.
        answer_correct = _codes_equivalent(claimed, correct) if correct else False
    accuracy = normalize_accuracy(item.get("explanation_accuracy"))
    reasonable = _as_bool(item.get("explanation_reasonable"))
    if reasonable is None:
        reasonable = accuracy == "good"
    return {
        "correct_answer": correct,
        "answer_correct": bool(answer_correct),
        "explanation_reasonable": bool(reasonable),
        "explanation_accuracy": accuracy,
        "notes": str(item.get("notes") or "").strip(),
    }


# ------------------------------------------------------------------ llm plumbing

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


def unwrap_verdicts(data, *, expect_list: bool):
    if expect_list:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("verdicts", "questions", "results", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
        return None
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return data[0] if data and isinstance(data[0], dict) else None
    return None


def llm_verdicts(
    model: str,
    messages: list[dict],
    *,
    expect_list: bool,
    json_mode: bool,
    max_retries: int,
    timeout: int,
):
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
        verdicts = unwrap_verdicts(data, expect_list=expect_list)
        if verdicts is not None:
            return verdicts
        last_error = ValueError(f"response was not usable JSON: {raw[:200]!r}")
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    "That was not valid JSON. Output ONLY the JSON described "
                    "above and nothing else."
                ),
            }
        )
    raise RuntimeError(
        f"failed after {max_retries + 1} attempt(s): {last_error}"
    ) from last_error


# -------------------------------------------------------------------- pipeline

def validate_questions(
    model: str,
    questions: list[dict],
    *,
    batch: int,
    json_mode: bool,
    max_retries: int,
    timeout: int,
    batch_size: int = 1,
    poll_interval: int = openai_batch.POLL_INTERVAL,
    multimodal: bool = False,
) -> list[dict]:
    verdicts: list[dict] = []
    total = len(questions)
    n_groups = (total + batch - 1) // batch

    # Build every group's messages up front so the OpenAI Batch API can submit
    # many calls at once; the sync path uses them one group at a time.
    # When multimodal, figures referenced by the questions are attached so the
    # model can actually see them (review needs the images).
    img_index = paper_images.ImageIndex(IMAGE_ROOTS) if multimodal else None
    groups: list[tuple[int, list[dict], list[dict]]] = []
    for start in range(0, total, batch):
        group = questions[start : start + batch]
        user_text = (
            single_user_prompt(group[0], start + 1)
            if batch == 1
            else batch_user_prompt(group, start + 1, json_mode=json_mode)
        )
        user_content = (
            paper_images.attach_images(user_text, img_index)
            if multimodal and img_index is not None
            else user_text
        )
        messages = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        groups.append((start, group, messages))

    def consume(start: int, group: list[dict], data):
        if batch == 1:
            items = [data] if data is not None else []
        else:
            items = data if isinstance(data, list) else []
        for offset, q in enumerate(group, start=start):
            verdict = (
                items[offset - start]
                if offset - start < len(items)
                else missing_verdict()
            )
            verdicts.append(normalize_verdict(verdict, q))

    def parse_verdict(raw: str):
        data = parse_json_response(raw)
        return unwrap_verdicts(data, expect_list=batch > 1)

    if batch_size > 1 and openai_batch.is_openai(model):
        tasks = [
            (f"group-{gi}", messages, parse_verdict)
            for gi, (_start, _group, messages) in enumerate(groups, 1)
        ]
        if tasks:
            print(
                f"  sending {len(tasks)} validation call(s) through the OpenAI "
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
            for gi, (start, group, _messages) in enumerate(groups, 1):
                consume(start, group, results.get(f"group-{gi}"))
    else:
        for group_idx, (start, group, messages) in enumerate(groups, 1):
            print(
                f"  questions {start + 1}-{start + len(group)} "
                f"({group_idx}/{n_groups}) ...",
                end="",
                flush=True,
            )
            items = llm_verdicts(
                model,
                messages,
                expect_list=batch > 1,
                json_mode=json_mode,
                max_retries=max_retries,
                timeout=timeout,
            )
            consume(start, group, items)
            print(" ok")
    return verdicts


def numbering_issues(questions: list[dict]) -> list[str]:
    """Navigation sanity checks: duplicate or out-of-order question numbers within
    a section/module. Gaps (skipped numbers) are not flagged — pages may be
    processed partially or questions skipped."""
    by_mod: dict[tuple[str, int], list[int]] = {}
    order: list[tuple[str, int]] = []
    for q in questions:
        key = (str(q.get("section") or "unknown"), int(q.get("module") or 0))
        if key not in by_mod:
            by_mod[key] = []
            order.append(key)
        n = int(q.get("number") or 0)
        if n > 0:
            by_mod[key].append(n)
    issues: list[str] = []
    for key in order:
        nums = by_mod[key]
        label = f"{key[0]}/M{key[1]}"
        dupes = sorted({n for n in nums if nums.count(n) > 1})
        if dupes:
            issues.append(f"{label}: duplicate number(s) {dupes}")
        prev = 0
        for n in nums:
            if prev and n < prev:
                issues.append(f"{label}: number {n} appears after {prev} (order problem)")
                break
            prev = n
    return issues


def build_report(info: dict, questions: list[dict], verdicts: list[dict]) -> dict:
    report: dict = {"info": info, "questions": []}
    ok_answer = ok_explanation = 0
    dist = {"good": 0, "partial": 0, "poor": 0, "unknown": 0}
    by_module: dict[str, dict[str, int]] = {}
    for i, (q, v) in enumerate(zip(questions, verdicts), 1):
        section = str(q.get("section") or "")
        module = int(q.get("module") or 0)
        key = f"{section or 'unknown'}:{module or 0}"
        mod = by_module.setdefault(key, {"total": 0, "answer_correct": 0, "explanation_reasonable": 0})
        mod["total"] += 1
        report["questions"].append(
            {
                "idx": i,
                "section": section,
                "module": module,
                "number": int(q.get("number") or 0),
                "background": q.get("background", ""),
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "answer": q.get("answer", ""),
                "explanations": q.get("explanations", ""),
                **v,
            }
        )
        ok_answer += 1 if v["answer_correct"] else 0
        ok_explanation += 1 if v["explanation_reasonable"] else 0
        if v["answer_correct"]:
            mod["answer_correct"] += 1
        if v["explanation_reasonable"]:
            mod["explanation_reasonable"] += 1
        dist[v["explanation_accuracy"]] = dist.get(v["explanation_accuracy"], 0) + 1
    # Sort modules: reading_writing before math, then by module number.
    def _mod_key(item: tuple[str, dict]) -> tuple[int, int, str]:
        s, _, m = item[0].partition(":")
        try:
            m_i = int(m)
        except ValueError:
            m_i = 99
        return (0 if s == "reading_writing" else 1, m_i, item[0])

    report["summary"] = {
        "total": len(report["questions"]),
        "answer_correct": ok_answer,
        "answer_mismatch": len(report["questions"]) - ok_answer,
        "explanation_reasonable": ok_explanation,
        "explanation_unreasonable": len(report["questions"]) - ok_explanation,
        "explanation_accuracy": dist,
        "by_module": dict(sorted(by_module.items(), key=_mod_key)),
        "numbering_issues": numbering_issues(questions),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate tidy_questions.py output: check each question's answer and "
            "explanation with an LLM (via litellm) and write a validated JSON report."
        )
    )
    parser.add_argument(
        "--tidy",
        action="append",
        required=True,
        metavar="FILE",
        help="a tidy JSON produced by tidy_questions.py. Repeat for multiple files.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LITELLM_MODEL", DEFAULT_MODEL),
        help=f"litellm model string (default: $LITELLM_MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="questions per LLM call (default: 1 = one call per question).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for validated JSON reports (default: <tidy dir>/validated).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="LLM retries per call when JSON parsing fails (default: 2).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-call timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Request provider-native JSON response format (OpenAI-compatible "
        "providers).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="LLM calls per OpenAI Batch API job (default: 1 = synchronous "
        "calls). With N > 1 and a native OpenAI model, requests go through "
        "OpenAI's Batch API (50%% cheaper, processed within 24h). Ignored for "
        "non-OpenAI providers.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Attach the figures a question references to the request so the "
        "model can see them (review needs the images; use a vision-capable "
        "model such as gpt-5.6-terra).",
    )
    args = parser.parse_args()

    if args.batch < 1:
        parser.error("--batch must be >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    try:
        import litellm
    except ImportError:
        print("litellm is not installed; run `uv sync` first.", file=sys.stderr)
        return 1
    litellm.suppress_debug_info = True

    if args.batch_size > 1:
        if openai_batch.is_openai(args.model):
            print(f"[OpenAI Batch API enabled (--batch-size {args.batch_size})]")
        else:
            print(
                f"[--batch-size requires a native OpenAI provider; {args.model!r} "
                "is not OpenAI — using synchronous calls]",
                file=sys.stderr,
            )

    if args.multimodal:
        print("[multimodal enabled: question figures will be attached to the request]")

    tidy_paths = [Path(p) for p in args.tidy]
    for path in tidy_paths:
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

    failures = 0
    for index, tidy_path in enumerate(tidy_paths, 1):
        print(f"\n=== [{index}/{len(tidy_paths)}] {tidy_path.name}")
        try:
            questions = load_tidy(tidy_path)
        except Exception as exc:
            failures += 1
            print(f"  FAIL loading {tidy_path.name}: {exc}", file=sys.stderr)
            continue
        info = parse_info_from_filename(tidy_path.name)
        print(
            f"  {len(questions)} question(s); info="
            f"year={info['year']} month={info['month']} "
            f"area={info['area']!r} code={info['code']!r}"
        )
        try:
            verdicts = validate_questions(
                args.model,
                questions,
                batch=args.batch,
                json_mode=args.json_mode,
                max_retries=args.max_retries,
                timeout=args.timeout,
                batch_size=args.batch_size,
                multimodal=args.multimodal,
            )
        except Exception as exc:
            failures += 1
            print(f"  FAIL {tidy_path.name}: {exc}", file=sys.stderr)
            continue
        report = build_report(info, questions, verdicts)
        out_dir = args.output_dir or (tidy_path.parent.parent / "validated")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tidy_path.stem}.validated.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            out_rel = out_path.relative_to(Path.cwd())
        except ValueError:
            out_rel = out_path
        s = report["summary"]
        print(
            f"  -> {out_rel}: {s['total']} questions, "
            f"{s['answer_correct']}/{s['total']} answers correct, "
            f"{s['explanation_reasonable']}/{s['total']} explanations reasonable"
        )
        for mod, ms in s["by_module"].items():
            print(
                f"      {mod}: {ms['total']} questions, "
                f"{ms['answer_correct']} answers correct, "
                f"{ms['explanation_reasonable']} explanations reasonable"
            )
        for issue in s["numbering_issues"]:
            print(f"  ! {issue}", file=sys.stderr)
        for q in report["questions"]:
            if not q["answer_correct"] or not q["explanation_reasonable"]:
                print(
                    f"  ! q{q['idx']}: claimed={q['answer']!r} "
                    f"correct={q['correct_answer']!r} answer_ok={q['answer_correct']} "
                    f"explanation_ok={q['explanation_reasonable']} "
                    f"[{q['explanation_accuracy']}]"
                )

    if failures:
        print(f"\nFinished with {failures} file failure(s).", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
