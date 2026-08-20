#!/usr/bin/env python3
"""End-to-end pipeline: PDF -> finalized paper bundle.

Runs transcribe (MinerU image crop) -> tidy (vision, with previous-question carry) -> validate -> fix -> finalize.
Requires a vision-capable model for tidy (e.g. gpt-5.6-terra).

Example:
    python scripts/pipeline.py \\
        --paper-pdf papers/pdfs/2024/demo-paper.pdf \\
        --answer-pdf papers/pdfs/2024/demo-key.pdf \\
        --output papers/final/demo-paper \\
        --year 2024 --month 5 --area Demo --code X

All intermediate JSONs are kept under --work-dir (default papers/jsons/<year>).
Pass --work-dir to control their location. Use --skip-tidy / --skip-validate /
--skip-fix to reuse existing intermediates.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_FLASH_MODEL = "gpt-5.6-luna"



def _regex_parse_key(elements) -> dict:
    import html as _html
    import re as _re
    answer_map: dict[tuple[str, int, int], str] = {}
    x_positions: set[tuple[str, int, int]] = set()
    counts: dict[tuple[str, int], int] = {}
    group_re = _re.compile(r"(\d+)-(\d+):\s*([A-Za-zxX]+)")
    item_re = _re.compile(r"(\d+)\.\s*([^ \s]+)")
    section: str | None = None
    module: int | None = None
    math_seen: dict[str, set[int]] = {}
    for e in elements:
        raw_text = str(e.get("text", ""))
        if isinstance(e.get("table_body"), str):
            raw_text += " " + _html.unescape(_re.sub(r"<[^>]+>", " ", e["table_body"]))
        t = _html.unescape(_re.sub(r"<[^>]+>", " ", raw_text))
        if not t.strip():
            continue
        head = _re.sub(r"\s+", "", t.lower())
        if "英语" in head and ":" not in head and "|" not in head:
            section = "reading_writing"
            module = None
            continue
        if "数学" in head and ":" not in head and "|" not in head:
            section = "math"
            module = None
            continue
        if not section:
            continue
        m = _re.match(r"^\s*M([12])\b", t, _re.IGNORECASE)
        if m:
            module = int(m.group(1))
            math_seen.setdefault(section, set()).add(module)
        elif section == "math" and not _re.search(r"\bM([12])\b", t, _re.IGNORECASE):
            seen = math_seen.get(section, set())
            module = 2 if 2 in seen else 1
        if module is None:
            continue
        key = (section, module)
        if section == "reading_writing":
            for lo, hi, letters in group_re.findall(t):
                lo_i, hi_i = int(lo), int(hi)
                letters = letters.upper()
                expect = hi_i - lo_i + 1
                if len(letters) != expect:
                    letters = letters[:expect].ljust(expect, "x")
                for offset, ch in enumerate(letters):
                    pos = lo_i + offset
                    answer_map[(section, module, pos)] = ch
                    if ch == "X":
                        x_positions.add((section, module, pos))
                counts[key] = max(counts.get(key, 0), hi_i)
        else:
            for num, ans in item_re.findall(t):
                pos = int(num)
                answer_map[(section, module, pos)] = ans
                counts[key] = max(counts.get(key, 0), pos)
    return {"answer_map": answer_map, "x_positions": x_positions, "counts": counts}


def _answer_source(q: dict, key_map: dict) -> dict:
    section, module, number = (str(q.get("section") or ""), int(q.get("module") or 0), int(q.get("number") or 0))
    amap = key_map.get("answer_map", {})
    x_pos = key_map.get("x_positions", set())
    k = (section, module, number)
    if k in x_pos:
        return {"answer_source": "key_x"}
    if k in amap:
        claimed = str(q.get("answer") or "").strip()
        key_ans = str(amap[k]).strip()
        if claimed and claimed.upper() == key_ans.upper():
            return {"answer_source": "key", "key_answer": key_ans}
        return {"answer_source": "key_disputed", "key_answer": key_ans}
    return {"answer_source": "model_solved"}


def _annotation_from_changes(q, changes_qs, position: int | None = None) -> dict:
    for c in changes_qs or []:
        if not isinstance(c, dict):
            continue
        if c.get("idx") != position:
            continue
        if c.get("answer_fixed"):
            note = f"answer corrected {c['answer_fixed']}"
            if c.get("explanation_rewritten"):
                note += "; explanation regenerated"
            return {"status": "disputed", "note": note}
        if c.get("unresolved"):
            return {"status": "unresolved", "note": "; ".join(c["unresolved"])}
        if c.get("explanation_rewritten"):
            return {"status": "rewritten", "note": "explanation regenerated"}
    return {}


def _dispute_note(src: dict) -> dict:
    if src.get("answer_source") != "key_disputed" or src.get("note"):
        return {}
    return {"note": f"answer differs from official key (key: {src.get('key_answer', '?')})"}


def _annotate_fixed_with_provenance(fixed_path: Path, answer_pdf: Path, work_dir: Path) -> None:
    try:
        questions = json.loads(fixed_path.read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            return
    except Exception:
        return
    # load changes
    changes_qs: list[dict] = []
    changes_path = fixed_path.parent / f"{fixed_path.stem.replace('.fixed','')}.changes.json"
    # fallback: try stem without suffix
    if not changes_path.is_file():
        # try with .changes
        alt = fixed_path.with_suffix("").with_suffix(".changes.json")
        if alt.is_file():
            changes_path = alt
    if changes_path.is_file():
        try:
            ch = json.loads(changes_path.read_text(encoding="utf-8"))
            if isinstance(ch, dict) and isinstance(ch.get("changes"), list):
                changes_qs = ch["changes"]
        except Exception:
            pass
    # load answer key json for answer_source
    key_map: dict | None = None
    for base in [work_dir, PROJECT_ROOT / "papers" / "jsons" / answer_pdf.parent.name, PROJECT_ROOT / "papers" / "jsons"]:
        cand = base / f"{answer_pdf.stem}.json"
        if cand.is_file():
            try:
                elems = json.loads(cand.read_text(encoding="utf-8"))
                key_map = _regex_parse_key(elems)
                break
            except Exception:
                pass
    if key_map is None:
        # also try papers/mds
        for base in [PROJECT_ROOT / "papers" / "mds" / answer_pdf.parent.name, PROJECT_ROOT / "papers" / "mds"]:
            cand = base / f"{answer_pdf.stem}.json"
            if cand.is_file():
                try:
                    elems = json.loads(cand.read_text(encoding="utf-8"))
                    key_map = _regex_parse_key(elems)
                    break
                except Exception:
                    pass
    if key_map is None or not key_map.get("answer_map"):
        print("  [annotate] no answer key map found, skipping answer_source")
        return
    # annotate per question
    for idx, q in enumerate(questions, 1):
        src = _answer_source(q, key_map)
        src.update(_annotation_from_changes(q, changes_qs, position=idx))
        src.update(_dispute_note(src))
        q.update(src)
    fixed_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [annotate] added provenance to {len(questions)} questions ({fixed_path.name})")



def run(cmd: list[str], *, step: str) -> None:
    print(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[{step}] failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF -> finalized paper pipeline")
    parser.add_argument("--paper-pdf", type=Path, required=True, help="question paper PDF")
    parser.add_argument("--answer-pdf", type=Path, required=True, help="answer key PDF")
    parser.add_argument("--output", type=Path, required=True, help="final bundle dir (paper.json + imgs/)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--area", required=True, help="e.g. 亚太 / 北美")
    parser.add_argument("--code", required=True, help="e.g. A / B")
    parser.add_argument("--organizer", default="", help="curator name for provenance")
    parser.add_argument("--work-dir", type=Path, help="intermediate JSON root (default papers/jsons/<year>)")
    parser.add_argument("--model", default=None, help="default model for all stages (overrides per-stage)")
    parser.add_argument("--tidy-model", default=None)
    parser.add_argument("--validate-model", default=None)
    parser.add_argument("--fix-model", default=None)
    parser.add_argument("--dpi", type=int, default=150, help="PDF render DPI for tidy (72-400)")
    parser.add_argument("--pages", help="tidy only pages START-END (0-based), for debugging")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--skip-transcribe", action="store_true", help="skip MinerU image cropping (reuse existing images)")
    parser.add_argument("--skip-tidy", action="store_true", help="reuse existing tidy JSON")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-fix", action="store_true")
    parser.add_argument("--answers-only", action="store_true", help="fix without LLM explanation rewrite")
    args = parser.parse_args()

    for p, label in [(args.paper_pdf, "paper"), (args.answer_pdf, "answer key")]:
        if not p.is_file():
            parser.error(f"{label} PDF not found: {p}")
        if p.suffix.lower() != ".pdf":
            parser.error(f"{label} must be a PDF: {p}")

    work_dir = args.work_dir or (PROJECT_ROOT / "papers" / "jsons" / str(args.year))
    work_dir.mkdir(parents=True, exist_ok=True)

    tidy_dir = work_dir / "tidy"
    validated_dir = work_dir / "validated"
    fixed_dir = work_dir / "fixed"

    stem = args.paper_pdf.stem
    tidy_json = tidy_dir / f"{stem}.json"
    validated_json = validated_dir / f"{stem}.validated.json"
    fixed_json = fixed_dir / f"{stem}.fixed.json"

    tidy_model = args.tidy_model or args.model or DEFAULT_FLASH_MODEL
    validate_model = args.validate_model or args.model or DEFAULT_MODEL
    fix_model = args.fix_model or args.model or DEFAULT_FLASH_MODEL

    py = sys.executable

    if not args.skip_transcribe:
        cmd = [py, "scripts/transcribe_pdfs.py"]
        run(cmd, step="transcribe")
    else:
        print("[skip transcribe] reusing existing cropped images")

    if not args.skip_tidy:
        tidy_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            py, "scripts/tidy_questions.py",
            "--model", tidy_model,
            "--pair", str(args.paper_pdf), str(args.answer_pdf),
            "--output-dir", str(tidy_dir),
            "--dpi", str(args.dpi),
        ]
        if args.pages:
            cmd += ["--pages", args.pages]
        if args.json_mode:
            cmd.append("--json-mode")
        run(cmd, step="tidy")
    else:
        if not tidy_json.is_file():
            print(f"--skip-tidy but tidy JSON not found: {tidy_json}", file=sys.stderr)
            return 1
        print(f"[skip tidy] reusing {tidy_json}")

    if not args.skip_validate:
        validated_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            py, "scripts/validate_tidy_questions.py",
            "--tidy", str(tidy_json),
            "--model", validate_model,
            "--output-dir", str(validated_dir),
        ]
        if args.json_mode:
            cmd.append("--json-mode")
        run(cmd, step="validate")
    else:
        if not validated_json.is_file():
            print(f"--skip-validate but validated JSON not found: {validated_json}", file=sys.stderr)
            return 1
        print(f"[skip validate] reusing {validated_json}")

    if not args.skip_fix:
        fixed_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            py, "scripts/fix_tidy_questions.py",
            "--tidy", str(tidy_json),
            "--validated", str(validated_json),
            "--model", fix_model,
            "--output-dir", str(fixed_dir),
        ]
        if args.answers_only:
            cmd.append("--answers-only")
        if args.json_mode:
            cmd.append("--json-mode")
        run(cmd, step="fix")
    else:
        if not fixed_json.is_file():
            print(f"--skip-fix but fixed JSON not found: {fixed_json}", file=sys.stderr)
            return 1
        print(f"[skip fix] reusing {fixed_json}")

    try:
        _annotate_fixed_with_provenance(fixed_json, args.answer_pdf, work_dir)
    except Exception as exc:
        print(f"  [annotate] skipped: {exc}", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    cmd = [
        py, "scripts/finalize_paper.py",
        "--year", str(args.year),
        "--month", str(args.month),
        "--area", args.area,
        "--code", args.code,
        "--paper-pdf", str(args.paper_pdf),
        "--answer-pdf", str(args.answer_pdf),
        "--questions", str(fixed_json),
        "--output", str(args.output),
    ]
    if args.organizer:
        cmd += ["--organizer", args.organizer]
    run(cmd, step="finalize")

    print(f"\nDone. Final bundle at {args.output}/paper.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
