#!/usr/bin/env python3
"""Finalize a paper into a self-contained bundle.

Package the final questions, user-provided metadata, the sha256 fingerprints
of the original PDFs (paper + answer key), and every image the questions
reference into a single directory:

    <output>/
      paper.json
      imgs/
        xxxx.jpg

``paper.json`` carries:

- ``meta``: the user-provided metadata (year, month, area, paper code; when
  ``--organizer`` / ``--curated-at`` are given, who curated it and when); when
  ``--supplement-report`` is given it also carries a ``supplement`` summary
  (expected / extracted / recovered / missing question counts).
- ``source_pdfs``: sha256 fingerprints of the source paper PDF and answer key
  PDF (so a bundle is tied to the exact source it was built from).
- ``questions``: the questions from a fixed/supplemented JSON, with any
  ``[image: images/<hash>.jpg]`` references rewritten to ``[image: imgs/<hash>.jpg]``
  and the referenced files copied under ``imgs/`` — making the bundle
  self-contained. Images are still content-addressed (sha256 filename), never
  embedded as base64.
- ``completeness`` (when ``--supplement-report`` is given): the supplement
  stage's full per-module completeness block, plus per-question
  ``answer_source``/``status`` that travel on the question objects themselves —
  so a partial extraction or a model-solved answer is explicit rather than
  silently presented as complete.

Usage:

    python scripts/finalize_paper.py \\
        --year 2024 --month 5 --area Demo --code X \\
        --paper-pdf papers/pdfs/2024/demo-paper.pdf \\
        --answer-pdf papers/pdfs/2024/demo-key.pdf \\
        --questions papers/jsons/2024/fixed/demo-paper.fixed.json \\
        --output papers/final/demo-paper
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import paper_images
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOTS = [PROJECT_ROOT / "papers" / "jsons", PROJECT_ROOT / "papers" / "mds"]


def _process_image(src: Path, dest: Path) -> bool:
    try:
        with Image.open(src) as im:
            gray = im.convert("L")
            # gray = ImageEnhance.Contrast(gray).enhance(1.5)
            gray = ImageEnhance.Sharpness(gray).enhance(2.0)
            gray.save(dest)
        return True
    except Exception:
        try:
            shutil.copyfile(src, dest)
            return True
        except Exception:
            return False

# Question fields whose text may reference images.
TEXT_FIELDS = ("background", "question", "explanations")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_question_refs(questions: list[dict]) -> list[str]:
    """All ``[image: <ref>]`` refs referenced across a question list."""
    refs: list[str] = []
    for q in questions:
        for field in TEXT_FIELDS:
            refs += paper_images.collect_refs(str(q.get(field) or ""))
        for opt in q.get("options") or []:
            if isinstance(opt, dict):
                refs += paper_images.collect_refs(str(opt.get("label") or ""))
    return refs


def rewrite_question_images(q: dict, ref_to_bundle: dict[str, str]) -> dict:
    """Return a copy of ``q`` with image refs rewritten via ``ref_to_bundle``."""
    q = dict(q)
    for field in TEXT_FIELDS:
        q[field] = paper_images.rewrite_refs(q[field], ref_to_bundle)
    if isinstance(q.get("options"), list):
        q["options"] = [
            {**o, "label": paper_images.rewrite_refs(o["label"], ref_to_bundle)}
            if isinstance(o, dict)
            else o
            for o in q["options"]
        ]
    return q


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bundle final questions, metadata, PDF fingerprints and referenced "
            "images into a self-contained directory."
        )
    )
    parser.add_argument("--year", type=int, required=True, help="paper year, e.g. 2024")
    parser.add_argument("--month", type=int, required=True, help="paper month, e.g. 10")
    parser.add_argument("--area", required=True, help="region, e.g. 北美 / 亚太")
    parser.add_argument("--code", required=True, help="paper code, e.g. A / B (\"\" if none)")
    parser.add_argument("--organizer", default="",
                        help="who organized/curated this bundle (a name or handle), "
                             "for the provenance trail.")
    parser.add_argument("--curated-at", default="",
                        help="ISO timestamp of curation (default: the current local "
                             "date/time).")
    parser.add_argument("--paper-pdf", type=Path, required=True, metavar="FILE")
    parser.add_argument("--answer-pdf", type=Path, required=True, metavar="FILE")
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        metavar="FILE",
        help="questions JSON (a fixed/tidy output, e.g. <...>.fixed.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="bundle directory that will contain paper.json and imgs/.",
    )
    parser.add_argument(
        "--supplement-report",
        type=Path,
        metavar="FILE",
        help="optional supplement_questions.py report (<stem>.supplement.json). When "
             "given, its completeness block is embedded into paper.json as "
             "\"completeness\". The questions' answer_source / status / note fields "
             "already travel on the question objects themselves in the supplemented "
             "JSON.",
    )
    args = parser.parse_args(argv)

    for pdf, label in ((args.paper_pdf, "paper"), (args.answer_pdf, "answer key")):
        if not pdf.is_file():
            print(f"{label} PDF not found: {pdf}", file=sys.stderr)
            return 1
    if not args.questions.is_file():
        print(f"questions file not found: {args.questions}", file=sys.stderr)
        return 1

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        print(
            f"questions file must be a JSON array of questions: {args.questions}",
            file=sys.stderr,
        )
        return 1

    index = paper_images.ImageIndex(IMAGE_ROOTS)
    refs = sorted(set(collect_question_refs(questions)))

    imgs_dir = args.output / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, str] = {}
    missing: list[str] = []
    for ref in refs:
        name = Path(ref).name
        src = index.path(name)
        if src is None:
            missing.append(ref)
            continue
        dest = imgs_dir / name
        if not _process_image(src, dest):
            missing.append(ref)
            continue
        bundle[ref] = f"imgs/{name}"
    for ref in missing:
        print(f"  ! referenced image not found on disk: {ref}", file=sys.stderr)

    paper = {
        "meta": {
            "year": args.year,
            "month": args.month,
            "area": args.area,
            "code": args.code,
        },
        "source_pdfs": {
            "paper": {
                "filename": args.paper_pdf.name,
                "sha256": sha256_file(args.paper_pdf),
            },
            "answer_key": {
                "filename": args.answer_pdf.name,
                "sha256": sha256_file(args.answer_pdf),
            },
        },
        "questions": [rewrite_question_images(q, bundle) for q in questions],
    }
    if args.organizer:
        paper["meta"]["organizer"] = args.organizer
    paper["meta"]["curated_at"] = (args.curated_at or
                                    datetime.now().astimezone().isoformat(timespec="seconds"))

    if args.supplement_report is not None:
        if not args.supplement_report.is_file():
            print(
                f"! supplement report not found: {args.supplement_report}",
                file=sys.stderr,
            )
            return 1
        report = json.loads(args.supplement_report.read_text(encoding="utf-8"))
        comp = report.get("completeness", {})
        # The questions themselves already carry answer_source / status / note
        # (annotated by supplement_questions.py, keyed by the question object so
        # the provenance can never be mis-matched). The report contributes the
        # completeness block plus a compact supplement summary on ``meta``.
        paper["completeness"] = comp
        paper["meta"]["supplement"] = {
            "expected": comp.get("expected_total", 0),
            "recovered": comp.get("recovered", 0),
            "missing": comp.get("missing", 0),
            "extracted": comp.get("extracted", 0),
        }

    paper_path = args.output / "paper.json"
    paper_path.write_text(json.dumps(paper, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  bundled {len(bundle)} image(s), {len(missing)} missing")
    print(f"  -> {paper_path} (+ imgs/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
