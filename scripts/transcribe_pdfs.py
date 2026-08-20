#!/usr/bin/env python3
"""Transcribe PDFs under papers/pdfs/ to markdown/json under papers/mds/ and papers/jsons/."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mineru import MinerU
from mineru.models import ExtractResult, Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_ROOT = PROJECT_ROOT / "papers" / "pdfs"
MD_ROOT = PROJECT_ROOT / "papers" / "mds"
JSON_ROOT = PROJECT_ROOT / "papers" / "jsons"
BATCH_SIZE = 50
TARGETS = ("md", "json")


def pdf_output_paths(pdf: Path, targets: set[str]) -> dict[str, Path]:
    rel = pdf.relative_to(PDF_ROOT)
    paths: dict[str, Path] = {}
    if "md" in targets:
        paths["md"] = MD_ROOT / rel.with_suffix(".md")
    if "json" in targets:
        paths["json"] = JSON_ROOT / rel.with_suffix(".json")
    return paths


def outputs_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def save_images(images: list[Image], directory: Path) -> None:
    if not images:
        return
    img_dir = directory / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        (img_dir / img.name).write_bytes(img.data)


def save_markdown(result: ExtractResult, path: Path) -> None:
    if result.markdown is None:
        raise ValueError("extract succeeded but markdown is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.markdown, encoding="utf-8")
    save_images(result.images, path.parent)


def save_content_list(result: ExtractResult, path: Path) -> None:
    if result.content_list is None:
        raise ValueError("extract succeeded but content_list is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.content_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_images(result.images, path.parent)


def collect_pdfs(targets: set[str], *, skip_existing: bool) -> list[Path]:
    pdfs = sorted(PDF_ROOT.rglob("*.pdf"))
    if not skip_existing:
        return pdfs
    return [pdf for pdf in pdfs if not outputs_complete(pdf_output_paths(pdf, targets))]


def transcribe_batch(
    client: MinerU,
    pdfs: list[Path],
    targets: set[str],
    *,
    model: str | None,
    language: str | None,
) -> list[tuple[Path, str, Path | Exception]]:
    sources = [str(pdf) for pdf in pdfs]
    output_sets = [pdf_output_paths(pdf, targets) for pdf in pdfs]
    results: list[tuple[Path, str, Path | Exception]] = []

    kwargs: dict = {}
    if model is not None:
        kwargs["model"] = model
    if language is not None:
        kwargs["language"] = language

    try:
        extracted = list(client.extract_batch(sources, **kwargs))
    except Exception as exc:
        for outputs in output_sets:
            for target, path in outputs.items():
                results.append((path, target, exc))
        return results

    if len(extracted) != len(pdfs):
        mismatch = RuntimeError(f"expected {len(pdfs)} results, got {len(extracted)}")
        for outputs in output_sets:
            for target, path in outputs.items():
                results.append((path, target, mismatch))
        return results

    for outputs, result in zip(output_sets, extracted, strict=True):
        if result.state == "failed":
            message = RuntimeError(result.error or f"extract failed (err_code={result.err_code})")
            for target, path in outputs.items():
                results.append((path, target, message))
            continue

        for target, path in outputs.items():
            try:
                if target == "md":
                    save_markdown(result, path)
                elif target == "json":
                    save_content_list(result, path)
                else:
                    raise ValueError(f"unknown target: {target}")
                results.append((path, target, path))
            except Exception as exc:
                results.append((path, target, exc))

    return results


def parse_targets(raw: str) -> set[str]:
    targets = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = targets - set(TARGETS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown targets: {', '.join(sorted(unknown))}")
    if not targets:
        raise argparse.ArgumentTypeError("at least one target is required")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe papers/pdfs/**/*.pdf to papers/mds/**/*.md and/or "
            "papers/jsons/**/*.json using MinerU."
        )
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs whose requested outputs already exist.",
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=parse_targets("md,json"),
        help="Comma-separated output targets: md, json (default: md,json).",
    )
    parser.add_argument(
        "--model",
        default="vlm",
        choices=["pipeline", "vlm", "html"],
        help="MinerU model version (default: vlm).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Document language code (default: en for exam papers).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Files per MinerU batch request (default: {BATCH_SIZE}, max 50).",
    )
    args = parser.parse_args()
    targets: set[str] = args.targets

    if not PDF_ROOT.is_dir():
        print(f"PDF root not found: {PDF_ROOT}", file=sys.stderr)
        return 1

    if not os.environ.get("MINERU_TOKEN"):
        print(
            "MINERU_TOKEN is not set. Create a token on the MinerU API Management page "
            "and export it before running this script.",
            file=sys.stderr,
        )
        return 1

    pdfs = collect_pdfs(targets, skip_existing=args.skip_existing)
    if not pdfs:
        print("No PDFs to transcribe.")
        return 0

    batch_size = max(1, min(args.batch_size, 50))
    if "md" in targets:
        MD_ROOT.mkdir(parents=True, exist_ok=True)
    if "json" in targets:
        JSON_ROOT.mkdir(parents=True, exist_ok=True)

    target_dirs = []
    if "md" in targets:
        target_dirs.append(str(MD_ROOT))
    if "json" in targets:
        target_dirs.append(str(JSON_ROOT))

    print(
        f"Transcribing {len(pdfs)} PDF(s) from {PDF_ROOT} "
        f"-> {', '.join(target_dirs)} [{', '.join(sorted(targets))}]"
    )

    failures = 0
    with MinerU() as client:
        for start in range(0, len(pdfs), batch_size):
            batch = pdfs[start : start + batch_size]
            print(f"\nBatch {start // batch_size + 1}: {len(batch)} file(s)")
            for pdf in batch:
                print(f"  - {pdf.relative_to(PDF_ROOT)}")

            for path, target, outcome in transcribe_batch(
                client,
                batch,
                targets,
                model=args.model,
                language=args.language,
            ):
                root = MD_ROOT if target == "md" else JSON_ROOT
                rel = path.relative_to(root)
                label = f"{target}:{rel}"
                if isinstance(outcome, Exception):
                    failures += 1
                    print(f"  FAIL {label}: {outcome}", file=sys.stderr)
                else:
                    print(f"  OK   {label}")

    if failures:
        print(f"\nFinished with {failures} failure(s).", file=sys.stderr)
        return 1

    print(f"\nDone. Wrote outputs under {', '.join(target_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
