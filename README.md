# paper-extract

Pipeline for turning exam paper PDFs into clean, per-question JSON: render each page as an image, extract questions with a vision LLM carrying cross-page context, validate the extraction, and fix flagged problems — all with an LLM (via litellm).

## Pipeline overview

| # | Script | What it does | Output |
|---|--------|--------------|--------|
| 1 | `scripts/tidy_questions.py` | render each PDF page as an image, extract questions with a vision model (previous question is carried forward so cross-page splits can be merged) | `papers/jsons/<year>/tidy/<paper>.json` |
| 2 | `scripts/validate_tidy_questions.py` | independently solve & judge each question | `<year>/validated/<paper>.validated.json` |
| 3 | `scripts/fix_tidy_questions.py` | apply verdicts: fix answers/explanations | `<year>/fixed/<paper>.fixed.json` (+ `.changes.json`) |
| 4 | `scripts/finalize_paper.py` | bundle final questions, metadata, PDF fingerprints & images | `<output>/paper.json` + `imgs/` |

All LLM scripts support **OpenAI Batch API** (`--batch-size`) where applicable.
Shared plumbing lives in `scripts/openai_batch.py` and `scripts/paper_images.py` (the latter is still used by validate/fix for figure attachment).

Requires a litellm-compatible vision provider. Configure it with standard environment variables (e.g. `OPENAI_API_KEY`, or `OPENAI_API_BASE`/`OPENAI_API_KEY` for an OpenAI-compatible endpoint) and optionally `LITELLM_MODEL`.

```bash
uv sync
```

## One-step pipeline: PDF in, bundle out

```bash
python scripts/pipeline.py \
  --paper-pdf papers/pdfs/2024/demo-paper.pdf \
  --answer-pdf papers/pdfs/2024/demo-key.pdf \
  --output papers/final/demo-paper \
  --year 2024 --month 5 --area Demo --code X \
  --model gpt-5.6-terra --organizer anson
```

Runs `tidy -> validate -> fix -> finalize` sequentially, keeping intermediates under `papers/jsons/<year>/{tidy,validated,fixed}` (override with `--work-dir`). Reuse existing stages with `--skip-tidy` / `--skip-validate` / `--skip-fix`. See `python scripts/pipeline.py --help` for all options.

## 1. Tidy questions

```bash
python scripts/tidy_questions.py \
    --model gpt-5.6-terra \
    --pair papers/pdfs/2024/demo-paper.pdf \
           papers/pdfs/2024/demo-key.pdf
```

Arguments are question-paper/answer-key PDF pairs (`--pair` can be repeated). Each PDF is rendered page by page (default 150 DPI, tunable with `--dpi`). For every paper page the vision model receives:

- the image of that page,
- all images of the answer-key PDF (so it can transcribe the official key), and
- the previous page's last question (as JSON) — so a question split across two pages can be merged. The model returns `{"updated_previous": <merged or null>, "questions": [...]}`; sequential processing guarantees cross-page questions are reconstructed.

The LLM's role is **tidy-up, not grading**: it faithfully transcribes the paper's and official answer key's content — including any mistakes the key might contain. It never corrects or editorializes; accuracy is judged by `validate_tidy_questions.py`. The model returns:

```json
{
  "updated_previous": null,
  "questions": [
    {
      "background": "passage or setup text (verbatim)",
      "question": "question wording (verbatim)",
      "options": [{"code": "A", "label": "option text"}],
      "answer": "option code transcribed verbatim from the answer key",
      "explanations": "transcribed from the answer key if present, else LLM-written, never correcting the key",
      "section": "reading_writing | math",
      "module": 1,
      "number": 15
    }
  ]
}
```

`section` / `module` / `number` are the question's location for navigation (Section 1 = Reading & Writing, Section 2 = Math; each split into Module 1/2). They are read from the page headers visible in the image.

Output goes to `papers/jsons/<year>/tidy/<paper>.json` (override with `--output-dir`). Run `python scripts/tidy_questions.py --help` for all options (e.g. `--pages 27-40` to only process certain pages, `--dpi 200` to increase render resolution, or `--json-mode`).

## 2. Validate tidy output

```bash
python scripts/validate_tidy_questions.py \
    --tidy papers/jsons/2024/tidy/demo-paper.json \
    --model gpt-5.6-luna
```

`validate` checks that the tidy extraction is accurate: for each question it asks the LLM to independently solve it and judge whether the claimed `answer` is correct and whether `explanations` are reasonable.

## 3. Fix validated output

```bash
python scripts/fix_tidy_questions.py \
    --tidy papers/jsons/2024/tidy/demo-paper.json \
    --model gpt-5.6-luna
```

For each flagged question it corrects the answer and rewrites the explanation when needed. Numbers are left as extracted (no supplement stage is needed because cross-page merges already happen in tidy).

For each tidy file it writes `<out>/<stem>.fixed.json` and `<out>/<stem>.changes.json` under `--output-dir` (default `<paper dir>/fixed`).

## 4. Finalize (bundle a paper)

```bash
python scripts/finalize_paper.py \
    --year 2024 --month 5 --area Demo --code X \
    --paper-pdf papers/pdfs/2024/demo-paper.pdf \
    --answer-pdf papers/pdfs/2024/demo-key.pdf \
    --questions papers/jsons/2024/fixed/demo-paper.fixed.json \
    --organizer anson \
    --output papers/final/demo-paper
```

Packages a paper into a single self-contained directory. `paper.json` contains `meta`, `source_pdfs` sha256 fingerprints, `questions`, and optionally `completeness` when a supplement report is supplied.

## OpenAI Batch API

`validate`/`fix` support the Batch API via `--batch-size`. `tidy` is sequential (it must carry the previous question forward) so `--batch-size` is ignored there.

## Notes

- The tidy model is a transcriber, not a grader: `answer` and `explanations` are taken verbatim from the official answer key, including any errors in it.
- Cross-page questions are merged via `updated_previous` — the previous page's last question is carried forward and the model may return a merged complete version.
- Answers marked `x` in the answer key (missing official answer) are answered by the model itself.
- Free-response math questions have `"options": []` and the answer value as a string in `"answer"`.
