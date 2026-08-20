"""Integration tests for the tidy/validate pipelines with mocked LLM calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import litellm
import tidy_questions as tq
import validate_tidy_questions as vq


def _args(output_dir, **over):
    base = dict(
        pages=None,
        output_dir=output_dir,
        model="fake-model",
        json_mode=False,
        max_retries=1,
        timeout=30,
        batch_size=1,
        dpi=150,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _write_pdf(tmp_path, name="paper.pdf"):
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _fake_images(pdfs):
    return [b"page0", b"page1", b"page2"]


def test_process_pair_sequential_and_dedupe(tmp_path, monkeypatch):
    paper = _write_pdf(tmp_path, "paper.pdf")
    answer = _write_pdf(tmp_path, "answer.pdf")
    monkeypatch.setattr(tq, "pdf_to_images", lambda p, dpi=150: [b"p0", b"p1"] if "paper" in str(p) else [b"ans"])

    def fake_extract(model, content, *, json_mode, max_retries, timeout):
        # first page returns 1 question, second page returns same question text duplicated + new one
        txt = str(content)
        if "PAGE: 0" in txt:
            return {"updated_previous": None, "questions": [{"question": "Q1", "options": [], "answer": "A", "explanations": "e", "section": "math", "module": 1, "number": 1}]}
        else:
            return {"updated_previous": None, "questions": [{"question": "Q1", "options": [], "answer": "A", "explanations": "e", "section": "math", "module": 1, "number": 1}, {"question": "Q2", "options": [], "answer": "B", "explanations": "e", "section": "math", "module": 1, "number": 2}]}

    monkeypatch.setattr(tq, "extract_page", fake_extract)
    out_dir = tmp_path / "tidy"
    pages, n_q, failures = tq.process_pair(_args(out_dir), paper, answer, 1, 1)
    assert failures == 0
    data = json.loads((out_dir / "paper.pdf".replace(".pdf", ".json") if False else out_dir / "paper.json").read_text() if (out_dir / "paper.json").exists() else (out_dir / f"{paper.stem}.json").read_text())
    # dedupe should drop duplicate Q1 from second page
    assert len(data) == 2
    assert data[0]["question"] == "Q1"
    assert data[1]["question"] == "Q2"


def test_process_pair_carries_previous_and_updates(tmp_path, monkeypatch):
    paper = _write_pdf(tmp_path, "paper.pdf")
    answer = _write_pdf(tmp_path, "answer.pdf")
    monkeypatch.setattr(tq, "pdf_to_images", lambda p, dpi=150: [b"p0", b"p1"] if "paper" in str(p) else [b"ans"])

    calls = []

    def fake_extract(model, content, *, json_mode, max_retries, timeout):
        calls.append(content)
        if len(calls) == 1:
            return {"updated_previous": None, "questions": [{"question": "Q1 part", "options": [], "answer": "A", "explanations": "e", "section": "reading_writing", "module": 1, "number": 1}]}
        else:
            # verify previous was carried
            text_parts = [c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"]
            assert any("Q1 part" in t for t in text_parts)
            return {"updated_previous": {"question": "Q1 complete", "options": [{"code": "A", "label": "x"}], "answer": "A", "explanations": "e", "section": "reading_writing", "module": 1, "number": 1}, "questions": []}

    monkeypatch.setattr(tq, "extract_page", fake_extract)
    out_dir = tmp_path / "tidy"
    pages, n_q, failures = tq.process_pair(_args(out_dir), paper, answer, 1, 1)
    assert failures == 0
    data = json.loads((out_dir / f"{paper.stem}.json").read_text())
    assert len(data) == 1
    assert data[0]["question"] == "Q1 complete"


def test_process_pair_pages_filter(tmp_path, monkeypatch):
    paper = _write_pdf(tmp_path, "paper.pdf")
    answer = _write_pdf(tmp_path, "answer.pdf")
    monkeypatch.setattr(tq, "pdf_to_images", lambda p, dpi=150: [b"p0", b"p1", b"p2"] if "paper" in str(p) else [b"ans"])
    monkeypatch.setattr(tq, "extract_page", lambda *a, **k: {"updated_previous": None, "questions": [{"question": "Q", "options": [], "answer": "A", "explanations": "e", "section": "math", "module": 1, "number": 1}]})
    out_dir = tmp_path / "tidy"
    args = _args(out_dir, pages="1-1")
    pages, n_q, failures = tq.process_pair(args, paper, answer, 1, 1)
    assert pages == 1
    assert failures == 0


def _fake_completion(contents):
    it = iter(contents)

    def completion(**kwargs):
        item = next(it)
        if isinstance(item, Exception):
            raise item
        body = SimpleNamespace(message=SimpleNamespace(content=item))
        return SimpleNamespace(choices=[body])

    return completion


def test_extract_page_returns_updated_previous_and_questions(monkeypatch):
    monkeypatch.setattr(litellm, "completion", _fake_completion(['{"updated_previous": null, "questions": [{"question": "Q"}]}']))
    monkeypatch.setattr(tq, "time", SimpleNamespace(sleep=lambda *a: None))
    out = tq.extract_page("m", [{"type": "text", "text": "p"}], json_mode=False, max_retries=1, timeout=10)
    assert out["questions"][0]["question"] == "Q"
    assert out["updated_previous"] is None


def test_extract_page_array_fallback(monkeypatch):
    monkeypatch.setattr(litellm, "completion", _fake_completion(['[{"question": "Q"}]']))
    monkeypatch.setattr(tq, "time", SimpleNamespace(sleep=lambda *a: None))
    out = tq.extract_page("m", [{"type": "text", "text": "p"}], json_mode=False, max_retries=1, timeout=10)
    assert out["questions"][0]["question"] == "Q"


def test_extract_page_retries_then_succeeds(monkeypatch):
    responses = [ValueError("boom"), '{"updated_previous": null, "questions": [{"question": "ok"}]}']
    monkeypatch.setattr(litellm, "completion", _fake_completion(responses))
    monkeypatch.setattr(tq, "time", SimpleNamespace(sleep=lambda *a: None))
    out = tq.extract_page("m", [{"type": "text", "text": "p"}], json_mode=False, max_retries=2, timeout=10)
    assert out["questions"][0]["question"] == "ok"


def test_extract_page_raises_after_retries(monkeypatch):
    monkeypatch.setattr(litellm, "completion", _fake_completion([ValueError("x"), ValueError("y")]))
    monkeypatch.setattr(tq, "time", SimpleNamespace(sleep=lambda *a: None))
    with pytest.raises(RuntimeError, match="failed after"):
        tq.extract_page("m", [{"type": "text", "text": "p"}], json_mode=False, max_retries=1, timeout=10)


def _mock_verdicts():
    def fake(model, messages, *, expect_list, json_mode, max_retries, timeout):
        verdict = {"correct_answer": "A", "answer_correct": True,
                   "explanation_reasonable": True, "explanation_accuracy": "good",
                   "notes": "ok"}
        return [verdict, verdict] if expect_list else verdict
    return fake


def _sample_questions():
    return [
        {"section": "reading_writing", "module": 1, "number": 1, "background": "",
         "question": "Q1", "options": [{"code": "A", "label": "a"}],
         "answer": "A", "explanations": "e"},
        {"section": "reading_writing", "module": 1, "number": 2, "background": "",
         "question": "Q2", "options": [{"code": "A", "label": "a"}],
         "answer": "A", "explanations": "e"},
    ]


def test_validate_questions_single_and_batch(monkeypatch):
    monkeypatch.setattr(vq, "llm_verdicts", _mock_verdicts())
    qs = _sample_questions()

    single = vq.validate_questions("m", qs, batch=1, json_mode=False, max_retries=0, timeout=10)
    assert len(single) == 2
    assert all(v["answer_correct"] is True for v in single)

    batch = vq.validate_questions("m", qs, batch=2, json_mode=False, max_retries=0, timeout=10)
    assert len(batch) == 2


def test_validate_questions_pads_missing_verdicts(monkeypatch):
    def short(model, messages, *, expect_list, json_mode, max_retries, timeout):
        return [{"correct_answer": "A", "answer_correct": True,
                 "explanation_reasonable": True, "explanation_accuracy": "good",
                 "notes": "ok"}]

    monkeypatch.setattr(vq, "llm_verdicts", short)
    qs = _sample_questions()
    out = vq.validate_questions("m", qs, batch=2, json_mode=False, max_retries=0, timeout=10)
    assert len(out) == 2
    assert out[1] == vq.missing_verdict()


def test_validate_main_flow_builds_report(monkeypatch, tmp_path):
    monkeypatch.setattr(vq, "llm_verdicts", _mock_verdicts())
    qs = _sample_questions()
    verdicts = vq.validate_questions("m", qs, batch=2, json_mode=False, max_retries=0, timeout=10)
    info = vq.parse_info_from_filename("2023年5月北美A卷.json")
    report = vq.build_report(info, qs, verdicts)

    assert report["info"]["year"] == 2023
    assert report["info"]["area"] == "北美"
    assert report["summary"]["total"] == 2
    assert report["summary"]["answer_correct"] == 2
    assert report["summary"]["by_module"]["reading_writing:1"]["total"] == 2
    assert report["summary"]["numbering_issues"] == []
