"""Unit tests for scripts/tidy_questions.py (pure logic, no LLM calls)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tidy_questions as tq


def test_normalize_code():
    assert tq.normalize_code("A") == "A"
    assert tq.normalize_code("A.") == "A"
    assert tq.normalize_code("A)") == "A"
    assert tq.normalize_code("(A)") == "A"
    assert tq.normalize_code("a") == "A"
    assert tq.normalize_code("E") == "E"
    assert tq.normalize_code("??") == "??"


def test_normalize_section():
    assert tq.normalize_section("reading_writing") == "reading_writing"
    assert tq.normalize_section("Math") == "math"
    assert tq.normalize_section("rw") == "reading_writing"
    assert tq.normalize_section("english") == "reading_writing"
    assert tq.normalize_section("英语") == "reading_writing"
    assert tq.normalize_section("阅读与写作") == "reading_writing"
    assert tq.normalize_section("mathematics") == "math"
    assert tq.normalize_section("数学") == "math"
    assert tq.normalize_section("bogus") == ""
    assert tq.normalize_section(None) == ""


def test_normalize_module():
    assert tq.normalize_module("M1") == 1
    assert tq.normalize_module("2") == 2
    assert tq.normalize_module("m2") == 2
    assert tq.normalize_module(1) == 1
    assert tq.normalize_module(2.0) == 2
    assert tq.normalize_module(True) == 0
    assert tq.normalize_module("M3") == 0
    assert tq.normalize_module(3) == 0
    assert tq.normalize_module("module1") == 0


def test_normalize_number():
    assert tq.normalize_number("15") == 15
    assert tq.normalize_number("#12") == 12
    assert tq.normalize_number(3.0) == 3
    assert tq.normalize_number(0) == 0
    assert tq.normalize_number("Q3") == 0
    assert tq.normalize_number(-4) == 0
    assert tq.normalize_number(True) == 0


def test_normalize_questions_full_record():
    items = [
        {
            "background": "bg",
            "question": "What is 2+2?",
            "options": [{"code": "A.", "label": "Alpha"}, {"code": "b)", "label": "Beta"}],
            "answer": "beta",
            "explanations": "Because 2+2=4.",
            "section": "Math",
            "module": "M2",
            "number": "15",
        }
    ]
    questions, issues = tq.normalize_questions(items)
    assert issues == []
    q = questions[0]
    assert q["options"] == [{"code": "A", "label": "Alpha"}, {"code": "B", "label": "Beta"}]
    assert q["answer"] == "B"
    assert q["section"] == "math"
    assert q["module"] == 2
    assert q["number"] == 15


def test_normalize_questions_free_response():
    items = [
        {
            "question": "Solve for x.",
            "options": [],
            "answer": "3.43",
            "explanations": "",
        }
    ]
    questions, issues = tq.normalize_questions(items)
    assert issues == []
    assert questions[0]["options"] == []
    assert questions[0]["answer"] == "3.43"


def test_normalize_questions_reports_bad_items():
    items = [
        "not an object",
        {"question": "", "options": []},
        {"question": "Q", "options": [{"code": "A", "label": "a"}], "answer": "Z"},
    ]
    questions, issues = tq.normalize_questions(items)
    assert len(questions) == 1
    joined = "\n".join(issues)
    assert "item 0: not an object" in joined
    assert "item 1: missing question text" in joined
    assert "answer 'Z' not in option codes" in joined


def test_normalize_questions_filters_bad_options():
    items = [
        {
            "question": "Q",
            "options": [
                {"code": "A", "label": "ok"},
                {"code": "", "label": "no code"},
                {"code": "B", "label": ""},
                "junk",
            ],
            "answer": "A",
        }
    ]
    questions, _ = tq.normalize_questions(items)
    assert questions[0]["options"] == [{"code": "A", "label": "ok"}]


def test_normalize_questions_preserves_unknown_navigation():
    items = [{"question": "Q", "options": [], "answer": "5"}]
    questions, _ = tq.normalize_questions(items)
    assert questions[0]["section"] == ""
    assert questions[0]["module"] == 0
    assert questions[0]["number"] == 0


def test_dedupe_drops_duplicate_question_text():
    questions = [
        {"question": "Same question"},
        {"question": "  same   question  "},
        {"question": "Different"},
    ]
    out, skipped = tq.dedupe(questions)
    assert skipped == 1
    assert len(out) == 2


def test_scan_json_extracts_balanced_value():
    assert tq._scan_json('prefix {"a": 1} suffix') == '{"a": 1}'


def test_scan_json_handles_escaped_quotes():
    raw = '[{"text": "a \\"quote\\" inside"}]'
    assert tq._scan_json("junk " + raw) == raw


def test_scan_json_returns_none_when_no_value():
    assert tq._scan_json("just text") is None


def test_parse_json_response_empty():
    assert tq.parse_json_response("") is None


def test_parse_json_response_fenced():
    raw = '```json\n[{"a": 1}]\n```'
    assert tq.parse_json_response(raw) == [{"a": 1}]


def test_parse_json_response_plain_object():
    assert tq.parse_json_response('{"answer": "A"}') == {"answer": "A"}


def test_parse_json_response_recovers_from_junk():
    assert tq.parse_json_response('here you go: [1, 2, 3] thanks!') == [1, 2, 3]


def test_parse_json_response_invalid_returns_none():
    assert tq.parse_json_response("not json at all") is None


def test_parse_page_response_array_fallback():
    raw = '[{"question": "Q", "options": [], "answer": "A", "explanations": "e", "section": "math", "module": 1, "number": 1}]'
    parsed = tq._parse_page_response(raw)
    assert parsed["updated_previous"] is None
    assert len(parsed["questions"]) == 1


def test_parse_page_response_object_with_updated_previous():
    raw = '{"updated_previous": {"question": "Qprev", "options": [], "answer": "A", "explanations": "e", "section": "math", "module": 1, "number": 1}, "questions": [{"question": "Q2", "options": [], "answer": "B", "explanations": "e2", "section": "math", "module": 1, "number": 2}]}'
    parsed = tq._parse_page_response(raw)
    assert parsed["updated_previous"]["question"] == "Qprev"
    assert parsed["questions"][0]["question"] == "Q2"


def test_parse_page_response_null_updated():
    raw = '{"updated_previous": null, "questions": []}'
    parsed = tq._parse_page_response(raw)
    assert parsed["updated_previous"] is None
    assert parsed["questions"] == []


def test_parse_index_range_valid():
    assert tq.parse_index_range("2-5", 10, "page") == (2, 5)
    assert tq.parse_index_range(" 0 - 3 ", 10, "page") == (0, 3)


def test_parse_index_range_invalid_format():
    with pytest.raises(Exception):
        tq.parse_index_range("abc", 10, "page")
    with pytest.raises(Exception):
        tq.parse_index_range("1,2", 10, "page")


def test_parse_index_range_bad_bounds():
    with pytest.raises(Exception):
        tq.parse_index_range("5-2", 10, "page")
    with pytest.raises(Exception):
        tq.parse_index_range("0-10", 10, "page")


@pytest.mark.parametrize(
    "phrase",
    [
        "TIDYING, not grading",
        "transcribe it verbatim",
        "No official explanation provided in the answer key.",
        "never change a letter or value because you believe the key is mistaken",
        'section',
        '"number"',
    ],
)
def test_tidy_prompt_reinforces_transcription_role(phrase):
    assert phrase.lower() in tq.SYSTEM_PROMPT.lower()


def test_pdf_to_images_mocked(monkeypatch, tmp_path):
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(tq, "pdf_to_images", lambda p, dpi=150: [b"img1", b"img2"])
    images = tq.pdf_to_images(fake_pdf, dpi=150)
    assert images == [b"img1", b"img2"]


def test_process_pair_carries_previous_question(tmp_path, monkeypatch):
    import litellm
    paper = tmp_path / "paper.pdf"
    answer = tmp_path / "answer.pdf"
    paper.write_bytes(b"%PDF")
    answer.write_bytes(b"%PDF")

    monkeypatch.setattr(tq, "pdf_to_images", lambda p, dpi=150: [b"page0", b"page1"] if "paper" in str(p) else [b"ans"])

    calls = []

    def fake_extract(model, content, *, json_mode, max_retries, timeout):
        calls.append(content)
        if len(calls) == 1:
            return {"updated_previous": None, "questions": [{"question": "Q1 split part", "options": [{"code": "A", "label": "x"}], "answer": "A", "explanations": "e", "section": "reading_writing", "module": 1, "number": 1}]}
        else:
            # second page: content should contain previous question
            assert any("Q1 split part" in str(part) for part in content if isinstance(part, dict) and part.get("type") == "text")
            return {"updated_previous": {"question": "Q1 complete", "options": [{"code": "A", "label": "x"}], "answer": "A", "explanations": "e", "section": "reading_writing", "module": 1, "number": 1}, "questions": [{"question": "Q2", "options": [{"code": "A", "label": "y"}], "answer": "A", "explanations": "e", "section": "reading_writing", "module": 1, "number": 2}]}

    monkeypatch.setattr(tq, "extract_page", fake_extract)
    out_dir = tmp_path / "tidy"
    args = SimpleNamespace(pages=None, output_dir=out_dir, model="fake", json_mode=False, max_retries=1, timeout=30, batch_size=1, dpi=150)
    pages, n_q, failures = tq.process_pair(args, paper, answer, 1, 1)
    assert failures == 0
    assert n_q == 2
    data = json.loads((out_dir / "paper.json").read_text())
    assert data[0]["question"] == "Q1 complete"
    assert data[1]["question"] == "Q2"
