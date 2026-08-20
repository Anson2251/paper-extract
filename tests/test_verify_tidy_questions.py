"""Unit tests for scripts/validate_tidy_questions.py (pure logic, no LLM calls)."""
import json

import pytest

import validate_tidy_questions as vq


# ------------------------------------------------------------ parse_from_name

def test_parse_info_from_filename_full():
    info = vq.parse_info_from_filename("2023年5月北美A卷.json")
    assert info["year"] == 2023
    assert info["month"] == 5
    assert info["area"] == "北美"
    assert info["code"] == "A"
    assert info["source"] == "2023年5月北美A卷.json"


def test_parse_info_from_filename_validated_suffix():
    info = vq.parse_info_from_filename("2024年3月北美A卷.validated.json")
    assert (info["year"], info["month"], info["area"], info["code"]) == (
        2024, 3, "北美", "A",
    )


def test_parse_info_from_filename_no_code():
    info = vq.parse_info_from_filename("2024年10月北美卷.json")
    assert info["area"] == "北美"
    assert info["code"] == ""


def test_parse_info_from_filename_fallback():
    info = vq.parse_info_from_filename("26年6月第1套阅读+数学+答案解析.json")
    assert info["year"] == 0 and info["month"] == 0
    assert info["area"] == "" and info["code"] == ""


# --------------------------------------------------------------------- _as_bool

def test_as_bool():
    assert vq._as_bool(True) is True
    assert vq._as_bool(False) is False
    assert vq._as_bool(1) is True
    assert vq._as_bool(0) is False
    assert vq._as_bool("true") is True
    assert vq._as_bool("是") is True
    assert vq._as_bool("wrong") is False
    assert vq._as_bool("maybe") is None
    assert vq._as_bool(None) is None


# -------------------------------------------------------------- codes_equiv

def test_codes_equivalent():
    # Loose string fallback: ignores case, whitespace and . ( ) [ ] - _ /
    assert vq._codes_equivalent("A", "a")
    assert vq._codes_equivalent("B.", "b")
    assert vq._codes_equivalent("x-y", "x_y")
    assert vq._codes_equivalent("3 1/2", "3_1/2")
    assert not vq._codes_equivalent("4", "5")
    assert not vq._codes_equivalent("", "5")
    assert not vq._codes_equivalent("4", "")
    # NOT numeric equivalence: that is the LLM's job via answer_correct.
    assert not vq._codes_equivalent("3.43", "343/100")


# ----------------------------------------------------------- normalize_accuracy

def test_normalize_accuracy():
    assert vq.normalize_accuracy("good") == "good"
    assert vq.normalize_accuracy("partial") == "partial"
    assert vq.normalize_accuracy("poor") == "poor"
    assert vq.normalize_accuracy("合理") == "good"
    assert vq.normalize_accuracy("部分合理") == "partial"
    assert vq.normalize_accuracy("错误") == "poor"
    assert vq.normalize_accuracy("weird") == "unknown"
    assert vq.normalize_accuracy(None) == "unknown"


def test_missing_verdict():
    mv = vq.missing_verdict()
    assert mv["correct_answer"] == ""
    assert mv["answer_correct"] is False
    assert mv["explanation_accuracy"] == "unknown"


# ------------------------------------------------------------ normalize_verdict

def _q(answer="A"):
    return {"answer": answer}


def test_normalize_verdict_full():
    v = vq.normalize_verdict(
        {"correct_answer": "B", "answer_correct": False,
         "explanation_reasonable": False, "explanation_accuracy": "poor",
         "notes": "Wrong."},
        _q(),
    )
    assert v["correct_answer"] == "B"
    assert v["answer_correct"] is False
    assert v["explanation_accuracy"] == "poor"
    assert v["notes"] == "Wrong."


def test_normalize_verdict_nested():
    v = vq.normalize_verdict(
        {"verdict": {"correct_answer": "B", "answer_correct": True,
                     "explanation_reasonable": True, "explanation_accuracy": "good",
                     "notes": "ok"}},
        _q("A"),
    )
    assert v["answer_correct"] is True
    assert v["explanation_reasonable"] is True


def test_normalize_verdict_missing_answer_correct_falls_back_to_compare():
    v = vq.normalize_verdict(
        {"correct_answer": "B", "explanation_accuracy": "good"}, _q("A")
    )
    assert v["answer_correct"] is False
    assert v["explanation_reasonable"] is True  # good -> reasonable


def test_normalize_verdict_equivalent_free_response():
    # fallback uses the loose string comparison when the model omits the boolean
    v = vq.normalize_verdict(
        {"correct_answer": "b", "explanation_accuracy": "good"}, _q("B.")
    )
    assert v["answer_correct"] is True


def test_normalize_verdict_chinese_values():
    v = vq.normalize_verdict(
        {"correct_answer": "B", "answer_correct": "否", "explanation_accuracy": "错误"},
        _q(),
    )
    assert v["answer_correct"] is False
    assert v["explanation_accuracy"] == "poor"


def test_normalize_verdict_clears_indeterminate():
    v = vq.normalize_verdict({"correct_answer": "unknown", "explanation_accuracy": "good"}, _q())
    assert v["correct_answer"] == ""


def test_normalize_verdict_non_dict_is_missing():
    assert vq.normalize_verdict(None, _q()) == vq.missing_verdict()
    assert vq.normalize_verdict([1, 2], _q()) == vq.missing_verdict()


# ------------------------------------------------------------------- nav_label

def test_nav_label_full():
    assert vq._nav_label({"section": "reading_writing", "module": 2, "number": 15}) == (
        "Section 1 (reading_writing), Module 2, Q15"
    )
    assert vq._nav_label({"section": "math"}) == "Section 2 (math)"
    assert vq._nav_label({"number": 5}) == "Q5"
    assert vq._nav_label({}) == ""


# -------------------------------------------------------------- render / prompt

def test_render_question_with_options():
    q = {"background": "bg", "question": "Which?", "options": [
        {"code": "A", "label": "One"}, {"code": "B", "label": "Two"}]}
    out = vq.render_question(q, 1)
    assert out.startswith("QUESTION 1")
    assert "bg" in out and "Which?" in out
    assert "A. One" in out and "B. Two" in out


def test_render_question_free_response():
    q = {"question": "Solve.", "options": []}
    out = vq.render_question(q, 3)
    assert "[Free response — no options]" in out


def test_claimed_block_includes_location_and_answer():
    q = {"section": "reading_writing", "module": 1, "number": 4,
         "answer": "B", "explanations": "exp"}
    block = vq._claimed_block(q)
    assert "Claimed location: " in block and "Section 1" in block
    assert "Claimed answer: B" in block and "Claimed explanation: exp" in block


def test_single_user_prompt():
    q = {"section": "math", "module": 1, "answer": "A", "explanations": ""}
    p = vq.single_user_prompt(q, 7)
    assert "QUESTION 7" in p
    assert "correct_answer" in p


def test_batch_user_prompt_array():
    group = [{"question": "A", "options": [], "answer": "1"},
             {"question": "B", "options": [], "answer": "2"}]
    p = vq.batch_user_prompt(group, 1, json_mode=False)
    assert "QUESTION 1" in p and "QUESTION 2" in p
    assert "JSON array with one object per question" in p


def test_batch_user_prompt_json_mode_object():
    group = [{"question": "A"}]
    p = vq.batch_user_prompt(group, 1, json_mode=True)
    assert '"verdicts"' in p


# -------------------------------------------------------------- unwrap_verdicts

def test_unwrap_verdicts_list():
    data = [{"a": 1}, {"b": 2}]
    assert vq.unwrap_verdicts(data, expect_list=True) == data


def test_unwrap_verdicts_wrapped_dict():
    for key in ("verdicts", "questions", "results", "items"):
        data = {key: [{"x": 1}]}
        assert vq.unwrap_verdicts(data, expect_list=True) == [{"x": 1}]


def test_unwrap_verdicts_single():
    assert vq.unwrap_verdicts({"correct_answer": "A"}, expect_list=False) == {
        "correct_answer": "A"
    }
    assert vq.unwrap_verdicts([{"correct_answer": "A"}], expect_list=False) == {
        "correct_answer": "A"
    }
    assert vq.unwrap_verdicts("junk", expect_list=False) is None


# ------------------------------------------------------------ numbering_issues

def test_numbering_issues_duplicates_and_order():
    questions = [
        {"section": "reading_writing", "module": 1, "number": 1},
        {"section": "reading_writing", "module": 1, "number": 2},
        {"section": "reading_writing", "module": 1, "number": 2},  # dup
        {"section": "reading_writing", "module": 1, "number": 5},  # gap ok
        {"section": "reading_writing", "module": 1, "number": 4},  # out of order
        {"section": "math", "module": 2, "number": 1},
    ]
    issues = vq.numbering_issues(questions)
    assert any("duplicate number" in i for i in issues)
    assert any("order problem" in i for i in issues)


def test_numbering_issues_clean():
    questions = [
        {"section": "reading_writing", "module": 1, "number": 1},
        {"section": "reading_writing", "module": 1, "number": 2},
        {"section": "math", "module": 2, "number": 7},
    ]
    assert vq.numbering_issues(questions) == []


def test_numbering_issues_ignores_undetermined():
    assert vq.numbering_issues([{"section": "", "module": 0, "number": 0}]) == []
    assert vq.numbering_issues([]) == []


# --------------------------------------------------------------- build_report

def _verdict(correct="B", ok=False, acc="poor", reasonable=False):
    return {
        "correct_answer": correct, "answer_correct": ok,
        "explanation_reasonable": reasonable, "explanation_accuracy": acc,
        "notes": "n",
    }


def test_build_report_passes_through_navigation():
    questions = [
        {"section": "reading_writing", "module": 1, "number": 3,
         "background": "b", "question": "Q?", "options": [{"code": "A", "label": "x"}],
         "answer": "A", "explanations": "e"},
    ]
    report = vq.build_report(
        {"year": 2024, "source": "x.json"}, questions, [_verdict(ok=True, acc="good", reasonable=True)]
    )
    q = report["questions"][0]
    assert q["section"] == "reading_writing"
    assert q["module"] == 1
    assert q["number"] == 3
    assert q["correct_answer"] == "B"
    assert q["answer_correct"] is True
    s = report["summary"]
    assert s["total"] == 1 and s["answer_correct"] == 1
    assert s["explanation_reasonable"] == 1
    assert s["by_module"]["reading_writing:1"]["total"] == 1


def test_build_report_by_module_sorted():
    questions = [
        {"section": "math", "module": 1, "number": 1, "question": "a",
         "options": [], "answer": "", "explanations": ""},
        {"section": "reading_writing", "module": 2, "number": 1, "question": "b",
         "options": [], "answer": "", "explanations": ""},
    ]
    report = vq.build_report({"source": "x"}, questions, [_verdict(ok=True), _verdict()])
    keys = list(report["summary"]["by_module"].keys())
    assert keys == ["reading_writing:2", "math:1"]


def test_build_report_legacy_records_without_navigation():
    questions = [
        {"idx": 1, "question": "q", "options": [], "answer": "3", "explanations": ""}
    ]
    report = vq.build_report({"source": "x"}, questions, [vq.missing_verdict()])
    q = report["questions"][0]
    assert q["section"] == "" and q["module"] == 0 and q["number"] == 0
    assert report["summary"]["numbering_issues"] == []


# ------------------------------------------------------------------ load_tidy

def test_load_tidy_filters_invalid(tmp_path):
    p = tmp_path / "tidy.json"
    p.write_text(json.dumps([
        {"question": "good"},
        {"question": "  "},
        "not a dict",
    ]), encoding="utf-8")
    assert len(vq.load_tidy(p)) == 1


def test_load_tidy_rejects_non_list(tmp_path):
    p = tmp_path / "tidy.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        vq.load_tidy(p)


# -------------------------------------------------------------- json scanning

def test_validate_parse_json_response():
    assert vq.parse_json_response('```json\n{"x": 1}\n```') == {"x": 1}
    assert vq.parse_json_response("junk [1] end") == [1]
    assert vq.parse_json_response("") is None
    assert vq.parse_json_response("nope") is None


# ------------------------------------------------- validate via batch API

def _sample_questions():
    return [
        {"section": "reading_writing", "module": 1, "number": 1, "background": "",
         "question": "Q1", "options": [{"code": "A", "label": "a"}],
         "answer": "A", "explanations": "e"},
        {"section": "reading_writing", "module": 1, "number": 2, "background": "",
         "question": "Q2", "options": [{"code": "A", "label": "a"}],
         "answer": "A", "explanations": "e"},
    ]


def test_validate_questions_uses_batch_api(monkeypatch):
    captured = {}
    monkeypatch.setattr(vq.openai_batch, "is_openai", lambda m: True)

    def fake_batch(model, tasks, **kw):
        captured["kw"] = kw
        captured["n_tasks"] = len(tasks)
        verdict = {"correct_answer": "A", "answer_correct": True,
                   "explanation_reasonable": True, "explanation_accuracy": "good",
                   "notes": "ok"}
        out = {}
        for cid, messages, _p in tasks:
            prompt = messages[-1]["content"]
            # batch=1 prompts ask for a single object; batch>1 ask for an array.
            if "JSON array with one object per question" in prompt:
                out[cid] = [verdict, verdict]
            else:
                out[cid] = verdict
        return out

    monkeypatch.setattr(vq.openai_batch, "batch_completions", fake_batch)
    qs = _sample_questions()

    single = vq.validate_questions("m", qs, batch=1, json_mode=False,
                                   max_retries=0, timeout=10, batch_size=2)
    assert len(single) == 2 and all(v["answer_correct"] is True for v in single)
    assert captured["n_tasks"] == 2
    assert captured["kw"]["batch_size"] == 2

    batch = vq.validate_questions("m", qs, batch=2, json_mode=False,
                                  max_retries=0, timeout=10, batch_size=2)
    assert len(batch) == 2 and all(v["answer_correct"] is True for v in batch)
    assert captured["n_tasks"] == 1  # both questions in one call -> one batch task


def test_validate_questions_batch_size_one_uses_sync(monkeypatch):
    monkeypatch.setattr(vq.openai_batch, "is_openai", lambda m: True)
    monkeypatch.setattr(
        vq.openai_batch, "batch_completions",
        lambda *a, **k: pytest.fail("batch should not be used when batch_size == 1"),
    )

    def fake_verdicts(model, messages, *, expect_list, json_mode, max_retries, timeout):
        verdict = {"correct_answer": "A", "answer_correct": True,
                   "explanation_reasonable": True, "explanation_accuracy": "good",
                   "notes": "ok"}
        return verdict if not expect_list else [verdict, verdict]

    monkeypatch.setattr(vq, "llm_verdicts", fake_verdicts)
    qs = _sample_questions()
    out = vq.validate_questions("m", qs, batch=1, json_mode=False,
                                max_retries=0, timeout=10, batch_size=1)
    assert len(out) == 2 and all(v["answer_correct"] is True for v in out)


# -------------------------------------------------------- validate multimodal

def test_validate_questions_multimodal_attaches_images(tmp_path, monkeypatch):
    fig = tmp_path / "fig.png"
    fig.write_bytes(b"bytes")

    class FakeIndex:
        def __init__(self, roots):
            pass

        def path(self, name):
            return fig

    monkeypatch.setattr(vq.paper_images, "ImageIndex", FakeIndex)
    captured = []

    def fake_verdicts(model, messages, *, expect_list, json_mode, max_retries, timeout):
        captured.append(messages)
        verdict = {"correct_answer": "A", "answer_correct": True,
                   "explanation_reasonable": True, "explanation_accuracy": "good",
                   "notes": "ok"}
        return verdict if not expect_list else [verdict]

    monkeypatch.setattr(vq, "llm_verdicts", fake_verdicts)
    qs = [{"section": "math", "module": 1, "number": 1, "background": "",
           "question": "What is in the figure? [image: images/fig.png]",
           "options": [], "answer": "3"}]

    out = vq.validate_questions("m", qs, batch=1, json_mode=False,
                                max_retries=0, timeout=10, multimodal=True)
    assert out[0]["answer_correct"] is True
    user = captured[0][-1]["content"]
    assert isinstance(user, list)
    assert user[-1]["type"] == "image_url"

    # multimodal off -> plain string, no attachment
    out2 = vq.validate_questions("m", qs, batch=1, json_mode=False,
                                 max_retries=0, timeout=10, multimodal=False)
    assert isinstance(captured[1][-1]["content"], str)
