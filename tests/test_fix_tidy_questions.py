"""Unit + integration tests for scripts/fix_tidy_questions.py (no LLM calls)."""
import json
from types import SimpleNamespace

import pytest

import fix_tidy_questions as fx


# -------------------------------------------------------------- number planning

def test_numbering_has_issue():
    assert fx._numbering_has_issue([1, 2, 2]) is True       # duplicate
    assert fx._numbering_has_issue([1, 3, 2]) is True        # out of order
    assert fx._numbering_has_issue([1, 2, 4]) is False       # gap is fine
    assert fx._numbering_has_issue([]) is False
    assert fx._numbering_has_issue([0, 0]) is False          # undetermined


def _qs():
    return [
        {"section": "reading_writing", "module": 1, "number": 1, "question": "a"},
        {"section": "reading_writing", "module": 1, "number": 3, "question": "b"},
        {"section": "reading_writing", "module": 1, "number": 3, "question": "c"},  # dup
        {"section": "math", "module": 2, "number": 1, "question": "d"},
    ]


def test_plan_renumber_fixes_only_problematic_module():
    fixes = fx.plan_renumber(_qs())
    assert set(fixes.keys()) == {("reading_writing", 1)}
    mapping = fixes[("reading_writing", 1)]
    # renumber indices 0..2 to 1..3 in file order
    assert mapping == {0: 1, 1: 2, 2: 3}


def test_plan_renumber_leaves_clean_modules():
    qs = [{"section": "math", "module": 1, "number": 1, "question": "a"},
          {"section": "math", "module": 1, "number": 2, "question": "b"}]
    assert fx.plan_renumber(qs) == {}


# --------------------------------------------------------------- answer helpers

def _mc(code="A"):
    return {"answer": code, "options": [{"code": "A", "label": "x"},
                                        {"code": "B", "label": "y"}]}


def test_answer_fix_mc_wrong():
    new, unresolved = fx.answer_fix(_mc("A"), {"answer_correct": False, "correct_answer": "B"})
    assert new == "B" and unresolved == []


def test_answer_fix_mc_correct_unchanged():
    new, unresolved = fx.answer_fix(_mc("A"), {"answer_correct": True})
    assert new is None and unresolved == []


def test_answer_fix_mc_not_in_options():
    new, unresolved = fx.answer_fix(_mc("A"), {"answer_correct": False, "correct_answer": "E"})
    assert new is None and len(unresolved) == 1


def test_answer_fix_mc_undeterminable():
    new, unresolved = fx.answer_fix(_mc("A"), {"answer_correct": False, "correct_answer": ""})
    assert new is None and len(unresolved) == 1


def test_answer_fix_free_response():
    q = {"answer": "3.43", "options": []}
    new, unresolved = fx.answer_fix(q, {"answer_correct": False, "correct_answer": "4"})
    assert new == "4" and unresolved == []


def test_should_rewrite_explanation():
    tidy = _mc("A")
    assert fx.should_rewrite_explanation(
        tidy, {"explanation_reasonable": False, "correct_answer": "B"}) is True
    assert fx.should_rewrite_explanation(
        tidy, {"explanation_reasonable": True}) is False
    assert fx.should_rewrite_explanation(
        tidy, {"explanation_reasonable": False, "correct_answer": ""}) is False
    assert fx.should_rewrite_explanation(tidy, {}) is False  # default reasonable


def test_should_rewrite_explanation_when_answer_corrected():
    # Answer corrected (A->B): the old explanation argues for A, so it must be
    # rewritten even when the explanation itself was judged "reasonable".
    tidy = _mc("A")  # claims A, options A/B
    assert fx.should_rewrite_explanation(
        tidy, {"answer_correct": False, "correct_answer": "B",
               "explanation_reasonable": True}) is True
    # a correction-driven rewrite still needs a validated correct answer
    assert fx.should_rewrite_explanation(
        tidy, {"answer_correct": False, "correct_answer": "",
               "explanation_reasonable": True}) is False
    # no correction + reasonable explanation -> no rewrite
    assert fx.should_rewrite_explanation(
        tidy, {"answer_correct": True, "correct_answer": "A",
               "explanation_reasonable": True}) is False


# ---------------------------------------------------------------- fix_question

def test_fix_question_patches_answer_and_explanation():
    tidy = _mc("A")
    tidy.update({"explanations": "old", "section": "reading_writing", "module": 1, "number": 1})
    report = {"answer_correct": False, "correct_answer": "B",
              "explanation_reasonable": False}
    q, changes = fx.fix_question(tidy, report, rewrite=True, new_explanation="new expl")
    assert q["answer"] == "B"
    assert q["explanations"] == "new expl"
    assert changes["answer_fixed"] == {"from": "A", "to": "B"}
    assert changes["explanation_rewritten"] is True
    assert "unresolved" not in changes


def test_fix_question_keeps_unchanged_question():
    tidy = _mc("A")
    tidy.update({"explanations": "ok", "section": "reading_writing", "module": 1, "number": 2})
    report = {"answer_correct": True, "correct_answer": "A", "explanation_reasonable": True}
    q, changes = fx.fix_question(tidy, report, rewrite=True, new_explanation=None)
    assert q["answer"] == "A" and q["explanations"] == "ok"
    assert changes["answer_fixed"] is None
    assert changes["explanation_rewritten"] is False
    assert "unresolved" not in changes


def test_fix_question_answers_only_does_not_rewrite():
    tidy = _mc("A")
    tidy.update({"explanations": "old"})
    report = {"answer_correct": False, "correct_answer": "B", "explanation_reasonable": False}
    q, changes = fx.fix_question(tidy, report, rewrite=False, new_explanation=None)
    assert q["answer"] == "B"          # answer still fixed in answers-only mode
    assert q["explanations"] == "old"  # but explanation left alone
    assert changes["explanation_rewritten"] is False


def test_fix_question_rewrite_failed_reports_unresolved():
    tidy = _mc("A")
    report = {"answer_correct": True, "correct_answer": "A", "explanation_reasonable": False}
    q, changes = fx.fix_question(tidy, report, rewrite=True, new_explanation=None)
    assert "unresolved" in changes
    assert changes["explanation_rewritten"] is False


def test_explain_user_prompt_uses_validated_answer():
    tidy = {"background": "bg", "question": "Q?", "options": [{"code": "A", "label": "a"}]}
    report = {"correct_answer": "B", "notes": "reviewer hint"}
    p = fx._explain_user_prompt(tidy, report, 5)
    assert "QUESTION 5" in p
    assert "validated CORRECT ANSWER: B" in p
    assert "reviewer hint" in p


# ------------------------------------------------------------- fix_one (e2e)

def _args(tmp_path, answers_only, batch=1, renumber=False):
    return SimpleNamespace(
        output_dir=tmp_path / "fixed", answers_only=answers_only,
        batch=batch, model="fake", json_mode=False, max_retries=0, timeout=10,
        batch_size=1, multimodal=False, renumber=renumber,
    )


def _tidy_questions():
    return [
        {"background": "", "question": "q1",
         "options": [{"code": "A", "label": "a"}, {"code": "B", "label": "b"}],
         "answer": "A", "explanations": "old1",
         "section": "reading_writing", "module": 1, "number": 1},
        {"background": "", "question": "q2",
         "options": [{"code": "A", "label": "a"}],
         "answer": "A", "explanations": "ok2",
         "section": "reading_writing", "module": 1, "number": 3},
        {"background": "", "question": "q3",
         "options": [{"code": "A", "label": "a"}, {"code": "B", "label": "b"}],
         "answer": "A", "explanations": "old3",
         "section": "reading_writing", "module": 1, "number": 3},  # dup number
        # q4: free-response wrong answer (math module clean -> no renumber)
        {"background": "", "question": "q4",
         "options": [],
         "answer": "5", "explanations": "ok4",
         "section": "math", "module": 1, "number": 1},
    ]


def _report_questions():
    return [
        # q1: wrong answer + poor explanation
        {"correct_answer": "B", "answer_correct": False,
         "explanation_reasonable": False, "explanation_accuracy": "poor", "notes": "hint1",
         "answer": "A", "explanations": "old1",
         "section": "reading_writing", "module": 1, "number": 1},
        # q2: fine
        {"correct_answer": "A", "answer_correct": True,
         "explanation_reasonable": True, "explanation_accuracy": "good", "notes": "",
         "answer": "A", "explanations": "ok2",
         "section": "reading_writing", "module": 1, "number": 3},
        # q3: answer fine, explanation poor (rewritten)
        {"correct_answer": "A", "answer_correct": True,
         "explanation_reasonable": False, "explanation_accuracy": "poor", "notes": "hint3",
         "answer": "A", "explanations": "old3",
         "section": "reading_writing", "module": 1, "number": 3},
        # q4: free-response wrong answer (math module clean -> no renumber)
        {"correct_answer": "6", "answer_correct": False,
         "explanation_reasonable": True, "explanation_accuracy": "good", "notes": "",
         "answer": "5", "explanations": "ok4",
         "section": "math", "module": 1, "number": 1},
    ]


def _write(tmp_path):
    tidy_p = tmp_path / "paper.json"
    tidy_p.write_text(json.dumps(_tidy_questions()), encoding="utf-8")
    report_p = tmp_path / "report.json"
    report_p.write_text(json.dumps({"info": {"source": "paper.json"},
                                    "questions": _report_questions(),
                                    "summary": {}}), encoding="utf-8")
    return tidy_p, report_p


def test_fix_one_answers_only(tmp_path):
    tidy_p, report_p = _write(tmp_path)
    rc = fx.fix_one("fake", tidy_p, report_p, _args(tmp_path, answers_only=True))
    assert rc == 0

    fixed = json.loads((tmp_path / "fixed" / "paper.fixed.json").read_text(encoding="utf-8"))
    # q1 answer corrected, explanation untouched (answers-only)
    assert fixed[0]["answer"] == "B" and fixed[0]["explanations"] == "old1"
    # q2 unchanged
    assert fixed[1]["answer"] == "A" and fixed[1]["explanations"] == "ok2"
    # q3 answer stays, explanation untouched
    assert fixed[2]["answer"] == "A" and fixed[2]["explanations"] == "old3"
    # q4 free-response corrected
    assert fixed[3]["answer"] == "6"
    # numbers are NOT renumbered by default: the tidy numbers (1, 3, 3) survive
    assert [q["number"] for q in fixed] == [1, 3, 3, 1]

    changes = json.loads((tmp_path / "fixed" / "paper.changes.json").read_text(encoding="utf-8"))
    assert changes["summary"]["answers_fixed"] == 2
    assert changes["summary"]["explanations_rewritten"] == 0
    assert changes["summary"]["numbers_fixed"] == 0
    assert changes["changes"][0]["answer_fixed"] == {"from": "A", "to": "B"}


def test_fix_one_renumber_opt_in(tmp_path):
    """With --renumber, modules with duplicate/out-of-order numbers are renumbered "
    to 1..N in file order (the legacy behaviour)."""
    tidy_p, report_p = _write(tmp_path)
    rc = fx.fix_one("fake", tidy_p, report_p, _args(tmp_path, answers_only=True, renumber=True))
    assert rc == 0
    fixed = json.loads((tmp_path / "fixed" / "paper.fixed.json").read_text(encoding="utf-8"))
    assert [q["number"] for q in fixed[:3]] == [1, 2, 3]
    assert fixed[3]["number"] == 1  # math module untouched
    changes = json.loads((tmp_path / "fixed" / "paper.changes.json").read_text(encoding="utf-8"))
    assert changes["summary"]["numbers_fixed"] == 3


def test_fix_one_rewrites_explanations(tmp_path, monkeypatch):
    tidy_p, report_p = _write(tmp_path)
    monkeypatch.setattr(
        fx, "llm_explanations",
        lambda model, items, **kw: {0: "new expl 1", 2: "new expl 3", 3: "new expl 4"},
    )
    rc = fx.fix_one("fake", tidy_p, report_p, _args(tmp_path, answers_only=False))
    assert rc == 0
    fixed = json.loads((tmp_path / "fixed" / "paper.fixed.json").read_text(encoding="utf-8"))
    assert fixed[0]["answer"] == "B" and fixed[0]["explanations"] == "new expl 1"
    assert fixed[2]["explanations"] == "new expl 3"
    assert fixed[1]["explanations"] == "ok2"
    # q4: answer corrected 5->6; its explanation supported the old answer and is
    # rewritten even though the validation judged it reasonable
    assert fixed[3]["answer"] == "6" and fixed[3]["explanations"] == "new expl 4"
    changes = json.loads((tmp_path / "fixed" / "paper.changes.json").read_text(encoding="utf-8"))
    assert changes["summary"]["explanations_rewritten"] == 3
    assert changes["changes"][3]["answer_fixed"] == {"from": "5", "to": "6"}
    assert changes["changes"][3]["explanation_rewritten"] is True


def test_fix_one_mismatched_length(tmp_path):
    tidy_p, report_p = _write(tmp_path)
    report_p.write_text(json.dumps({"questions": _report_questions()[:2], "summary": {}}),
                        encoding="utf-8")
    rc = fx.fix_one("f", tidy_p, report_p, _args(tmp_path, answers_only=True))
    assert rc == 1


def test_fix_one_validates_report_shape(tmp_path):
    tidy_p, _ = _write(tmp_path)
    report_p = tmp_path / "report.json"
    report_p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert fx.fix_one("f", tidy_p, report_p, _args(tmp_path, answers_only=True)) == 1


def test_base_stem_strips_suffixes():
    assert fx.base_stem("demo-paper") == "demo-paper"
    assert fx.base_stem("demo-paper.validated") == "demo-paper"
    assert fx.base_stem("demo-paper.verified") == "demo-paper"
    assert fx.base_stem("demo-paper.fixed") == "demo-paper"
    assert fx.base_stem("a.fixed.b") == "a.fixed.b"  # not a trailing suffix


def test_is_validated_report():
    q = {"question": "Q", "correct_answer": "B", "answer_correct": False}
    assert fx.is_validated_report({"questions": [q]}) is True
    assert fx.is_validated_report([q]) is True
    assert fx.is_validated_report({"questions": [{"question": "Q"}]}) is False
    assert fx.is_validated_report([{"question": "Q"}]) is False
    assert fx.is_validated_report([]) is False
    assert fx.is_validated_report({"a": 1}) is False


def test_fix_one_self_contained_validated_input(tmp_path):
    """Passing an already-validated report as the only input works (no separate
    tidy or validated file needed) and output stems don't double the suffix."""
    vp = tmp_path / "paper.json"
    vp.write_text(json.dumps({"info": {"source": "paper.json"},
                              "questions": _report_questions(), "summary": {}}),
                  encoding="utf-8")
    rc = fx.fix_one("fake", vp, None, _args(tmp_path, answers_only=True))
    assert rc == 0
    fixed = json.loads((tmp_path / "fixed" / "paper.fixed.json").read_text(encoding="utf-8"))
    # same fixes as the paired mode: q1 answer A->B, q4 answer 5->6; no renumbering
    assert fixed[0]["answer"] == "B"
    assert [q["number"] for q in fixed] == [1, 3, 3, 1]
    assert fixed[3]["answer"] == "6"


# -------------------------------------------------------------- json scanning

def test_parse_json_response():
    assert fx.parse_json_response('```json\n{"explanation": "x"}\n```') == {"explanation": "x"}
    assert fx.parse_json_response("junk [1,2] end") == [1, 2]
    assert fx.parse_json_response("") is None
    assert fx.parse_json_response("nope") is None


# ------------------------------------------------- explanations via batch API

def test_llm_explanations_uses_batch_api(monkeypatch):
    monkeypatch.setattr(fx.openai_batch, "is_openai", lambda m: True)

    def fake_batch(model, tasks, **kw):
        return {cid: {"explanation": f"new {i}"} for i, (cid, _m, _p) in enumerate(tasks, 1)}

    monkeypatch.setattr(fx.openai_batch, "batch_completions", fake_batch)
    items = [
        (0, {"question": "q1"}, {"correct_answer": "B"}),
        (2, {"question": "q2"}, {"correct_answer": "A"}),
    ]
    out = fx.llm_explanations("m", items, batch=1, json_mode=False,
                              max_retries=0, timeout=10, batch_size=2)
    # fake_batch assigns new 0 / new 1 by task position
    assert out == {0: "new 1", 2: "new 2"}


def test_llm_explanations_batch_size_one_uses_sync(monkeypatch):
    monkeypatch.setattr(fx.openai_batch, "is_openai", lambda m: True)
    monkeypatch.setattr(
        fx.openai_batch, "batch_completions",
        lambda *a, **k: pytest.fail("batch should not be used when batch_size == 1"),
    )
    monkeypatch.setattr(
        fx, "_call",
        lambda model, messages, **kw: {"explanation": "sync expl"},
    )
    items = [(0, {"question": "q1"}, {"correct_answer": "B"})]
    out = fx.llm_explanations("m", items, batch=1, json_mode=False,
                              max_retries=0, timeout=10, batch_size=1)
    assert out == {0: "sync expl"}


# ------------------------------------------------- explanations multimodal

def test_llm_explanations_multimodal_attaches_images(tmp_path, monkeypatch):
    fig = tmp_path / "fig.png"
    fig.write_bytes(b"bytes")

    class FakeIndex:
        def __init__(self, roots):
            pass

        def path(self, name):
            return fig

    monkeypatch.setattr(fx.paper_images, "ImageIndex", FakeIndex)
    captured = {}

    def fake_call(model, messages, **kw):
        captured["messages"] = messages
        return {"explanation": "new"}

    monkeypatch.setattr(fx, "_call", fake_call)
    items = [(0, {"question": "Q? [image: images/fig.png]"}, {"correct_answer": "B"})]
    out = fx.llm_explanations("m", items, batch=1, json_mode=False,
                              max_retries=0, timeout=10, multimodal=True)
    assert out == {0: "new"}
    user = captured["messages"][-1]["content"]
    assert isinstance(user, list)
    assert user[-1]["type"] == "image_url"
