"""Unit tests for scripts/finalize_paper.py (no network, mocked image index)."""
import json

import finalize_paper as fp


def test_finalize_bundles_meta_pdfs_images_and_rewrites(tmp_path, monkeypatch):
    questions = [
        {
            "background": "B [image: images/aaa.jpg]",
            "question": "Q? [image: images/aaa.jpg]",
            "options": [{"code": "A", "label": "opt [image: images/bbb.jpg]"}],
            "answer": "A",
            "explanations": "e",
            "section": "math", "module": 1, "number": 1,
        }
    ]
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

    paper_pdf = tmp_path / "paper.pdf"
    answer_pdf = tmp_path / "answer.pdf"
    paper_pdf.write_bytes(b"paper-pdf-bytes")
    answer_pdf.write_bytes(b"answer-pdf-bytes")

    a = tmp_path / "aaa.jpg"
    b = tmp_path / "bbb.jpg"
    a.write_bytes(b"IMG-A")
    b.write_bytes(b"IMG-B")

    class FakeIndex:
        def __init__(self, roots):
            pass

        def path(self, name):
            return a if name == "aaa.jpg" else (b if name == "bbb.jpg" else None)

    monkeypatch.setattr(fp.paper_images, "ImageIndex", FakeIndex)

    out = tmp_path / "bundle"
    rc = fp.main([
        "--year", "2024", "--month", "10", "--area", "Demo", "--code", "X",
        "--paper-pdf", str(paper_pdf), "--answer-pdf", str(answer_pdf),
        "--questions", str(qp), "--output", str(out),
    ])
    assert rc == 0

    data = json.loads((out / "paper.json").read_text(encoding="utf-8"))
    assert data["meta"]["year"] == 2024
    assert data["meta"]["month"] == 10
    assert data["meta"]["area"] == "Demo"
    assert data["meta"]["code"] == "X"
    # curated_at is auto-set even when --curated-at is not passed
    assert data["meta"]["curated_at"]
    assert "organizer" not in data["meta"]
    assert data["source_pdfs"]["paper"]["sha256"] == fp.sha256_file(paper_pdf)
    assert data["source_pdfs"]["answer_key"]["sha256"] == fp.sha256_file(answer_pdf)
    assert data["source_pdfs"]["paper"]["filename"] == "paper.pdf"

    # image refs rewritten to the bundle's imgs/
    q = data["questions"][0]
    assert q["background"] == "B [image: imgs/aaa.jpg]"
    assert q["question"] == "Q? [image: imgs/aaa.jpg]"
    assert q["options"][0]["label"] == "opt [image: imgs/bbb.jpg]"

    # images copied under imgs/, content-addressed by name
    assert (out / "imgs" / "aaa.jpg").read_bytes() == b"IMG-A"
    assert (out / "imgs" / "bbb.jpg").read_bytes() == b"IMG-B"


def test_finalize_merges_supplement_into_meta(tmp_path, monkeypatch):
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps([{
        "background": "", "question": "Q", "options": [], "answer": "1",
        "explanations": "", "section": "math", "module": 1, "number": 1,
    }]), encoding="utf-8")
    paper_pdf = tmp_path / "paper.pdf"
    answer_pdf = tmp_path / "answer.pdf"
    paper_pdf.write_bytes(b"p")
    answer_pdf.write_bytes(b"a")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "completeness": {"expected_total": 98, "extracted": 69,
                          "recovered": 24, "missing": 5,
                          "per_module": {}},
        "questions": [],
    }), encoding="utf-8")
    monkeypatch.setattr(fp.paper_images, "ImageIndex",
                        type("I", (), {"__init__": lambda self, r: None,
                                        "path": lambda self, n: None}))

    out = tmp_path / "bundle"
    rc = fp.main([
        "--year", "2024", "--month", "10", "--area", "Demo", "--code", "X",
        "--paper-pdf", str(paper_pdf), "--answer-pdf", str(answer_pdf),
        "--questions", str(qp), "--output", str(out),
        "--supplement-report", str(report),
    ])
    assert rc == 0
    data = json.loads((out / "paper.json").read_text(encoding="utf-8"))
    assert data["meta"]["supplement"] == {
        "expected": 98, "extracted": 69, "recovered": 24, "missing": 5,
    }
    assert data["completeness"]["expected_total"] == 98


def test_finalize_warns_on_missing_image(tmp_path, monkeypatch, capsys):
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps([{
        "background": "", "question": "Q [image: images/nope.jpg]",
        "options": [], "answer": "1", "explanations": "",
        "section": "math", "module": 1, "number": 1,
    }]), encoding="utf-8")
    paper_pdf = tmp_path / "paper.pdf"
    answer_pdf = tmp_path / "answer.pdf"
    paper_pdf.write_bytes(b"p")
    answer_pdf.write_bytes(b"a")

    class EmptyIndex:
        def __init__(self, roots):
            pass

        def path(self, name):
            return None

    monkeypatch.setattr(fp.paper_images, "ImageIndex", EmptyIndex)
    out = tmp_path / "bundle"

    rc = fp.main([
        "--year", "2024", "--month", "10", "--area", "Demo", "--code", "X",
        "--paper-pdf", str(paper_pdf), "--answer-pdf", str(answer_pdf),
        "--questions", str(qp), "--output", str(out),
    ])
    assert rc == 0
    # missing image left as-is, not rewritten
    data = json.loads((out / "paper.json").read_text(encoding="utf-8"))
    assert data["questions"][0]["question"] == "Q [image: images/nope.jpg]"
    # reported on stderr
    assert "not found" in capsys.readouterr().err


def test_finalize_errors_on_missing_inputs(tmp_path, capsys):
    rc = fp.main([
        "--year", "2024", "--month", "10", "--area", "Demo", "--code", "X",
        "--paper-pdf", str(tmp_path / "no.pdf"), "--answer-pdf", str(tmp_path / "no2.pdf"),
        "--questions", str(tmp_path / "no.json"), "--output", str(tmp_path / "bundle"),
    ])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_finalize_records_organizer_and_curated_at(tmp_path, monkeypatch):
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps([{
        "background": "", "question": "Q", "options": [], "answer": "1",
        "explanations": "", "section": "math", "module": 1, "number": 1,
        "key_answer": "1", "page": 60,
    }]), encoding="utf-8")
    paper_pdf = tmp_path / "paper.pdf"
    answer_pdf = tmp_path / "answer.pdf"
    paper_pdf.write_bytes(b"p")
    answer_pdf.write_bytes(b"a")
    monkeypatch.setattr(fp.paper_images, "ImageIndex",
                        type("I", (), {"__init__": lambda self, r: None,
                                        "path": lambda self, n: None}))

    out = tmp_path / "bundle"
    rc = fp.main([
        "--year", "2024", "--month", "10", "--area", "Demo", "--code", "X",
        "--paper-pdf", str(paper_pdf), "--answer-pdf", str(answer_pdf),
        "--questions", str(qp), "--output", str(out),
        "--organizer", "anson", "--curated-at", "2026-08-21",
    ])
    assert rc == 0
    data = json.loads((out / "paper.json").read_text(encoding="utf-8"))
    assert data["meta"]["organizer"] == "anson"
    assert data["meta"]["curated_at"] == "2026-08-21"
    # per-question provenance fields pass through untouched
    assert data["questions"][0]["key_answer"] == "1"
    assert data["questions"][0]["page"] == 60


def test_finalize_auto_sets_curated_at(tmp_path, monkeypatch):
    import re
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps([{"background": "", "question": "Q", "options": [],
                               "answer": "1", "explanations": "",
                               "section": "math", "module": 1, "number": 1}]),
                  encoding="utf-8")
    paper_pdf, answer_pdf = tmp_path / "paper.pdf", tmp_path / "answer.pdf"
    paper_pdf.write_bytes(b"p")
    answer_pdf.write_bytes(b"a")
    monkeypatch.setattr(fp.paper_images, "ImageIndex",
                        type("I", (), {"__init__": lambda self, r: None,
                                        "path": lambda self, n: None}))
    out = tmp_path / "bundle"
    rc = fp.main([
        "--year", "2024", "--month", "10", "--area", "Demo", "--code", "X",
        "--paper-pdf", str(paper_pdf), "--answer-pdf", str(answer_pdf),
        "--questions", str(qp), "--output", str(out),
    ])
    assert rc == 0
    data = json.loads((out / "paper.json").read_text(encoding="utf-8"))
    # curated_at auto-filled with a current ISO timestamp when not provided
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)?",
                        data["meta"]["curated_at"])
    assert not data["meta"]["curated_at"].startswith("xxx")
