"""Unit tests for scripts/openai_batch.py (no network, mocked litellm batch API)."""
import json
from types import SimpleNamespace

import pytest

import openai_batch as ob


# ------------------------------------------------------------------- provider

def test_openai_model_name():
    assert ob.openai_model_name("openai/gpt-5.6-luna") == "gpt-5.6-luna"
    assert ob.openai_model_name("gpt-5.6-luna") == "gpt-5.6-luna"


def test_provider_of_unknown_returns_empty(monkeypatch):
    def boom(model):
        raise ValueError(f"unknown model {model}")

    monkeypatch.setattr("litellm.get_llm_provider", boom)
    assert ob.provider_of("totally-fake") == ("", None)
    assert ob.is_openai("totally-fake") is False


def test_is_openai_requires_native_provider(monkeypatch):
    monkeypatch.setattr(
        "litellm.get_llm_provider",
        lambda m: (m, "openai", None, None),
    )
    assert ob.is_openai("gpt-5.6-luna") is True
    # A custom api_base (e.g. a proxy/vLLM endpoint) must NOT use the cloud
    # Batch API.
    monkeypatch.setattr(
        "litellm.get_llm_provider",
        lambda m: (m, "openai", None, "https://my-proxy.example.com/v1"),
    )
    assert ob.is_openai("gpt-5.6-terra") is False
    monkeypatch.setattr(
        "litellm.get_llm_provider",
        lambda m: (m, "anthropic", None, None),
    )
    assert ob.is_openai("anthropic/claude-3-5-sonnet") is False
    monkeypatch.setattr(
        "litellm.get_llm_provider",
        lambda m: (m, "azure", None, None),
    )
    assert ob.is_openai("azure/gpt-5.6-terra") is False


def test_chat_request_body():
    body = ob.chat_request_body(
        "openai/gpt-5.6-luna", [{"role": "user", "content": "hi"}], json_mode=True
    )
    assert body == {
        "model": "gpt-5.6-luna",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    }
    body2 = ob.chat_request_body("gpt-5.6-luna", [], json_mode=False)
    assert "response_format" not in body2
    assert body2["model"] == "gpt-5.6-luna"


def _chat_body(content):
    """A realistic chat-completion response body as returned by the Batch API."""
    return {"choices": [{"message": {"content": content}}]}


def test_chat_content_extracts_message_text():
    assert ob.chat_content(_chat_body('{"a": 1}')) == '{"a": 1}'
    assert ob.chat_content(_chat_body(None)) == ""
    assert ob.chat_content(_chat_body("")) == ""
    assert ob.chat_content({}) == ""
    assert ob.chat_content("junk") == ""
    assert ob.chat_content({"choices": []}) == ""


# ------------------------------------------------------------------- submit_batch

class _FakeContent:
    def __init__(self, text):
        self.text = text


def _fake_llm_submit(monkeypatch, output_text, statuses):
    """Patch litellm's batch helpers so submit_batch runs without a network."""
    calls = {"create_file": 0, "create_batch": 0, "retrieve": 0}

    def create_file(file, purpose, **kw):
        calls["create_file"] += 1
        assert purpose == "batch"
        assert kw.get("custom_llm_provider") == "openai"
        return SimpleNamespace(id="file-in-1")

    def create_batch(**kw):
        calls["create_batch"] += 1
        assert kw["endpoint"] == "/v1/chat/completions"
        assert kw["completion_window"] == "24h"
        assert kw["input_file_id"] == "file-in-1"
        return SimpleNamespace(id="batch-1", status="in_progress")

    statuses_iter = iter(statuses)

    def retrieve_batch(batch_id, **kw):
        calls["retrieve"] += 1
        return SimpleNamespace(
            id=batch_id, status=next(statuses_iter), output_file_id="file-out-1", error=None
        )

    def file_content(file_id, **kw):
        assert file_id == "file-out-1"
        return _FakeContent(output_text)

    monkeypatch.setattr("litellm.create_file", create_file)
    monkeypatch.setattr("litellm.create_batch", create_batch)
    monkeypatch.setattr("litellm.retrieve_batch", retrieve_batch)
    monkeypatch.setattr("litellm.file_content", file_content)
    monkeypatch.setattr(ob.time, "sleep", lambda secs: calls.setdefault("sleeps", []).append(secs))
    return calls


def _req(cid, content="[1]"):
    return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "gpt-5.6-luna", "messages": []}}


def test_submit_batch_polls_then_returns_bodies(monkeypatch):
    output_text = "\n".join([
        json.dumps({"custom_id": "a", "response": {"status_code": 200, "body": _chat_body("A")}}),
        json.dumps({"custom_id": "b", "response": {"status_code": 200, "body": _chat_body("B")}}),
        json.dumps({"custom_id": "c", "response": {"status_code": 400, "body": {"error": {"message": "bad"}}}}),
    ])
    calls = _fake_llm_submit(monkeypatch, output_text, ["in_progress", "completed"])
    result = ob.submit_batch([_req("a"), _req("b"), _req("c")], poll_interval=5)
    assert result == {"a": _chat_body("A"), "b": _chat_body("B")}
    assert calls["sleeps"] == [5]
    assert calls["retrieve"] == 2


def test_submit_batch_raises_on_terminal_failure(monkeypatch):
    _fake_llm_submit(monkeypatch, "", ["in_progress", "failed"])
    with pytest.raises(RuntimeError, match="failed"):
        ob.submit_batch([_req("a")])


def test_submit_batch_empty_returns_empty(monkeypatch):
    assert ob.submit_batch([]) == {}


# ------------------------------------------------------------------ batch_completions

def test_batch_completions_returns_usable_parse(monkeypatch):
    def fake_submit(requests, *, custom_llm_provider="openai", poll_interval=30, log=None):
        return {
            "a": _chat_body('[{"question": "A"}]'),
            "b": _chat_body("not json at all"),
        }

    monkeypatch.setattr(ob, "submit_batch", fake_submit)
    parse = lambda raw: json.loads(raw) if raw.strip().startswith("[") else None
    tasks = [
        ("a", [{"role": "user", "content": "p1"}], parse),
        ("b", [{"role": "user", "content": "p2"}], parse),
    ]
    out = ob.batch_completions("gpt-5.6-luna", tasks, batch_size=2, json_mode=False, max_retries=1)
    assert out == {"a": [{"question": "A"}]}


def test_batch_completions_retries_bad_json_with_correction_turn(monkeypatch):
    calls = []

    def fake_submit(requests, *, custom_llm_provider="openai", poll_interval=30, log=None):
        calls.append(requests)
        result = {}
        for req in requests:
            cid = req["custom_id"]
            n_turns = len(req["body"]["messages"])
            if n_turns > 2:  # retry round carries the error-correction turn
                result[cid] = _chat_body('[{"ok": true}]')
            else:
                result[cid] = _chat_body("not json")
        return result

    monkeypatch.setattr(ob, "submit_batch", fake_submit)
    parse = lambda raw: json.loads(raw) if raw.strip().startswith("[") else None
    out = ob.batch_completions(
        "gpt-5.6-luna",
        [("a", [{"role": "user", "content": "p"}], parse)],
        batch_size=1, json_mode=False, max_retries=2,
    )
    assert out == {"a": [{"ok": True}]}
    # round 1 failed (1 turn), round 2 succeed (3 turns)
    assert len(calls) == 2
    assert [len(c["body"]["messages"]) for r in calls for c in r] == [1, 3]


def test_batch_completions_chunks_by_batch_size(monkeypatch):
    sizes = []

    def fake_submit(requests, *, custom_llm_provider="openai", poll_interval=30, log=None):
        sizes.append(len(requests))
        return {r["custom_id"]: _chat_body("[null]") for r in requests}

    monkeypatch.setattr(ob, "submit_batch", fake_submit)
    parse = lambda raw: [1]
    tasks = [(f"id-{i}", [], parse) for i in range(5)]
    out = ob.batch_completions("gpt-5.6-luna", tasks, batch_size=2, json_mode=False, max_retries=0)
    assert sizes == [2, 2, 1]
    assert len(out) == 5
