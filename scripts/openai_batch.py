"""OpenAI Batch API plumbing shared by the LLM scripts (tidy/validate/fix).

OpenAI's Batch API runs chat-completion requests asynchronously at a 50%
discount: you upload a JSONL file of request objects, OpenAI processes the job
within a 24h completion window, and you download a JSONL file of responses.

This module wraps litellm's own Batch API helpers (``create_file`` /
``create_batch`` / ``retrieve_batch`` / ``file_content``), which talk to
OpenAI's batches endpoints for native OpenAI providers. Only native OpenAI is
used here (no ``api_base`` override); anything else falls back to the scripts'
existing synchronous per-call path via ``completion()``.

The scripts decide *when* to batch (their ``--batch-size`` argument); this
module only implements the transport. Because a batch job can take a long time
(up to the 24h completion window), every step logs progress through a ``log``
callback so the user is never left staring at a silent screen.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

POLL_INTERVAL = 5  # seconds between batch-status polls
COMPLETION_WINDOW = "24h"
BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_PROVIDER = "openai"

# Terminal statuses that end the status poll. May be "validating"/"in_progress"/
# "finalizing"/"cancelling" while the job is still running.
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _fmt_duration(seconds: float) -> str:
    """Format a seconds count as a compact human-readable duration."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def provider_of(model: str) -> tuple[str, str | None]:
    """Return (litellm provider name, api_base) for a model string.

    Unknown models yield ("", None) instead of raising, so callers never
    crash on a typo'd model name — they just don't batch.
    """
    try:
        from litellm import get_llm_provider

        _model, provider, _api_key, api_base = get_llm_provider(model)
        return provider or "", api_base
    except Exception:
        return "", None


def is_openai(model: str) -> bool:
    """True only for native OpenAI endpoints where the Batch API exists."""
    provider, api_base = provider_of(model)
    return provider == "openai" and not api_base


def openai_model_name(model: str) -> str:
    """Strip a leading 'openai/' prefix; native OpenAI model names pass through.

    The Batch API request body must use the bare model name (no provider
    prefix), unlike litellm.
    """
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def chat_request_body(model: str, messages: list[dict], *, json_mode: bool) -> dict:
    """Build the ``body`` of one JSONL chat-completion request."""
    body: dict = {"model": openai_model_name(model), "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def chat_content(body: object) -> str:
    """Extract the assistant message text from a chat-completion response body.

    A Batch API response ``body`` is a full ChatCompletion object, so the text
    lives at ``choices[0].message.content`` — not at the top level. Returns ""
    when the body isn't a chat completion (or content is missing/refused).
    """
    if not isinstance(body, dict):
        return ""
    try:
        choices = body["choices"]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content)


def _upload_requests(requests: list[dict], *, custom_llm_provider: str) -> str:
    """Write requests to a temp JSONL file and upload it; return the file id."""
    import litellm

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=False
    ) as fh:
        for req in requests:
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")
        path = fh.name
    try:
        with open(path, "rb") as fh:
            file_obj = litellm.create_file(
                file=fh, purpose="batch", custom_llm_provider=custom_llm_provider
            )
        return file_obj.id
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def submit_batch(
    requests: list[dict],
    *,
    custom_llm_provider: str = BATCH_PROVIDER,
    poll_interval: int = POLL_INTERVAL,
    log=None,
) -> dict[str, dict]:
    """Upload ``requests`` as one batch job, poll until it completes, and
    return {custom_id: response body} for every successful request.

    Raises when the batch job itself fails; requests whose individual response
    is an error are simply absent from the returned mapping.

    Every step is reported through ``log`` (initially to print progress lines);
    if omitted, it defaults to a no-op. The polling loop reports status, how
    many requests are done, and elapsed time on each poll so the user sees the
    job is progressing rather than hanging.
    """
    if not requests:
        return {}
    log = log or (lambda *a, **k: None)
    import litellm

    input_file_id = _upload_requests(requests, custom_llm_provider=custom_llm_provider)
    batch = litellm.create_batch(
        input_file_id=input_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        custom_llm_provider=custom_llm_provider,
    )

    # Make the async nature explicit: the job is queued, not running inline.
    log(
        f"  submitted batch {batch.id}: {len(requests)} request(s) queued on "
        f"OpenAI (completion window {COMPLETION_WINDOW}); polling for results..."
    )
    started = time.monotonic()
    while True:
        batch = litellm.retrieve_batch(batch.id, custom_llm_provider=custom_llm_provider)
        status = batch.status
        elapsed = time.monotonic() - started
        counts = getattr(batch, "request_counts", None) or {}
        done = getattr(counts, "completed", None)
        total = getattr(counts, "total", None)
        failed = getattr(counts, "failed", None)

        if status == "completed":
            log(
                f"  batch {batch.id} completed: "
                f"{done or 0}/{total or len(requests)} done "
                f"({0 if failed is None else failed} failed) in {_fmt_duration(elapsed)}"
            )
            break
        if status in _TERMINAL_STATUSES:
            err = getattr(batch, "errors", None) or getattr(batch, "error_file_id", None)
            raise RuntimeError(
                f"OpenAI batch {batch.id} ended with status {status}: {err}"
            )

        # Liveness feedback: nothing is logged for a while only when a single
        # poll takes long, so the user always sees the job is moving.
        progress = f"{done}/{total}" if total else "?"
        log(
            f"  batch {batch.id}: {status} — {progress} done, "
            f"elapsed {_fmt_duration(elapsed)}"
        )
        time.sleep(poll_interval)

    content = litellm.file_content(
        batch.output_file_id, custom_llm_provider=custom_llm_provider
    )
    results: dict[str, dict] = {}
    for line in content.text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        cid = record.get("custom_id")
        response = record.get("response") or {}
        if cid and response.get("status_code") == 200 and isinstance(
            response.get("body"), dict
        ):
            results[cid] = response["body"]
    return results


def batch_completions(
    model: str,
    tasks: list[tuple[str, list[dict], object]],
    *,
    batch_size: int,
    json_mode: bool,
    max_retries: int,
    poll_interval: int = POLL_INTERVAL,
    log=None,
) -> dict[str, object]:
    """Run chat-completion tasks through the OpenAI Batch API.

    ``tasks`` is a list of ``(custom_id, messages, parse)`` tuples, where
    ``parse(raw_content)`` returns the desired value or None when the response
    is unusable. Requests are submitted in chunks of ``batch_size``; responses
    whose parse fails are re-submitted (with an error-correction turn) for up
    to ``max_retries`` additional rounds. Returns {custom_id: value} for the
    tasks that eventually produced a usable response.

    Progress is streamed through ``log`` (default: a no-op); pass ``print`` to
    surface submission, polling and completion messages.
    """
    log = log or (lambda *a, **k: None)
    results: dict[str, object] = {}
    pending = list(tasks)
    for round_no in range(max_retries + 1):
        if not pending:
            break
        retry: list[tuple[str, list[dict], object]] = []
        chunks = [
            pending[i : i + batch_size] for i in range(0, len(pending), batch_size)
        ]
        for ci, chunk in enumerate(chunks, 1):
            requests = [
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": BATCH_ENDPOINT,
                    "body": chat_request_body(model, messages, json_mode=json_mode),
                }
                for cid, messages, _ in chunk
            ]
            if len(chunks) > 1:
                log(
                    f"  submitting chunk {ci}/{len(chunks)}: {len(chunk)} request(s) "
                    f"to OpenAI Batch API (round {round_no + 1})"
                )
            else:
                log(
                    f"  submitting {len(chunk)} request(s) to OpenAI Batch API "
                    f"(round {round_no + 1})"
                )
            try:
                bodies = submit_batch(requests, poll_interval=poll_interval, log=log)
            except Exception as exc:
                log(f"  batch submission failed: {exc}", file=sys.stderr)
                if round_no < max_retries:
                    retry.extend(chunk)
                else:
                    log(
                        f"  {len(chunk)} request(s) dropped after "
                        f"{max_retries + 1} round(s)",
                        file=sys.stderr,
                    )
                continue
            for cid, messages, parse in chunk:
                raw = chat_content(bodies.get(cid))
                value = parse(raw)
                if value is not None:
                    results[cid] = value
                    continue
                if round_no < max_retries and raw.strip():
                    retry.append(
                        (
                            cid,
                            list(messages)
                            + [
                                {"role": "assistant", "content": raw},
                                {
                                    "role": "user",
                                    "content": (
                                        "That was not valid JSON. Output ONLY the "
                                        "JSON described above and nothing else."
                                    ),
                                },
                            ],
                            parse,
                        )
                    )
                else:
                    log(
                        f"  {cid}: no usable response after {round_no + 1} round(s)",
                        file=sys.stderr,
                    )
        log(f"  round {round_no + 1} done: {len(results)} resolved so far")
        pending = retry
    if results:
        log(f"  batch run finished: {len(results)}/{len(tasks)} request(s) produced usable results")
    return results