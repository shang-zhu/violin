"""Translate transcript segments using Together AI Qwen3.5-397B."""

import json
import os
import time
from datetime import datetime, timezone

from together import Together
from together import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

_TRANSIENT_ERRORS = (APITimeoutError, InternalServerError, RateLimitError)

from . import config as _conf
from .costs import CostTracker
from .transcriber import Segment

import prompts as _prompts


def _tcfg() -> dict:
    return _conf.get()["translation"]

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
    },
    "required": ["translation"],
    "additionalProperties": False,
}


def _dump_debug(tag: str, attempt: int, prompt: str, raw: str, error: str, texts: list[str]) -> str:
    """Write a debug log file and return its path."""
    os.makedirs(_tcfg()["debug_dir"], exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_tcfg()["debug_dir"], f"{ts}_attempt{attempt}_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": ts,
                "attempt": attempt,
                "error": error,
                "expected_count": len(texts),
                "input_texts": texts,
                "prompt": prompt,
                "raw_response": raw,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path


def _translate_single(
    text: str,
    target_language: str,
    source_language: str,
    client: Together,
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
) -> str:
    """Translate one segment with retry on transient API errors."""
    cfg = _conf.get()
    max_retries = cfg["translation"]["max_retries"]
    temp = style_temperature if style_temperature is not None else cfg["translation"]["temperature"]

    fmt = dict(
        source_language=source_language,
        target_language=target_language,
        text=json.dumps(text, ensure_ascii=False),
        style_directives=style_directives,
    )
    if style_directives:
        system_msg = _prompts.load("translate", "single_system_styled", **fmt)
        user_msg = _prompts.load("translate", "single_user_styled", **fmt)
    else:
        system_msg = _prompts.load("translate", "single_system", **fmt)
        user_msg = _prompts.load("translate", "single_user", **fmt)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg["models"]["translation"],
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temp,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "single_translation",
                        "strict": True,
                        "schema": SINGLE_SCHEMA,
                    },
                },
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if tracker and hasattr(response, "usage") and response.usage:
                tracker.add_llm_usage(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)["translation"]

        except _TRANSIENT_ERRORS as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"        ⚠ API error (attempt {attempt}): {exc}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _try_batch(
    texts: list[str],
    target_language: str,
    source_language: str,
    client: Together,
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
) -> list[str] | None:
    """Attempt to translate a batch. Returns translations on success, None on failure."""
    numbered = "\n".join(
        f"[{i}]: {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(texts)
    )

    fmt = dict(
        source_language=source_language,
        target_language=target_language,
        num_segments=len(texts),
        numbered_segments=numbered,
        style_directives=style_directives,
    )
    if style_directives:
        system_msg = _prompts.load("translate", "batch_system_styled", **fmt)
        prompt = _prompts.load("translate", "batch_user_styled", **fmt)
    else:
        system_msg = _prompts.load("translate", "batch_system", **fmt)
        prompt = _prompts.load("translate", "batch_user", **fmt)

    cfg = _conf.get()
    max_retries = cfg["translation"]["max_retries"]
    temp = style_temperature if style_temperature is not None else cfg["translation"]["temperature"]
    for attempt in range(1, max_retries + 1):
        raw = ""
        try:
            response = client.chat.completions.create(
                model=cfg["models"]["translation"],
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation_response",
                        "strict": True,
                        "schema": BATCH_SCHEMA,
                    },
                },
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            if tracker and hasattr(response, "usage") and response.usage:
                tracker.add_llm_usage(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                )

            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            translated = result["translations"]

            if len(translated) == len(texts):
                return translated

            err_msg = f"count mismatch: expected {len(texts)}, got {len(translated)}"
            _dump_debug("count_mismatch", attempt, prompt, raw, err_msg, texts)
            if attempt < max_retries:
                print(f"      ⚠ Count mismatch (attempt {attempt}), retrying...")
                time.sleep(2 ** attempt)

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            _dump_debug("parse_error", attempt, prompt, raw, f"{type(exc).__name__}: {exc}", texts)
            if attempt < max_retries:
                print(f"      ⚠ Parse error (attempt {attempt}): {exc}, retrying...")
                time.sleep(2 ** attempt)

        except _TRANSIENT_ERRORS as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"      ⚠ API error (attempt {attempt}): {exc}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ✗ API error after {max_retries} attempts: {exc}")

    return None


def _translate_batch(
    texts: list[str],
    target_language: str,
    source_language: str,
    client: Together,
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
) -> list[str]:
    """Translate a batch with binary-split fallback on failure."""
    result = _try_batch(texts, target_language, source_language, client, tracker, style_directives, style_temperature)
    if result is not None:
        return result

    if len(texts) == 1:
        print(f"        → single-segment fallback...", end="", flush=True)
        t = _translate_single(texts[0], target_language, source_language, client, tracker, style_directives, style_temperature)
        print(" done")
        return [t]

    mid = len(texts) // 2
    print(f"      ↓ Splitting failed batch of {len(texts)} → {mid} + {len(texts) - mid}")
    left = _translate_batch(texts[:mid], target_language, source_language, client, tracker, style_directives, style_temperature)
    right = _translate_batch(texts[mid:], target_language, source_language, client, tracker, style_directives, style_temperature)
    return left + right


def translate_segments(
    segments: list[Segment],
    target_language: str,
    client: Together,
    source_language: str = "auto-detect",
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
) -> list[Segment]:
    """Translate all segments, batching to stay within LLM context limits."""
    translated_texts: list[str] = []

    batch_size = _tcfg()["batch_size"]
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        texts = [s.text for s in batch]
        print(f"      Translating segments {i + 1}–{i + len(batch)} / {len(segments)}...")
        translated_texts.extend(
            _translate_batch(texts, target_language, source_language, client, tracker, style_directives, style_temperature)
        )

    return [
        Segment(id=s.id, start=s.start, end=s.end, text=t)
        for s, t in zip(segments, translated_texts)
    ]
