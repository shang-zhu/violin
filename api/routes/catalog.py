"""Catalog endpoints: list supported languages, voices, and styles."""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from api.models import VoiceCandidate, VoiceMatchRequest, VoiceMatchResponse
from pipeline import config as _conf
from pipeline.languages import all_languages, language_code
from pipeline.llm_client import get_translation_model, make_translation_client
from pipeline.styles import list_styles
from pipeline.tts import all_voices, native_voices_for, voice_descriptions
import prompts as _prompts

load_dotenv(override=True)

router = APIRouter(tags=["catalog"])


@router.get("/languages")
def list_languages() -> dict[str, str]:
    """Return a mapping of language name → BCP-47 code for all supported languages."""
    return all_languages()


@router.get("/voices")
def list_voices() -> dict[str, list[str]]:
    """Return all known native Cartesia Sonic 3 voices grouped by BCP-47 language code."""
    return all_voices()


@router.get("/voices/{language}")
def voices_for_language(language: str) -> list[str]:
    """Return native voices for a specific language name or BCP-47 code."""
    return native_voices_for(language_code(language))


@router.get("/styles")
def get_styles() -> list[dict]:
    """Return all available translation style profiles."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "tts_speed": s.tts_speed,
            "tts_emotion": s.tts_emotion,
        }
        for s in list_styles()
    ]


def _build_voice_catalog(target_lang: str) -> str:
    """Format the active provider's voice catalog as a string for the LLM prompt.

    Each voice is rendered as `- <name> — <description>`. For Cartesia the
    description is the name itself (e.g. 'german conversational woman'); for
    ElevenLabs it is the official metadata description.
    """
    voices = all_voices()
    descriptions = voice_descriptions()
    target_code = language_code(target_lang) if target_lang else ""
    lines: list[str] = []

    def _fmt(name: str) -> str:
        d = descriptions.get(name, "")
        return f"  - {name} — {d}" if d and d != name else f"  - {name}"

    if target_code and target_code in voices:
        lines.append(f"== Voices for target language ({target_code}) ==")
        for v in voices[target_code]:
            lines.append(_fmt(v))
        lines.append("")

    for code, voice_list in sorted(voices.items()):
        if code == target_code:
            continue
        header = "All voices (multilingual)" if code == "multi" else code
        lines.append(f"== {header} ==")
        for v in voice_list:
            lines.append(_fmt(v))
    return "\n".join(lines)


_MAX_VOICE_MATCH_RETRIES = 3


def _parse_voice_candidates(raw: str, all_ids: list[str]) -> list[VoiceCandidate]:
    """Try to extract a list of VoiceCandidates from the raw LLM response."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    bracket_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if bracket_match:
        raw = bracket_match.group(0)

    items = json.loads(raw)
    if isinstance(items, dict):
        items = [items]

    all_lower = {v.lower(): v for v in all_ids}

    def _normalize(name: str) -> str | None:
        return all_lower.get(name.lower().strip())

    candidates = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            voice, explanation = _normalize(item), ""
        else:
            voice = _normalize(item.get("voice", ""))
            explanation = item.get("explanation", "")
        if voice and voice.lower() not in seen:
            seen.add(voice.lower())
            candidates.append(VoiceCandidate(voice=voice, explanation=explanation))
        if len(candidates) == 3:
            break
    return candidates


@router.post("/voice-match", response_model=VoiceMatchResponse)
def match_voice(payload: VoiceMatchRequest):
    """Use an LLM to map a natural language voice description to the best voice in the catalog.

    Reuses the translation client + model (``models.translation``) — no separate
    configuration needed.
    """
    cfg = _conf.get()
    try:
        client = make_translation_client(
            cfg,
            together_key_override=payload.together_api_key.strip() or None,
            openai_key_override=payload.openai_api_key.strip() or None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    catalog = _build_voice_catalog(payload.language)
    all_ids: list[str] = []
    for v_list in all_voices().values():
        all_ids.extend(v_list)
    all_ids = list(dict.fromkeys(all_ids))

    messages = [
        {
            "role": "system",
            "content": _prompts.load("voice_match", "system", catalog=catalog),
        },
        {
            "role": "user",
            "content": _prompts.load(
                "voice_match", "user",
                language=payload.language or "not specified",
                description=payload.description,
            ),
        },
    ]

    # voice_match reuses the translation client + model — same LLM, same provider.
    model = get_translation_model(cfg)

    last_error = ""
    for attempt in range(_MAX_VOICE_MATCH_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=400,
            )
            msg = response.choices[0].message
            raw = (msg.content or "").strip()

            if not raw:
                last_error = "LLM returned empty response"
                continue

            candidates = _parse_voice_candidates(raw, all_ids)
            if candidates:
                return VoiceMatchResponse(candidates=candidates)
            last_error = "parsed JSON but found no valid voice names"
        except (json.JSONDecodeError, Exception) as exc:
            last_error = str(exc)

    raise HTTPException(status_code=502, detail=f"Voice matching failed after {_MAX_VOICE_MATCH_RETRIES} attempts ({last_error})")
