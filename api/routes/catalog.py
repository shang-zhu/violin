"""Catalog endpoints: list supported languages, voices, and styles."""

from __future__ import annotations

from fastapi import APIRouter

from pipeline.languages import all_languages
from pipeline.styles import list_styles
from pipeline.tts import all_voices, native_voices_for

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
    from pipeline.languages import language_code
    code = language_code(language)
    return native_voices_for(code)


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
