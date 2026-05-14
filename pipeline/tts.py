"""TTS public dispatcher — picks Cartesia, ElevenLabs, or OpenAI based on config."""

from __future__ import annotations

import os
from typing import Any

from . import config as _conf
from .costs import CostTracker
from .transcriber import Segment


def _tts_entry() -> dict[str, Any]:
    return _conf.get()["models"]["tts"]


def get_tts_provider() -> str:
    return _tts_entry().get("provider", "together")


def get_tts_model() -> str:
    return _tts_entry()["model"]


def _backend(provider: str | None = None):
    """Resolve the active provider's backend module."""
    p = provider or get_tts_provider()
    if p == "elevenlabs":
        from . import tts_elevenlabs as _imp
    elif p == "openai":
        from . import tts_openai as _imp
    else:
        from . import tts_together as _imp
    return _imp


def native_voices_for(language_code: str) -> list[str]:
    """Return [primary_male, primary_female] voices for a language."""
    return _backend().native_voices_for(language_code)


def all_voices() -> dict[str, list[str]]:
    """Return the full voice catalog grouped by language code."""
    return _backend().all_voices()


def voice_descriptions() -> dict[str, str]:
    """Return name → description mapping for the active provider's voices."""
    return _backend().voice_descriptions()


def _make_client(
    provider: str,
    *,
    together_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
    openai_api_key: str | None = None,
):
    """Build the right SDK client for the active provider, honoring caller-supplied API keys."""
    if provider == "elevenlabs":
        from elevenlabs.client import ElevenLabs
        api_key = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. Provide one via env var or "
                "pass elevenlabs_api_key= when calling synthesize_segments."
            )
        return ElevenLabs(api_key=api_key)

    if provider == "openai":
        from openai import OpenAI
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide one via env var or "
                "pass openai_api_key= when calling synthesize_segments."
            )
        return OpenAI(api_key=api_key)

    # cartesia (default) — Together-hosted; honor user-provided key, fall back to env.
    from together import Together
    api_key = together_api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TOGETHER_API_KEY is not set. Provide one via env var or "
            "pass together_api_key= when calling synthesize_segments."
        )
    return Together(api_key=api_key)


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
    *,
    together_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> list[str]:
    """Synthesize all segments concurrently using the configured TTS provider.

    Each provider's *_api_key kwarg overrides the corresponding env var
    (TOGETHER_API_KEY / ELEVENLABS_API_KEY / OPENAI_API_KEY).
    """
    provider = get_tts_provider()
    backend_client = _make_client(
        provider,
        together_api_key=together_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        openai_api_key=openai_api_key,
    )

    return _backend(provider).synthesize_segments(
        segments, voice, output_dir, backend_client, language,
        voice_map, tracker, speed, emotion,
    )
