"""TTS public dispatcher — picks Cartesia, ElevenLabs, or OpenAI based on config."""

from __future__ import annotations

import os
from typing import Any

from . import config as _conf
from .costs import CostTracker
from .transcriber import Segment


def _tts_entry() -> dict[str, Any]:
    """Return the models.tts config entry as a dict.

    Supports both the new dict format and the legacy plain-string format:
        # new
        tts:
          provider: cartesia
          model: cartesia/sonic-3
        # legacy
        tts: "cartesia/sonic-3"
    """
    entry = _conf.get()["models"]["tts"]
    if isinstance(entry, dict):
        return entry
    return {"provider": "cartesia", "model": entry}


def get_tts_provider() -> str:
    return _tts_entry().get("provider", "cartesia")


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
        from . import tts_cartesia as _imp
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
    together_client: Any | None = None,
    elevenlabs_api_key: str | None = None,
    openai_api_key: str | None = None,
):
    """Build (or reuse) the right SDK client for the active provider."""
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

    # cartesia (default) — uses the Together client passed by the caller, or
    # build one from TOGETHER_API_KEY.
    if together_client is not None:
        return together_client
    from together import Together
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set.")
    return Together(api_key=api_key)


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: Any | None = None,
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
    *,
    elevenlabs_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> list[str]:
    """Synthesize all segments concurrently using the configured TTS provider.

    *client* is the legacy Cartesia path's Together client; ignored for
    elevenlabs / openai providers, which use their own API keys (env var or
    the *_api_key kwargs).
    """
    provider = get_tts_provider()
    backend_client = _make_client(
        provider,
        together_client=client,
        elevenlabs_api_key=elevenlabs_api_key,
        openai_api_key=openai_api_key,
    )

    return _backend(provider).synthesize_segments(
        segments, voice, output_dir, backend_client, language,
        voice_map, tracker, speed, emotion,
    )
