"""External API pricing — used for cost telemetry only (CLI summary + stats DB).

Not user configuration; update these values when a provider changes prices.
Cost tracking is informational — wrong numbers here do not affect pipeline
behaviour, just the dollar figure reported alongside each job.

Sources (re-check these pages when bumping LAST_UPDATED):
  - Together AI:  https://www.together.ai/pricing
  - OpenAI:       https://openai.com/api/pricing/
  - ElevenLabs:   https://elevenlabs.io/pricing
"""

from __future__ import annotations

LAST_UPDATED = "2026-05-12"   # bump this whenever any rate below changes

# USD per audio minute.
WHISPER: dict[str, float] = {
    "together": 0.0015,   # Together AI — Whisper Large v3
    "openai":   0.006,    # OpenAI — whisper-1
}

# USD per 1,000,000 characters.
TTS: dict[str, dict[str, float]] = {
    "together":   {"per_m_characters": 65.00},   # Cartesia Sonic 3 ($65/M). Kokoro $4/M, Orpheus $15/M when added.
    "elevenlabs": {"per_m_characters": 165.00},  # eleven_v3 (~$0.165 / 1k chars on Creator tier)
    "openai":     {"per_m_characters": 30.00},   # tts-1-hd ($30/M); tts-1 is $15/M
}

# USD per 1,000,000 tokens (input / output split).
TRANSLATION: dict[str, dict[str, float]] = {
    "together": {"per_m_input_tokens": 0.30, "per_m_output_tokens": 1.20},
    "openai":   {"per_m_input_tokens": 0.75, "per_m_output_tokens": 4.50},
}


def whisper_per_minute(provider: str) -> float:
    return WHISPER.get(provider, WHISPER["together"])


def tts_per_m_characters(provider: str) -> float:
    return TTS.get(provider, TTS["together"])["per_m_characters"]


def translation_rates(provider: str) -> dict[str, float]:
    """Return both input and output per-million-token rates for a provider."""
    return TRANSLATION.get(provider, TRANSLATION["together"])
