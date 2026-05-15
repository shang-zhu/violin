"""Factory for translation + transcription clients — Together AI and OpenAI.

OpenAI stages may use ``base_url`` (YAML per ``models.<stage>`` or ``OPENAI_BASE_URL``)
for private OpenAI-compatible servers.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)


def _resolve_openai_base_url(cfg: dict[str, Any], section: str) -> str | None:
    """Return OpenAI-compatible API base URL for *section* (``models.<section>``).

    Order: non-empty ``base_url`` on that model's dict entry, else ``OPENAI_BASE_URL`` env.
    *section* is one of: ``transcription``, ``translation``, ``chat``, ``tts``.
    """
    entry = cfg.get("models", {}).get(section)
    if isinstance(entry, dict):
        url = entry.get("base_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    env_url = os.environ.get("OPENAI_BASE_URL")
    return env_url.strip() if env_url and env_url.strip() else None


def make_openai_client_for_section(
    cfg: dict[str, Any],
    section: str,
    *,
    openai_key_override: str | None = None,
):
    """Build ``OpenAI`` SDK client with optional ``base_url`` (private / compatible gateways).

    ``OPENAI_API_KEY`` may be empty (e.g. gateways that do not check the key); use
    ``openai_key_override`` to pass a key explicitly, including ``""``.
    """
    from openai import OpenAI

    if openai_key_override is not None:
        api_key = openai_key_override.strip()
    else:
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = _resolve_openai_base_url(cfg, section)
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _parse_translation_config(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (provider, model) from the models.translation config entry.

    Supports both the new dict format and the legacy plain-string format:
        # new
        translation:
          provider: openai
          model: gpt-4.1
        # legacy (treated as together)
        translation: "Qwen/Qwen3.5-397B-A17B"
    """
    entry = cfg["models"]["translation"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_translation_model(cfg: dict[str, Any]) -> str:
    """Return the model name string for translation."""
    _, model = _parse_translation_config(cfg)
    return model


def get_translation_provider(cfg: dict[str, Any]) -> str:
    """Return 'openai' or 'together'."""
    provider, _ = _parse_translation_config(cfg)
    return provider


def make_translation_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
):
    """Create the appropriate chat client based on the translation provider config.

    *together_key_override* is used when provider is 'together'.
    *openai_key_override* is used when provider is 'openai'.
    Each falls back to the corresponding environment variable if not provided.
    """
    provider, _ = _parse_translation_config(cfg)

    if provider == "openai":
        return make_openai_client_for_section(
            cfg, "translation", openai_key_override=openai_key_override
        )

    from together import Together
    api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)


# ── Chat (video Q&A) client ─────────────────────────────────

def _parse_chat_config(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (provider, model) from the models.chat config entry.

    Supports both the dict format and the legacy plain-string format:
        # current
        chat:
          provider: together
          model: Qwen/Qwen3.5-397B-A17B
        # legacy
        chat: "Qwen/Qwen3.5-397B-A17B"
    """
    entry = cfg["models"]["chat"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_chat_provider(cfg: dict[str, Any]) -> str:
    provider, _ = _parse_chat_config(cfg)
    return provider


def get_chat_model(cfg: dict[str, Any]) -> str:
    _, model = _parse_chat_config(cfg)
    return model


def make_chat_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
):
    """Create the chat client based on ``models.chat.provider``.

    Independent from translation because chat needs a vision-language model;
    the translation provider may not host one.
    """
    provider, _ = _parse_chat_config(cfg)

    if provider == "openai":
        return make_openai_client_for_section(
            cfg, "chat", openai_key_override=openai_key_override
        )

    from together import Together
    api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)


# ── Startup validation ──────────────────────────────────────

_PROVIDER_ENV_KEY = {
    "together":   "TOGETHER_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
}


def required_env_keys(cfg: dict[str, Any]) -> set[str]:
    """Return the env var names required by the active provider config."""
    keys: set[str] = set()

    keys.add(_PROVIDER_ENV_KEY[get_transcription_provider(cfg)])
    keys.add(_PROVIDER_ENV_KEY[get_translation_provider(cfg)])
    keys.add(_PROVIDER_ENV_KEY[get_chat_provider(cfg)])

    tts_entry = cfg["models"].get("tts")
    if isinstance(tts_entry, dict):
        tts_provider = tts_entry.get("provider", "together")
    else:
        tts_provider = "together"
    if tts_provider == "cartesia":  # legacy alias
        tts_provider = "together"
    keys.add(_PROVIDER_ENV_KEY[tts_provider])

    return keys


def validate_env(cfg: dict[str, Any]) -> list[str]:
    """Return env var names that are required but unset (sorted, deduped)."""
    return sorted(
        k
        for k in required_env_keys(cfg)
        if k != "OPENAI_API_KEY" and not os.environ.get(k)
    )


def _parse_transcription_config(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (provider, model) from the models.transcription config entry.

    Supports both the new dict format and the legacy plain-string format:
        # new
        transcription:
          provider: openai
          model: whisper-1
        # legacy (treated as together)
        transcription: "openai/whisper-large-v3"
    """
    entry = cfg["models"]["transcription"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_transcription_model(cfg: dict[str, Any]) -> str:
    _, model = _parse_transcription_config(cfg)
    return model


def get_transcription_provider(cfg: dict[str, Any]) -> str:
    provider, _ = _parse_transcription_config(cfg)
    return provider


def make_transcription_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
):
    """Create the appropriate Whisper client based on the transcription provider config.

    The resulting client exposes `audio.transcriptions.create(...)` — both the
    Together and OpenAI SDKs share that surface, so the transcriber code does
    not need to branch.
    """
    provider, _ = _parse_transcription_config(cfg)

    if provider == "openai":
        return make_openai_client_for_section(
            cfg, "transcription", openai_key_override=openai_key_override
        )

    from together import Together
    api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)
