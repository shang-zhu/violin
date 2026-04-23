"""Factory for translation LLM clients — supports Together AI and OpenAI."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


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


def make_translation_client(cfg: dict[str, Any], *, api_key_override: str | None = None):
    """Create the appropriate chat client based on the translation provider config.

    If *api_key_override* is provided it is used instead of the environment variable.
    """
    provider, _ = _parse_translation_config(cfg)

    if provider == "openai":
        from openai import OpenAI
        api_key = api_key_override or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key)

    from together import Together
    api_key = api_key_override or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)
