"""Style profiles for controlling translation tone and voice delivery.

Profiles live in ``prompts/styles.yaml`` — edit that file to add or tweak
styles. Loaded lazily on first access and cached for the process lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_STYLES_PATH = Path(__file__).resolve().parent.parent / "prompts" / "styles.yaml"


@dataclass(frozen=True)
class StyleProfile:
    name: str
    description: str
    translation_directives: str
    temperature: float | None
    tts_speed: float | None
    tts_emotion: str | None


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, Any]]:
    with open(_STYLES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{_STYLES_PATH} must contain a top-level mapping of style names")
    return data


def resolve(name: str) -> StyleProfile:
    """Look up a style profile by name."""
    styles = _load()
    if name not in styles:
        available = ", ".join(sorted(styles)) or "(none defined)"
        raise ValueError(f"Unknown style {name!r}. Available: {available}")

    entry = styles[name]
    trans = entry.get("translation") or {}
    tts = entry.get("tts") or {}
    return StyleProfile(
        name=name,
        description=entry.get("description", ""),
        translation_directives=trans.get("directives", "") or "",
        temperature=trans.get("temperature"),
        tts_speed=tts.get("speed"),
        tts_emotion=tts.get("emotion"),
    )


def list_styles() -> list[StyleProfile]:
    """Return all available style profiles sorted by name."""
    return [resolve(name) for name in sorted(_load())]
