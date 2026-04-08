"""Style profiles for controlling translation tone and voice delivery."""

from __future__ import annotations

from dataclasses import dataclass

from . import config as _conf


@dataclass(frozen=True)
class StyleProfile:
    name: str
    description: str
    translation_directives: str
    temperature: float | None
    tts_speed: float | None
    tts_emotion: str | None


def resolve(name: str) -> StyleProfile:
    """Look up a style profile by name from the loaded config."""
    styles = _conf.get().get("styles", {})
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
    styles = _conf.get().get("styles", {})
    return [resolve(name) for name in sorted(styles)]
