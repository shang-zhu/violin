"""Prompt loader — reads YAML prompt files from the prompts/ directory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def _load_yaml(name: str) -> dict[str, str]:
    path = _PROMPTS_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load(name: str, key: str, **kwargs: object) -> str:
    """Load a prompt template and fill ``{var}`` placeholders.

    Usage::

        prompts.load("translate", "batch_system", source_language="English", target_language="Chinese")
    """
    data = _load_yaml(name)
    template = data[key].strip()

    class _Safe(dict):
        def __missing__(self, k: str) -> str:
            return "{" + k + "}"

    return template.format_map(_Safe(kwargs))
