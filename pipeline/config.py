"""Centralized configuration loaded from config/default.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_PATH = _CONFIG_DIR / "default.yaml"

_cfg: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config, optionally merging a user override file on top of defaults."""
    global _cfg
    with open(_DEFAULT_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    if config_path is not None:
        with open(config_path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        base = _deep_merge(base, override)

    # Allow env-var overrides for common knobs
    env_map = {
        "TRANSLATE_DEBUG_DIR": ("translation", "debug_dir"),
        "TTS_WORKERS": ("tts", "workers", int),
        "MERGE_WORKERS": ("merge_video", "workers", int),
        "JOBS_DIR": ("api", "jobs_dir"),
        "MAX_WORKERS": ("api", "max_workers", int),
    }
    for env_key, spec in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            *path, last = spec if not callable(spec[-1]) else spec[:-1]
            cast = spec[-1] if callable(spec[-1]) else str
            section = base
            for p in path:
                section = section.setdefault(p, {})
            section[last] = cast(val)

    _cfg = base
    return _cfg


def get() -> dict[str, Any]:
    """Return the loaded config, loading defaults if not yet initialized."""
    if _cfg is None:
        return load()
    return _cfg
