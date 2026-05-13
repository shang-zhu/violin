"""Centralized configuration loaded from config/default.yaml.

Override per-deployment with ``--config path/to/override.yaml`` — values
deep-merge so the override file only needs the keys it actually changes.
"""

from __future__ import annotations

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
    """Load defaults, optionally deep-merging an override YAML on top."""
    global _cfg
    with open(_DEFAULT_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    if config_path is not None:
        with open(config_path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        base = _deep_merge(base, override)

    _cfg = base
    return _cfg


def get() -> dict[str, Any]:
    """Return the loaded config, loading defaults if not yet initialized."""
    if _cfg is None:
        return load()
    return _cfg
