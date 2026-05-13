"""IP-based usage tracking for free trial enforcement.

Stores a simple JSON file mapping IP addresses to the number of jobs
completed using the server's API key (i.e. without a user-provided key).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import FREE_TRIAL_JOBS, JOBS_DIR

_USAGE_FILE = JOBS_DIR / "_ip_usage.json"
_lock = threading.Lock()


def _load() -> dict[str, int]:
    if _USAGE_FILE.exists():
        try:
            return json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data: dict[str, int]) -> None:
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USAGE_FILE.write_text(json.dumps(data), encoding="utf-8")


def get_usage(ip: str) -> int:
    with _lock:
        return _load().get(ip, 0)


def record_usage(ip: str) -> None:
    """Increment the server-key usage count for an IP."""
    with _lock:
        data = _load()
        data[ip] = data.get(ip, 0) + 1
        _save(data)


def has_free_trial(ip: str) -> bool:
    """Return True if this IP still has free trial jobs remaining.

    ``FREE_TRIAL_JOBS <= 0`` means BYOK-only: no free trial is offered, every
    request must carry its own API key.
    """
    if FREE_TRIAL_JOBS <= 0:
        return False
    return get_usage(ip) < FREE_TRIAL_JOBS


def remaining_trials(ip: str) -> int:
    """Return how many free trial jobs remain for this IP (0 in BYOK-only mode)."""
    if FREE_TRIAL_JOBS <= 0:
        return 0
    return max(0, FREE_TRIAL_JOBS - get_usage(ip))
