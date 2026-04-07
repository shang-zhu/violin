"""API configuration."""

from pathlib import Path

from pipeline import config as _conf

_api = _conf.get()["api"]

JOBS_DIR = Path(_api["jobs_dir"])
MAX_WORKERS = _api["max_workers"]
