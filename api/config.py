"""API configuration."""

from pathlib import Path

from pipeline import config as _conf

_api = _conf.get()["api"]

JOBS_DIR = Path(_api["jobs_dir"])
MAX_WORKERS = _api["max_workers"]
JOB_TTL_HOURS = _api.get("job_ttl_hours", 24)
FREE_TRIAL_JOBS = _api.get("free_trial_jobs", 1)
