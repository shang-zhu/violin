"""File-based job storage.

Each job lives in JOBS_DIR/{job_id}/:
    meta.json       — JobStatus + parameters
    progress.jsonl  — append-only progress events (one JSON object per line)
    input.<ext>     — uploaded source video
    output.mp4      — dubbed video (present when status=done)
    output.srt      — subtitle file (present when status=done and subtitles=True)
    segments.json   — aligned subtitle/timeline segments for playback + chat context
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .config import JOBS_DIR
from .models import JobResponse, JobStatus, ProgressEvent

logger = logging.getLogger(__name__)

_lock = threading.Lock()


# Patterns for stripping API keys out of error messages before persisting them
# to disk. Matches the known prefixes (OpenAI / ElevenLabs) and any sufficiently
# long hex / alphanumeric run that looks like a Together key.
_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),       # OpenAI / Anthropic-style
    re.compile(r"sk_[A-Za-z0-9_-]{20,}"),       # ElevenLabs
    re.compile(r"\b[A-Fa-f0-9]{48,}\b"),        # Together (long hex token)
]


def _redact(text: str) -> str:
    """Strip anything that looks like an API key from a string."""
    for pat in _KEY_PATTERNS:
        text = pat.sub("***", text)
    return text


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def _progress_path(job_id: str) -> Path:
    return _job_dir(job_id) / "progress.jsonl"


def create_job(job_id: str, params: dict[str, Any]) -> None:
    """Initialize a new job directory and meta.json."""
    job_dir = _job_dir(job_id)
    with _lock:
        job_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "id": job_id,
        "status": JobStatus.queued,
        **params,
        "error": None,
    }
    _atomic_write(_meta_path(job_id), json.dumps(meta))
    _progress_path(job_id).write_text("", encoding="utf-8")


def update_status(job_id: str, status: JobStatus, error: str | None = None) -> None:
    """Update the job status (and optionally record an error message)."""
    meta = _read_meta(job_id)
    meta["status"] = status
    if error is not None:
        meta["error"] = _redact(error)
    _atomic_write(_meta_path(job_id), json.dumps(meta))


def append_progress(job_id: str, step: int, total: int, message: str) -> None:
    """Append a progress event to the job's progress log."""
    event = json.dumps({"step": step, "total": total, "message": _redact(message)})
    with open(_progress_path(job_id), "a", encoding="utf-8") as f:
        f.write(event + "\n")


def get_job(job_id: str) -> JobResponse | None:
    """Read job metadata and progress, returning None if the job doesn't exist."""
    meta_path = _meta_path(job_id)
    if not meta_path.exists():
        return None

    try:
        meta = _read_meta(job_id)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read meta for job %s: %s", job_id, exc)
        return None

    progress: list[ProgressEvent] = []
    progress_path = _progress_path(job_id)
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                progress.append(ProgressEvent(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue

    return JobResponse(
        id=meta["id"],
        status=meta["status"],
        language=meta["language"],
        voice=meta["voice"],
        source_language=meta["source_language"],
        subtitles=meta["subtitles"],
        progress=progress,
        error=meta.get("error"),
    )


def input_path(job_id: str) -> Path:
    """Return the path where the uploaded video is stored."""
    job_dir = _job_dir(job_id)
    for p in job_dir.glob("input.*"):
        return p
    raise FileNotFoundError(f"No input file for job {job_id}")


def output_video_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output.mp4"


def voiceover_video_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output_voiceover.mp4"


def output_srt_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output.srt"


def original_audio_path(job_id: str) -> Path:
    return _job_dir(job_id) / "original_audio.m4a"


def segments_path(job_id: str) -> Path:
    return _job_dir(job_id) / "segments.json"


def save_segments(job_id: str, segments: list[dict[str, Any]]) -> None:
    segments_path(job_id).write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_segments(job_id: str) -> list[dict[str, Any]]:
    path = segments_path(job_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def delete_job(job_id: str) -> bool:
    """Remove a job directory. Returns True if the job existed."""
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        return False
    shutil.rmtree(job_dir)
    return True


def cleanup_old_jobs(max_age_hours: float) -> int:
    """Delete completed/failed jobs whose meta.json is older than *max_age_hours*.

    Returns the number of jobs deleted. Skips running/queued jobs.
    """
    if max_age_hours <= 0 or not JOBS_DIR.exists():
        return 0

    cutoff = time.time() - max_age_hours * 3600
    deleted = 0

    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        meta_file = job_dir / "meta.json"
        if not meta_file.exists():
            continue
        try:
            if meta_file.stat().st_mtime > cutoff:
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            status = meta.get("status", "")
            if status not in (JobStatus.done, JobStatus.failed, "done", "failed"):
                continue
            shutil.rmtree(job_dir)
            deleted += 1
        except Exception:
            continue

    return deleted


def _atomic_write(path: Path, data: str) -> None:
    """Write data to a file atomically via temp-file + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, data.encode("utf-8"))
        os.close(fd)
        fd = -1
        os.replace(tmp, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_meta(job_id: str) -> dict[str, Any]:
    """Read meta.json with a single retry in case of a partial-write race."""
    path = _meta_path(job_id)
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            if attempt < 2:
                time.sleep(0.05)
            else:
                raise
