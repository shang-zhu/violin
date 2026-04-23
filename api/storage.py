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
import threading
from pathlib import Path
from typing import Any

from .config import JOBS_DIR
from .models import JobResponse, JobStatus, ProgressEvent

_lock = threading.Lock()  # guards directory creation; individual files use atomic writes


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
    _meta_path(job_id).write_text(json.dumps(meta), encoding="utf-8")
    _progress_path(job_id).write_text("", encoding="utf-8")


def update_status(job_id: str, status: JobStatus, error: str | None = None) -> None:
    """Update the job status (and optionally record an error message)."""
    meta = _read_meta(job_id)
    meta["status"] = status
    if error is not None:
        meta["error"] = error
    _meta_path(job_id).write_text(json.dumps(meta), encoding="utf-8")


def append_progress(job_id: str, step: int, total: int, message: str) -> None:
    """Append a progress event to the job's progress log."""
    event = json.dumps({"step": step, "total": total, "message": message})
    with open(_progress_path(job_id), "a", encoding="utf-8") as f:
        f.write(event + "\n")


def get_job(job_id: str) -> JobResponse | None:
    """Read job metadata and progress, returning None if the job doesn't exist."""
    meta_path = _meta_path(job_id)
    if not meta_path.exists():
        return None

    meta = _read_meta(job_id)

    progress: list[ProgressEvent] = []
    progress_path = _progress_path(job_id)
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                progress.append(ProgressEvent(**json.loads(line)))

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
    import shutil
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        return False
    shutil.rmtree(job_dir)
    return True


def _read_meta(job_id: str) -> dict[str, Any]:
    return json.loads(_meta_path(job_id).read_text(encoding="utf-8"))
