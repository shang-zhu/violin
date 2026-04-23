"""Job lifecycle endpoints: create, status, delete."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from api.models import JobResponse, JobStatus
from api.storage import create_job, delete_job, get_job, input_path, output_video_path
from api.usage import has_free_trial, record_usage, remaining_trials
from api.worker import submit_job, submit_url_job

router = APIRouter(prefix="/jobs", tags=["jobs"])

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_MAX_DURATION_SECONDS = 1800


def _client_ip(request: Request) -> str:
    """Extract the real client IP, respecting X-Forwarded-For from Caddy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("", response_model=JobResponse, status_code=202)
async def create_translation_job(
    request: Request,
    file: UploadFile,
    language: str = Form(..., description="Target language name, e.g. Spanish, Japanese"),
    voice: str = Form("", description="Cartesia Sonic 3 voice (empty = auto native voice)"),
    source_language: str = Form("auto-detect", description="Source language hint for translation"),
    subtitles: bool = Form(True, description="Generate SRT subtitle file"),
    style: str = Form("standard", description="Translation style profile (e.g. standard, kids, academic)"),
    voiceover: bool = Form(True, description="Voice-over mode: keep original audio underneath the dub"),
    user_api_key: str = Form("", description="User-provided Together API key (optional)"),
):
    """Upload a video and start a translation job. Returns immediately with a job ID."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    client_ip = _client_ip(request)
    using_own_key = bool(user_api_key.strip())

    if not using_own_key and not has_free_trial(client_ip):
        raise HTTPException(
            status_code=403,
            detail="Free trial used. Please provide your own Together API key to continue.",
        )

    job_id = uuid.uuid4().hex
    params = {
        "language": language,
        "voice": voice,
        "source_language": source_language,
        "subtitles": subtitles,
        "style": style,
        "voiceover": voiceover,
    }

    create_job(job_id, params)

    from api.storage import _job_dir
    dest = _job_dir(job_id) / f"input{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    api_key_override = user_api_key.strip() or None
    submit_job(job_id, params, api_key_override=api_key_override)

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to read job after creation.")

    if not using_own_key:
        record_usage(client_ip)

    return job


class UrlJobRequest(BaseModel):
    url: str
    language: str
    voice: str = ""
    source_language: str = "auto-detect"
    subtitles: bool = True
    style: str = "standard"
    voiceover: bool = True
    user_api_key: str = ""


@router.post("/from-url", response_model=JobResponse, status_code=202)
async def create_job_from_url(request: Request, body: UrlJobRequest):
    """Create a translation job from a video URL (YouTube, etc.)."""
    if not body.url.strip():
        raise HTTPException(status_code=400, detail="URL is required.")

    client_ip = _client_ip(request)
    using_own_key = bool(body.user_api_key.strip())

    if not using_own_key and not has_free_trial(client_ip):
        raise HTTPException(
            status_code=403,
            detail="Free trial used. Please provide your own Together API key to continue.",
        )

    job_id = uuid.uuid4().hex
    params = {
        "language": body.language,
        "voice": body.voice,
        "source_language": body.source_language,
        "subtitles": body.subtitles,
        "style": body.style,
        "voiceover": body.voiceover,
    }

    create_job(job_id, params)

    api_key_override = body.user_api_key.strip() or None
    submit_url_job(job_id, params, body.url.strip(), api_key_override=api_key_override)

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to read job after creation.")

    if not using_own_key:
        record_usage(client_ip)

    return job


@router.get("/trial-status")
async def trial_status(request: Request):
    """Check how many free trial jobs remain for this client IP."""
    client_ip = _client_ip(request)
    remaining = remaining_trials(client_ip)
    return {"remaining": remaining, "needs_key": remaining == 0}


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    """Poll a job's current status and progress log."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.post("/{job_id}/cancel")
def cancel_translation_job(job_id: str):
    """Cancel a running job."""
    from api.storage import update_status
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status not in (JobStatus.queued, JobStatus.running):
        raise HTTPException(status_code=409, detail=f"Job is already {job.status.value}.")
    update_status(job_id, JobStatus.cancelled)
    return {"status": "cancelled"}


@router.delete("/{job_id}", status_code=204)
def delete_translation_job(job_id: str):
    """Delete a job and all its associated files."""
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
