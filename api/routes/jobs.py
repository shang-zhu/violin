"""Job lifecycle endpoints: create, status, delete."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.models import JobResponse, JobStatus
from api.storage import create_job, delete_job, get_job, input_path, output_video_path
from api.worker import submit_job

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Allowed video MIME types / extensions
_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@router.post("", response_model=JobResponse, status_code=202)
async def create_translation_job(
    file: UploadFile,
    language: str = Form(..., description="Target language name, e.g. Spanish, Japanese"),
    voice: str = Form("", description="Cartesia Sonic 3 voice (empty = auto native voice)"),
    source_language: str = Form("auto-detect", description="Source language hint for translation"),
    subtitles: bool = Form(True, description="Generate SRT subtitle file"),
    style: str = Form("standard", description="Translation style profile (e.g. standard, kids, academic)"),
    voiceover: bool = Form(True, description="Voice-over mode: keep original audio underneath the dub"),
):
    """Upload a video and start a translation job. Returns immediately with a job ID."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
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

    # Persist metadata first so GET /jobs/{id} works immediately
    create_job(job_id, params)

    # Save uploaded file
    from api.storage import _job_dir  # local import to avoid circular
    dest = _job_dir(job_id) / f"input{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    submit_job(job_id, params)

    job = get_job(job_id)
    return job


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    """Poll a job's current status and progress log."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.delete("/{job_id}", status_code=204)
def delete_translation_job(job_id: str):
    """Delete a job and all its associated files."""
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
