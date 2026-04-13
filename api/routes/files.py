"""File download endpoints for completed jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.models import JobStatus
from api.storage import get_job, original_audio_path, output_srt_path, output_video_path

router = APIRouter(prefix="/jobs", tags=["files"])


def _assert_done(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not complete (status: {job.status}).",
        )


@router.get("/{job_id}/video", response_class=FileResponse)
def download_video(job_id: str):
    """Download the dubbed output video. Only available when status=done."""
    _assert_done(job_id)
    path = output_video_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output video not found.")
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=f"{job_id}_dubbed.mp4",
    )


@router.get("/{job_id}/original-audio")
def get_original_audio(job_id: str):
    """Serve the original audio track (aligned to the dubbed timeline) for voice-over mixing."""
    _assert_done(job_id)
    path = original_audio_path(job_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Original audio track not available. The job may not have used voice-over mode.",
        )
    return FileResponse(
        path=str(path),
        media_type="audio/mp4",
        filename=f"{job_id}_original.m4a",
    )


@router.get("/{job_id}/srt", response_class=FileResponse)
def download_srt(job_id: str):
    """Download the SRT subtitle file. Only available when status=done and subtitles=true."""
    _assert_done(job_id)
    path = output_srt_path(job_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="SRT file not found. The job may have been created with subtitles=false.",
        )
    return FileResponse(
        path=str(path),
        media_type="text/plain; charset=utf-8",
        filename=f"{job_id}.srt",
    )
