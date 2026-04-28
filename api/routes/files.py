"""File download endpoints for completed jobs."""

from __future__ import annotations

import subprocess

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from api.models import JobStatus
from api.storage import get_job, original_audio_path, output_srt_path, output_video_path, voiceover_video_path
from pipeline.ffmpeg_utils import FFMPEG_EXE

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


@router.get("/{job_id}/video-voiceover", response_class=FileResponse)
def download_voiceover_video(
    job_id: str,
    volume: float = Query(0.1, ge=0.0, le=1.0, description="Original audio volume (0.0–1.0)"),
):
    """Download the dubbed video with original audio mixed in at the given volume."""
    _assert_done(job_id)

    dubbed = output_video_path(job_id)
    orig_audio = original_audio_path(job_id)
    if not dubbed.exists():
        raise HTTPException(status_code=404, detail="Output video not found.")
    if not orig_audio.exists():
        raise HTTPException(status_code=404, detail="Original audio not available. Job may not have used voice-over mode.")

    out = voiceover_video_path(job_id)
    if not out.exists() or not _volume_matches(job_id, volume):
        _mix_voiceover(str(dubbed), str(orig_audio), str(out), volume)
        _save_volume(job_id, volume)

    return FileResponse(
        path=str(out),
        media_type="video/mp4",
        filename=f"{job_id}_voiceover.mp4",
    )


def _mix_voiceover(video_path: str, audio_path: str, output_path: str, volume: float) -> None:
    """Mix original audio into the dubbed video using ffmpeg."""
    subprocess.run([
        FFMPEG_EXE,
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        f"[0:a]volume=1.0[dub];[1:a]volume={volume}[orig];[dub][orig]amix=inputs=2:duration=first[out]",
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y", output_path,
    ], check=True, capture_output=True)


def _volume_matches(job_id: str, volume: float) -> bool:
    """Check if the cached voiceover was mixed at the same volume."""
    vol_file = voiceover_video_path(job_id).with_suffix(".vol")
    if not vol_file.exists():
        return False
    try:
        return abs(float(vol_file.read_text().strip()) - volume) < 0.001
    except (ValueError, OSError):
        return False


def _save_volume(job_id: str, volume: float) -> None:
    vol_file = voiceover_video_path(job_id).with_suffix(".vol")
    vol_file.write_text(str(volume))


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
