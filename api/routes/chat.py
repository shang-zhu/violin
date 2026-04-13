"""Video chat endpoints for completed jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models import SubtitleSegment, VideoChatRequest, VideoChatResponse
from api.storage import get_job, load_segments
from api.video_chat import answer_video_question

router = APIRouter(prefix="/jobs", tags=["chat"])


@router.get("/{job_id}/segments", response_model=list[SubtitleSegment])
def get_job_segments(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' is not complete.")
    return [SubtitleSegment(**item) for item in load_segments(job_id)]


@router.post("/{job_id}/chat", response_model=VideoChatResponse)
def chat_with_video(job_id: str, payload: VideoChatRequest):
    try:
        return answer_video_question(
            job_id=job_id,
            question=payload.question,
            current_time=payload.current_time,
            history=payload.history,
            language=payload.language,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
