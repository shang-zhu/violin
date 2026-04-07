"""Pydantic models for API requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class ProgressEvent(BaseModel):
    step: int
    total: int
    message: str


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    language: str
    voice: str
    source_language: str
    diarize: bool
    subtitles: bool
    progress: list[ProgressEvent] = Field(default_factory=list)
    error: str | None = None


class CreateJobRequest(BaseModel):
    """Used internally — the route uses Form() fields directly."""
    language: str
    voice: str = ""
    source_language: str = "auto-detect"
    diarize: bool = True
    subtitles: bool = True
