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
    cancelled = "cancelled"


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
    subtitles: bool
    style: str = "standard"
    progress: list[ProgressEvent] = Field(default_factory=list)
    error: str | None = None


class CreateJobRequest(BaseModel):
    """Used internally — the route uses Form() fields directly."""
    language: str
    voice: str = ""
    source_language: str = "auto-detect"
    subtitles: bool = True


class SubtitleSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"


class ChatMessage(BaseModel):
    role: str
    content: str


class VoiceMatchRequest(BaseModel):
    description: str
    language: str = ""
    together_api_key: str = ""
    openai_api_key: str = ""


class VoiceCandidate(BaseModel):
    voice: str
    explanation: str


class VoiceMatchResponse(BaseModel):
    candidates: list[VoiceCandidate]


class VideoChatRequest(BaseModel):
    question: str
    current_time: float = Field(ge=0)
    history: list[ChatMessage] = Field(default_factory=list)
    language: str = ""


class VideoChatResponse(BaseModel):
    answer: str
    context_start: float
    context_end: float
    subtitle_context: list[SubtitleSegment] = Field(default_factory=list)
    sampled_timestamps: list[float] = Field(default_factory=list)
    style: str = "standard"
