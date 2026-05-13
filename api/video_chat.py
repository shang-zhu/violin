"""Frame-aware chat helpers — uses the same LLM provider as translation."""

from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from api.models import ChatMessage, SubtitleSegment, VideoChatResponse
from api.storage import get_job, load_segments, output_video_path
from pipeline import config as _conf
from pipeline.ffmpeg_utils import FFMPEG_EXE, get_duration_video
from pipeline.llm_client import get_chat_model, make_chat_client
import prompts as _prompts

load_dotenv(override=True)


def _chat_cfg() -> dict:
    cfg = _conf.get()
    return {**cfg["chat"], "model": get_chat_model(cfg)}


def _pick_context_window(segments: list[SubtitleSegment], current_time: float) -> tuple[float, float, list[SubtitleSegment]]:
    half_window = _chat_cfg()["context_window_seconds"] / 2
    window_start = max(0.0, current_time - half_window)
    window_end = current_time + half_window

    selected = [
        segment
        for segment in segments
        if segment.end >= window_start and segment.start <= window_end
    ]

    if selected:
        window_start = min(window_start, selected[0].start)
        window_end = max(window_end, selected[-1].end)

    return window_start, window_end, selected


def _sample_timestamps(video_duration: float, window_start: float, window_end: float) -> list[float]:
    cfg = _chat_cfg()
    interval = max(1, int(cfg["frame_interval_seconds"]))
    max_frames = max(1, int(cfg["max_frames"]))

    timestamps: list[float] = []
    t = window_start
    while t <= window_end and len(timestamps) < max_frames:
        timestamps.append(round(min(max(t, 0.0), max(video_duration - 0.1, 0.0)), 2))
        t += interval

    if not timestamps:
        timestamps.append(round(min(max(window_start, 0.0), max(video_duration - 0.1, 0.0)), 2))

    return timestamps


def _frame_as_data_url(video_path: Path, timestamp: float) -> str | None:
    """Extract a single frame as a base64 data URL. Returns None on failure."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                FFMPEG_EXE,
                "-ss",
                str(timestamp),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "4",
                "-y",
                str(frame_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0 or not frame_path.exists() or frame_path.stat().st_size == 0:
            return None
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None
    finally:
        frame_path.unlink(missing_ok=True)


def _build_messages(
    history: list[ChatMessage],
    question: str,
    subtitle_context: list[SubtitleSegment],
    frame_urls: list[str],
    current_time: float,
    window_start: float,
    window_end: float,
    language: str = "",
) -> list[dict]:
    transcript_lines = [
        f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}"
        for segment in subtitle_context
    ]
    transcript_block = "\n".join(transcript_lines) or "(no subtitle lines found in this window)"

    lang_instruction = ""
    if language:
        lang_instruction = f"You understand {language}. Reply in whatever language the user writes in. "

    system_content = _prompts.load("video_chat", "system", lang_instruction=lang_instruction)

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for item in history[-8:]:
        if item.role not in {"user", "assistant"}:
            continue
        messages.append({"role": item.role, "content": item.content})

    user_text = _prompts.load(
        "video_chat", "user",
        current_time=f"{current_time:.2f}",
        window_start=f"{window_start:.2f}",
        window_end=f"{window_end:.2f}",
        transcript_block=transcript_block,
        num_frames=len(frame_urls),
        question=question,
    )

    user_content: list[dict] = [{"type": "text", "text": user_text}]
    user_content.extend({"type": "image_url", "image_url": {"url": url}} for url in frame_urls)
    messages.append({"role": "user", "content": user_content})
    return messages


def answer_video_question(job_id: str, question: str, current_time: float, history: list[ChatMessage], language: str = "") -> VideoChatResponse:
    job = get_job(job_id)
    if job is None:
        raise FileNotFoundError(f"Job '{job_id}' not found.")
    if job.status != "done":
        raise RuntimeError(f"Job '{job_id}' is not complete.")

    video_path = output_video_path(job_id)
    if not video_path.exists():
        raise FileNotFoundError("Output video not found.")

    segments = [SubtitleSegment(**item) for item in load_segments(job_id)]
    window_start, window_end, subtitle_context = _pick_context_window(segments, current_time)
    duration = get_duration_video(str(video_path))
    sampled_timestamps = _sample_timestamps(duration, window_start, window_end)
    frame_urls = [
        url for ts in sampled_timestamps
        if (url := _frame_as_data_url(video_path, ts)) is not None
    ]

    client = make_chat_client(_conf.get())
    cfg = _chat_cfg()
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=_build_messages(
            history=history,
            question=question,
            subtitle_context=subtitle_context,
            frame_urls=frame_urls,
            current_time=current_time,
            window_start=window_start,
            window_end=window_end,
            language=language,
        ),
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    answer = response.choices[0].message.content.strip()
    return VideoChatResponse(
        answer=answer,
        context_start=window_start,
        context_end=window_end,
        subtitle_context=subtitle_context,
        sampled_timestamps=sampled_timestamps,
    )
