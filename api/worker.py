"""Background worker that runs the translation pipeline in a thread pool."""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from together import Together

from pipeline.diarizer import assign_speakers
from pipeline.extractor import extract_audio, get_video_duration
from pipeline.languages import language_code
from pipeline.merger import build_aligned_video, generate_srt
from pipeline.transcriber import merge_continuous_segments, transcribe
from pipeline.translator import translate_segments
from pipeline.tts import native_voices_for, synthesize_segments

from .config import MAX_WORKERS
from .models import JobStatus
from .storage import (
    append_progress,
    input_path,
    output_srt_path,
    output_video_path,
    save_segments,
    update_status,
)

load_dotenv()

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

TOTAL_STEPS = 5


def _progress(job_id: str, step: int, message: str) -> None:
    append_progress(job_id, step, TOTAL_STEPS, message)


def _run_job(job_id: str, params: dict) -> None:
    update_status(job_id, JobStatus.running)

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        update_status(job_id, JobStatus.failed, "TOGETHER_API_KEY is not configured.")
        return

    hf_token = os.environ.get("HF_TOKEN")
    target_language = params["language"]
    voice = params["voice"]
    source_language = params["source_language"]
    diarize = params["diarize"]
    subtitles = params["subtitles"]

    if diarize and not hf_token:
        update_status(job_id, JobStatus.failed, "HF_TOKEN is not configured; cannot run diarization.")
        return

    try:
        src = input_path(job_id)
        out_video = output_video_path(job_id)
        out_srt = output_srt_path(job_id)

        client = Together(api_key=api_key)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"vidtrans_{job_id}_"))

        try:
            _progress(job_id, 1, "Extracting audio…")
            audio_path = extract_audio(str(src), str(tmp_dir / "audio.wav"))
            total_duration = get_video_duration(str(src))

            _progress(job_id, 2, f"Transcribing with Whisper Large v3… (video duration: {total_duration:.0f}s)")
            segments = transcribe(audio_path, client)

            lang_code = language_code(target_language)
            voice_map: dict[str, str] | None = None

            if diarize:
                _progress(job_id, 2, "Diarizing speakers with pyannote…")
                segments, voice_map = assign_speakers(segments, audio_path, hf_token, lang_code)

            segments = merge_continuous_segments(segments)
            _progress(job_id, 3, f"Translating {len(segments)} segments to {target_language}…")
            translated = translate_segments(segments, target_language, client, source_language)
            translated = merge_continuous_segments(translated)

            effective_voice = voice or native_voices_for(lang_code)[0]
            _progress(job_id, 4, f"Synthesizing TTS with Cartesia Sonic 3 (voice: {effective_voice})…")
            tts_dir = tmp_dir / "tts"
            tts_dir.mkdir()
            tts_paths = synthesize_segments(
                translated, effective_voice, str(tts_dir), client,
                language=lang_code, voice_map=voice_map,
            )

            _progress(job_id, 5, "Building aligned video…")
            aligned_segments = build_aligned_video(
                str(src), translated, tts_paths, total_duration, str(out_video),
            )
            save_segments(
                job_id,
                [
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "speaker": seg.speaker,
                    }
                    for seg in aligned_segments
                ],
            )

            if subtitles:
                generate_srt(aligned_segments, str(out_srt))

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        update_status(job_id, JobStatus.done)

    except Exception as exc:
        update_status(job_id, JobStatus.failed, str(exc))


def submit_job(job_id: str, params: dict) -> None:
    """Submit a job to the thread pool for background execution."""
    _executor.submit(_run_job, job_id, params)
