"""Background worker that runs the translation pipeline in a thread pool."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from together import Together

from pipeline.extractor import extract_audio, get_video_duration
from pipeline.languages import language_code
from pipeline.merger import build_aligned_video, build_gap_chunks, generate_srt, prepare_merge
from pipeline.styles import resolve as resolve_style
from pipeline.transcriber import find_main_speaker, merge_continuous_segments, transcribe
from pipeline.translator import translate_segments
from pipeline.tts import native_voices_for, synthesize_segments

from .config import MAX_WORKERS
from .models import JobStatus
from .storage import (
    append_progress,
    input_path,
    original_audio_path,
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

    target_language = params["language"]
    voice = params["voice"]
    source_language = params["source_language"]
    diarize = params["diarize"]
    subtitles = params["subtitles"]
    voiceover = params.get("voiceover", True)
    style = resolve_style(params.get("style", "standard"))

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

            label = "Transcribing with Whisper Large v3…"
            if diarize:
                label += " (+ diarization)"
            _progress(job_id, 2, f"{label} (video duration: {total_duration:.0f}s)")
            segments = transcribe(audio_path, client, diarize=diarize)

            lang_code = language_code(target_language)

            if diarize:
                main_speaker = find_main_speaker(segments)
                segments = [s for s in segments if s.speaker == main_speaker]

            segments = merge_continuous_segments(segments)

            _progress(job_id, 3, f"Translating {len(segments)} segments to {target_language} "
                       f"(style: {style.name})…")
            translated = translate_segments(
                segments, target_language, client, source_language,
                style_directives=style.translation_directives,
                style_temperature=style.temperature,
            )
            translated = merge_continuous_segments(translated)

            effective_voice = voice or native_voices_for(lang_code)[0]
            _progress(job_id, 4, f"Synthesizing TTS with Cartesia Sonic 3 (voice: {effective_voice})…")
            tts_dir = tmp_dir / "tts"
            tts_dir.mkdir()

            plan = prepare_merge(
                str(src), translated, total_duration,
                preserve_gap_audio=diarize or voiceover,
                original_audio_volume=1.0 if voiceover else 0.0,
            )
            gap_exc: list[Exception] = []

            def _build_gaps():
                try:
                    build_gap_chunks(plan)
                except Exception as e:
                    gap_exc.append(e)

            gap_thread = threading.Thread(target=_build_gaps, daemon=True)
            gap_thread.start()

            tts_paths = synthesize_segments(
                translated, effective_voice, str(tts_dir), client,
                language=lang_code,
                speed=style.tts_speed, emotion=style.tts_emotion,
            )
            gap_thread.join()
            if gap_exc:
                raise gap_exc[0]

            _progress(job_id, 5, "Building aligned video…")
            orig_audio = str(original_audio_path(job_id)) if voiceover else None
            aligned_segments = build_aligned_video(
                str(src), translated, tts_paths, total_duration, str(out_video),
                merge_plan=plan,
                original_audio_path=orig_audio,
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
