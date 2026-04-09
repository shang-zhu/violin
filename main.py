"""
VideoLingua CLI

Usage:
    uv run main.py <input_video> <output_video> --language <target_language>

Examples:
    uv run main.py lecture.mp4 lecture_es.mp4 --language Spanish
    uv run main.py lesson.mp4 lesson_ja.mp4 --language Japanese
    uv run main.py talk.mp4 talk_fr.mp4 --language French --no-diarize
    uv run main.py talk.mp4 talk_zh.mp4 --language Chinese --style kids
"""

import argparse
import os
import shutil
import sys
import tempfile
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from together import Together

from pipeline import config as pipeline_config
from pipeline.costs import CostTracker
from pipeline.languages import language_code as _language_code
from pipeline.styles import StyleProfile, list_styles, resolve as resolve_style
from pipeline.tts import native_voices_for
from pipeline.extractor import extract_audio, get_video_duration
from pipeline.merger import build_aligned_video, build_gap_chunks, generate_srt, prepare_merge
from pipeline.transcriber import find_main_speaker, merge_continuous_segments, transcribe
from pipeline.translator import translate_segments
from pipeline.tts import synthesize_segments

load_dotenv()


def _print_styles() -> None:
    """Print available style profiles and exit."""
    styles = list_styles()
    if not styles:
        print("No styles defined in config.")
        return
    print("Available styles:\n")
    for s in styles:
        print(f"  {s.name:14s}  {s.description}")
        parts = []
        if s.tts_speed is not None:
            parts.append(f"speed={s.tts_speed}")
        if s.tts_emotion:
            parts.append(f"emotion={s.tts_emotion}")
        if parts:
            print(f"  {'':14s}  TTS: {', '.join(parts)}")


def translate_video(
    input_path: str,
    output_path: str,
    target_language: str,
    voice: str = "tutorial man",
    subtitles: bool = True,
    source_language: str = "auto-detect",
    diarize: bool = True,
    style: StyleProfile | None = None,
) -> None:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")

    client = Together(api_key=api_key)
    tmp_dir = Path(tempfile.mkdtemp(prefix="vidtrans_"))
    tracker = CostTracker()

    if style is None:
        style = resolve_style("standard")

    try:
        tracker.start_timer()

        if style.name != "standard":
            print(f"\n  Style: {style.name} — \"{style.description}\"")

        print(f"\n[1/5] Extracting audio from {input_path}...")
        audio_path = extract_audio(input_path, str(tmp_dir / "audio.wav"))
        total_duration = get_video_duration(input_path)
        print(f"      Duration: {total_duration:.1f}s")
        tracker.record_step("Audio extraction")

        step_label = "Transcribing with Whisper Large v3"
        if diarize:
            step_label += " + speaker diarization"
        print(f"\n[2/5] {step_label}...")
        segments = transcribe(audio_path, client, diarize=diarize)
        print(f"      {len(segments)} segments found")
        tracker.audio_minutes = total_duration / 60.0
        tracker.record_step("Transcription (Whisper)")

        lang_code = _language_code(target_language)

        if diarize:
            main_speaker = find_main_speaker(segments)
            all_segments = segments
            main_segments = [s for s in segments if s.speaker == main_speaker]

            speaker_durations: dict[str, float] = {}
            for seg in all_segments:
                speaker_durations[seg.speaker] = (
                    speaker_durations.get(seg.speaker, 0) + (seg.end - seg.start)
                )
            print(f"      Speakers detected: {len(speaker_durations)}")
            for spk, dur in sorted(speaker_durations.items(), key=lambda x: -x[1]):
                marker = " <-- main" if spk == main_speaker else ""
                print(f"        {spk}: {dur:.1f}s{marker}")
            print(f"      Translating {len(main_segments)} segments from {main_speaker}")
            print(f"      Preserving {len(all_segments) - len(main_segments)} segments "
                  "from other speakers (original audio)")

            segments = main_segments

        raw_count = len(segments)
        segments = merge_continuous_segments(segments)
        print(f"      Merged {raw_count} → {len(segments)} segments (sentence grouping)")

        print(f"\n[3/5] Translating to {target_language}...")
        translated = translate_segments(
            segments, target_language, client, source_language,
            tracker=tracker, style_directives=style.translation_directives,
            style_temperature=style.temperature,
        )
        tracker.record_step("Translation (LLM)")

        pre_merge = len(translated)
        translated = merge_continuous_segments(translated)
        if len(translated) < pre_merge:
            print(f"      Post-translation merge: {pre_merge} → {len(translated)} segments")

        print("\n[4/5] Synthesizing dubbed audio with Cartesia Sonic 3...")
        tts_dir = tmp_dir / "tts"
        tts_dir.mkdir()
        if voice != "tutorial man":
            effective_voice = voice
        else:
            gender_idx = 0 if pipeline_config.get()["pipeline"]["voice_gender"] == "male" else 1
            effective_voice = native_voices_for(lang_code)[gender_idx]

        plan = prepare_merge(
            input_path, translated, total_duration,
            preserve_gap_audio=diarize,
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
            language=lang_code, tracker=tracker,
            speed=style.tts_speed, emotion=style.tts_emotion,
        )
        gap_thread.join()
        if gap_exc:
            raise gap_exc[0]
        tracker.record_step("TTS (Cartesia Sonic 3)")

        print("\n[5/5] Building aligned video...")
        aligned_segments = build_aligned_video(
            input_path, translated, tts_paths, total_duration, output_path,
            merge_plan=plan,
        )
        tracker.record_step("Video alignment & merge")

        if subtitles:
            srt_path = Path(output_path).with_suffix(".srt")
            generate_srt(aligned_segments, str(srt_path))
            print(f"      Subtitles → {srt_path}")

        print(f"\nDone! Output → {output_path}")
        tracker.print_summary()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate a video to another language using Together AI."
    )
    parser.add_argument("input", nargs="?", help="Input video file path")
    parser.add_argument("output", nargs="?", help="Output video file path")
    parser.add_argument(
        "--language", "-l", default=None,
        help="Target language (e.g. Spanish, French, Japanese, Arabic)"
    )
    parser.add_argument(
        "--voice", "-v", default="tutorial man",
        help='Cartesia Sonic 3 voice for translated speech (default: "tutorial man")'
    )
    parser.add_argument(
        "--source-language", default="auto-detect",
        help="Source language hint for translation (default: auto-detect)"
    )
    parser.add_argument(
        "--no-subtitles", action="store_true",
        help="Skip generating SRT subtitle file"
    )
    parser.add_argument(
        "--diarize", action="store_true", default=None,
        help="Enable speaker diarization — only translate the main speaker"
    )
    parser.add_argument(
        "--no-diarize", action="store_true", default=None,
        help="Translate all speakers (no diarization)"
    )
    parser.add_argument(
        "--style", "-s", default=None,
        help='Translation style profile (e.g. standard, kids, academic, casual). '
             'Use "--style list" to see all available styles.'
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a YAML config file (overrides config/default.yaml)"
    )

    args = parser.parse_args()

    pipeline_config.load(args.config)

    if args.style == "list":
        _print_styles()
        sys.exit(0)

    if not args.input or not args.output or not args.language:
        parser.error("input, output, and --language are required (unless using --style list)")

    if args.diarize:
        diarize = True
    elif args.no_diarize:
        diarize = False
    else:
        diarize = pipeline_config.get()["pipeline"]["diarize"]

    style_name = args.style or pipeline_config.get()["pipeline"].get("style", "standard")
    style = resolve_style(style_name)

    translate_video(
        args.input,
        args.output,
        args.language,
        args.voice,
        not args.no_subtitles,
        args.source_language,
        diarize,
        style,
    )


if __name__ == "__main__":
    main()
