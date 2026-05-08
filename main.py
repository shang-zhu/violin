"""
Violin CLI

Usage:
    uv run main.py <input_video> <output_video> --language <target_language>

Examples:
    uv run main.py lecture.mp4 lecture_es.mp4 --language Spanish
    uv run main.py lesson.mp4 lesson_ja.mp4 --language Japanese
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
from pipeline.llm_client import make_transcription_client, make_translation_client
from pipeline.styles import StyleProfile, list_styles, resolve as resolve_style
from pipeline.tts import native_voices_for
from pipeline.extractor import extract_audio, get_video_duration
from pipeline.merger import build_aligned_video, build_gap_chunks, generate_srt, prepare_merge
from pipeline.transcriber import merge_continuous_segments, transcribe
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
    style: StyleProfile | None = None,
    voiceover: bool = True,
    timings_out: str | None = None,
) -> None:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")

    client = Together(api_key=api_key)
    cfg = pipeline_config.get()
    translation_client = make_translation_client(cfg)
    transcription_client = make_transcription_client(cfg)
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

        print("\n[2/5] Transcribing with Whisper Large v3...")
        segments = transcribe(audio_path, transcription_client)
        print(f"      {len(segments)} segments found")
        tracker.audio_minutes = total_duration / 60.0
        tracker.record_step("Transcription (Whisper)")

        lang_code = _language_code(target_language)

        raw_count = len(segments)
        segments = merge_continuous_segments(segments)
        print(f"      Merged {raw_count} → {len(segments)} segments (sentence grouping)")

        print(f"\n[3/5] Translating to {target_language}...")
        translated = translate_segments(
            segments, target_language, translation_client, source_language,
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

        vo_volume = pipeline_config.get()["merge_video"].get("voiceover_volume", 0.35)
        gap_vol = min(1.0, 2 * vo_volume) if voiceover else 1.0
        plan = prepare_merge(
            input_path, translated, total_duration,
            preserve_gap_audio=voiceover,
            mix_volume=vo_volume if voiceover else 0.0,
            gap_volume=gap_vol,
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
        orig_audio_out = None
        if voiceover:
            out_p = Path(output_path)
            orig_audio_out = str(out_p.with_stem(out_p.stem + "_original").with_suffix(".m4a"))
        aligned_segments = build_aligned_video(
            input_path, translated, tts_paths, total_duration, output_path,
            merge_plan=plan,
            original_audio_path=orig_audio_out,
        )
        if orig_audio_out:
            print(f"      Original audio → {orig_audio_out}")

        if subtitles:
            srt_path = Path(output_path).with_suffix(".srt")
            generate_srt(aligned_segments, str(srt_path))
            print(f"      Subtitles → {srt_path}")

        print(f"\nDone! Output → {output_path}")
        tracker.print_summary()

        if timings_out:
            import json
            steps = list(tracker._steps)
            payload = {
                "total": sum(s["elapsed"] for s in steps),
                "steps": steps,
                "cost": tracker.cost_breakdown(),
            }
            Path(timings_out).write_text(json.dumps(payload, indent=2) + "\n")
            print(f"      Timings → {timings_out}")

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
        "--voiceover", action="store_true", default=None,
        help="Voice-over mode: keep original audio underneath the dub (default)"
    )
    parser.add_argument(
        "--no-voiceover", action="store_true", default=None,
        help="Full replacement: dubbed audio only, no original audio"
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
    parser.add_argument(
        "--timings-out", default=None,
        help="Write per-step wall-clock timings as JSON to this path on success"
    )

    args = parser.parse_args()

    pipeline_config.load(args.config)

    if args.style == "list":
        _print_styles()
        sys.exit(0)

    if not args.input or not args.output or not args.language:
        parser.error("input, output, and --language are required (unless using --style list)")

    if args.no_voiceover:
        voiceover = False
    elif args.voiceover:
        voiceover = True
    else:
        voiceover = True

    style_name = args.style or pipeline_config.get()["pipeline"].get("style", "standard")
    style = resolve_style(style_name)

    translate_video(
        args.input,
        args.output,
        args.language,
        args.voice,
        not args.no_subtitles,
        args.source_language,
        style,
        voiceover,
        timings_out=args.timings_out,
    )


if __name__ == "__main__":
    main()
