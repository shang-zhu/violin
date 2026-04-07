"""
video-translate CLI

Usage:
    uv run main.py <input_video> <output_video> --language <target_language>

Examples:
    uv run main.py lecture.mp4 lecture_es.mp4 --language Spanish
    uv run main.py lesson.mp4 lesson_ja.mp4 --language Japanese
    uv run main.py talk.mp4 talk_fr.mp4 --language French --no-diarize
"""

import argparse
import os
import shutil
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from together import Together

from pipeline import config as pipeline_config
from pipeline.costs import CostTracker
from pipeline.diarizer import assign_speakers
from pipeline.languages import language_code as _language_code
from pipeline.tts import native_voices_for
from pipeline.extractor import extract_audio, get_video_duration
from pipeline.merger import build_aligned_video, generate_srt
from pipeline.transcriber import merge_continuous_segments, transcribe
from pipeline.translator import translate_segments
from pipeline.tts import synthesize_segments

load_dotenv()


def translate_video(
    input_path: str,
    output_path: str,
    target_language: str,
    voice: str = "tutorial man",
    subtitles: bool = True,
    source_language: str = "auto-detect",
    diarize: bool = True,
) -> None:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")

    hf_token = os.environ.get("HF_TOKEN")
    if diarize and not hf_token:
        raise RuntimeError("HF_TOKEN is required for speaker diarization. Set it in .env or use --no-diarize.")

    client = Together(api_key=api_key)
    tmp_dir = Path(tempfile.mkdtemp(prefix="vidtrans_"))
    tracker = CostTracker()

    try:
        tracker.start_timer()

        print(f"\n[1/5] Extracting audio from {input_path}...")
        audio_path = extract_audio(input_path, str(tmp_dir / "audio.wav"))
        total_duration = get_video_duration(input_path)
        print(f"      Duration: {total_duration:.1f}s")
        tracker.record_step("Audio extraction")

        print("\n[2/5] Transcribing with Whisper Large v3...")
        segments = transcribe(audio_path, client)
        print(f"      {len(segments)} segments found")
        tracker.audio_minutes = total_duration / 60.0
        tracker.record_step("Transcription (Whisper)")

        lang_code = _language_code(target_language)

        if diarize:
            print("\n[2.5] Diarizing speakers with pyannote...")
            segments, voice_map = assign_speakers(segments, audio_path, hf_token, lang_code)
            for speaker, spk_voice in voice_map.items():
                print(f"      {speaker} → \"{spk_voice}\"")
            tracker.record_step("Diarization (pyannote)")
        else:
            voice_map = None

        raw_count = len(segments)
        segments = merge_continuous_segments(segments)
        print(f"      Merged {raw_count} → {len(segments)} segments (sentence grouping)")

        print(f"\n[3/5] Translating to {target_language} with Qwen3.5-397B...")
        translated = translate_segments(
            segments, target_language, client, source_language, tracker=tracker,
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
        tts_paths = synthesize_segments(
            translated, effective_voice, str(tts_dir), client,
            language=lang_code, voice_map=voice_map, tracker=tracker,
        )
        tracker.record_step("TTS (Cartesia Sonic 3)")

        print("\n[5/5] Building aligned video...")
        aligned_segments = build_aligned_video(
            input_path, translated, tts_paths, total_duration, output_path,
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
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    parser.add_argument(
        "--language", "-l", required=True,
        help="Target language (e.g. Spanish, French, Japanese, Arabic)"
    )
    parser.add_argument(
        "--voice", "-v", default="tutorial man",
        help='Fallback Cartesia Sonic 3 voice when diarization is off (default: "tutorial man")'
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
        help="Enable speaker diarization (overrides config)"
    )
    parser.add_argument(
        "--no-diarize", action="store_true", default=None,
        help="Skip speaker diarization and use a single voice for all segments"
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a YAML config file (overrides config/default.yaml)"
    )

    args = parser.parse_args()

    pipeline_config.load(args.config)

    if args.diarize:
        diarize = True
    elif args.no_diarize:
        diarize = False
    else:
        diarize = pipeline_config.get()["pipeline"]["diarize"]

    translate_video(
        args.input,
        args.output,
        args.language,
        args.voice,
        not args.no_subtitles,
        args.source_language,
        diarize,
    )


if __name__ == "__main__":
    main()
