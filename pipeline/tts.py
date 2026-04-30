"""Synthesize speech using Cartesia Sonic 3 via Together AI (serverless)."""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from together import Together

from . import config as _conf
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment

# Native-sounding voices per language — matched to Cartesia's language-specific voice catalog.
# Ordered: [primary male, primary female].
_NATIVE_VOICES: dict[str, list[str]] = {
    "zh": ["chinese commercial man", "chinese female conversational"],
    "ja": ["japanese male conversational", "japanese woman conversational"],
    "ko": ["korean narrator man", "korean calm woman"],
    "es": ["spanish narrator man", "spanish narrator lady"],
    "fr": ["french narrator man", "french narrator lady"],
    "de": ["german reporter man", "german conversational woman"],
    "it": ["italian narrator man", "italian narrator woman"],
    "nl": ["dutch confident man", "dutch man"],
    "ru": ["russian narrator man 1", "russian narrator woman"],
    "pt": ["friendly brazilian man", "pleasant brazilian lady"],
    "hi": ["hindi narrator man", "hindi narrator woman"],
    "tr": ["turkish narrator man", "turkish calm man"],
    "pl": ["polish confident man", "polish narrator woman"],
    "sv": ["swedish narrator man", "swedish calm lady"],
    "ar": ["middle eastern woman", "middle eastern woman"],  # one option available
}

# English fallback voices for unmatched languages
_EN_VOICES = ["tutorial man", "helpful woman", "nonfiction man", "reading man"]


def native_voices_for(language_code: str) -> list[str]:
    """Return [male_voice, female_voice] for the given BCP-47 language code."""
    return _NATIVE_VOICES.get(language_code, _EN_VOICES)


def all_voices() -> dict[str, list[str]]:
    """Return all known native voices grouped by BCP-47 language code."""
    result = dict(_NATIVE_VOICES)
    result["en"] = list(_EN_VOICES)
    return result


def _apply_ssml(text: str, speed: float | None, emotion: str | None) -> str:
    """Prepend Cartesia SSML tags for speed and emotion when set."""
    prefix = ""
    if speed is not None:
        prefix += f'<speed ratio="{speed}"/> '
    if emotion:
        prefix += f'<emotion value="{emotion}"/> '
    return prefix + text if prefix else text


def _append_silence(path: str, ms: int) -> None:
    """Append ms milliseconds of silence to a WAV file in-place via ffmpeg."""
    if ms <= 0:
        return
    tmp = path + ".pad.wav"
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", path,
         "-af", f"apad=pad_dur={ms / 1000:.3f}",
         tmp],
        check=True, capture_output=True,
    )
    Path(tmp).replace(Path(path))


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client: Together,
    language: str = "en",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    """Synthesize a single text segment to a WAV file."""
    response = client.audio.speech.create(
        model=_conf.get()["models"]["tts"],
        input=_apply_ssml(text, speed, emotion),
        voice=voice,
        response_format="wav",
        language=language,
    )
    response.write_to_file(output_path)
    tail_ms = _conf.get().get("tts", {}).get("tail_silence_ms", 0)
    _append_silence(output_path, tail_ms)
    return output_path


def tts_one_segment(
    seg: Segment,
    voice: str,
    output_dir: str,
    client: Together,
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> tuple[int, str]:
    """Synthesize a single Segment and return ``(seg.id, output_path)``."""
    vm = voice_map or {}
    path = str(Path(output_dir) / f"seg_{seg.id:05d}.wav")
    seg_voice = vm.get(seg.speaker, voice)
    synthesize_segment(seg.text, seg_voice, path, client, language, speed, emotion)
    if tracker:
        tracker.add_tts_usage(len(seg.text))
    return seg.id, path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: Together,
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> list[str]:
    """Synthesize all segments concurrently.

    If voice_map is provided (speaker -> voice), each segment uses the voice
    assigned to its speaker. Falls back to *voice* for unlabeled segments.
    """
    total = len(segments)
    paths = [""] * total

    def _do(idx: int, seg: Segment) -> tuple[int, str]:
        _, path = tts_one_segment(
            seg, voice, output_dir, client, language, voice_map, tracker,
            speed, emotion,
        )
        return idx, path

    done_count = 0
    with ThreadPoolExecutor(max_workers=_conf.get()["tts"]["workers"]) as pool:
        futures = {pool.submit(_do, i, seg): i for i, seg in enumerate(segments)}
        for future in as_completed(futures):
            idx, path = future.result()
            paths[idx] = path
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"      TTS progress: {done_count}/{total} segments done")

    return paths
