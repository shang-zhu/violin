"""OpenAI TTS backend (tts-1, tts-1-hd, gpt-4o-mini-tts).

OpenAI's TTS voices are model-side multilingual — the same six voices speak
every supported language reasonably well, so the language code is not used to
pick a voice (we keep one catalog under "en" and let other languages fall
through to it, matching the public API).

Output is MP3, converted in one ffmpeg pass to PCM WAV with optional tail
silence so the downstream merger sees the same format as the other backends.
"""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from . import config as _conf
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment


# OpenAI TTS voices (the original 6 supported by every model including
# tts-1 / tts-1-hd). Newer models add coral/sage/ballad/verse/ash but those
# only work with gpt-4o-*-tts; we stick to the universally-supported set.
_NATIVE_VOICES: dict[str, dict[str, dict]] = {
    "en": {
        "onyx":    {"gender": "male",   "description": "Deep, authoritative male — narration and serious content."},
        "echo":    {"gender": "male",   "description": "Calm, even male — versatile for tutorials and explainers."},
        "fable":   {"gender": "male",   "description": "Warm British male — storytelling and audiobook feel."},
        "alloy":   {"gender": "neutral","description": "Neutral, balanced voice — general-purpose."},
        "nova":    {"gender": "female", "description": "Bright, energetic female — engaging for social/short-form."},
        "shimmer": {"gender": "female", "description": "Gentle, soft female — calm narration and friendly content."},
    },
}


def native_voices_for(language_code: str) -> list[str]:
    """Return [primary_male, primary_female]. OpenAI voices are multilingual,
    so we always use the same 'en' catalog regardless of language."""
    voices = _NATIVE_VOICES["en"]
    names = list(voices.keys())
    male = next((n for n in names if voices[n]["gender"] == "male"), names[0])
    female = next((n for n in names if voices[n]["gender"] == "female"), names[-1])
    return [male, female]


def all_voices() -> dict[str, list[str]]:
    return {lang: list(voices.keys()) for lang, voices in _NATIVE_VOICES.items()}


def voice_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    for voices in _NATIVE_VOICES.values():
        for name, meta in voices.items():
            out[name] = f"{meta['gender']} — {meta['description']}"
    return out


def _to_wav(mp3_path: str, wav_path: str, tail_ms: int) -> None:
    af = []
    if tail_ms > 0:
        af = ["-af", f"apad=pad_dur={tail_ms / 1000:.3f}"]
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", mp3_path,
         *af,
         "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
         wav_path],
        check=True, capture_output=True,
    )


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client: OpenAI,
    language: str = "en",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    """Synthesize one segment to a WAV file via OpenAI TTS."""
    cfg = _conf.get()
    tts_entry = cfg["models"]["tts"]
    model_id = tts_entry["model"] if isinstance(tts_entry, dict) else "tts-1-hd"

    kwargs = dict(model=model_id, voice=voice, input=text, response_format="mp3")
    # OpenAI accepts speed in [0.25, 4.0].
    if speed is not None and 0.25 <= speed <= 4.0:
        kwargs["speed"] = speed

    mp3_path = output_path + ".tmp.mp3"
    with client.audio.speech.with_streaming_response.create(**kwargs) as resp:
        resp.stream_to_file(mp3_path)

    tcfg = cfg.get("tts", {})
    if re.search(r'[.!?。！？]\s*$', text):
        tail_ms = tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    else:
        tail_ms = tcfg.get("tail_silence_ms", 0)
    _to_wav(mp3_path, output_path, tail_ms)
    Path(mp3_path).unlink(missing_ok=True)
    return output_path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: OpenAI,
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> list[str]:
    total = len(segments)
    paths = [""] * total
    vm = voice_map or {}

    def _do(idx: int, seg: Segment) -> tuple[int, str]:
        path = str(Path(output_dir) / f"seg_{seg.id:05d}.wav")
        seg_voice = vm.get(seg.speaker, voice)
        synthesize_segment(seg.text, seg_voice, path, client, language, speed, emotion)
        if tracker:
            tracker.add_tts_usage(len(seg.text))
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
