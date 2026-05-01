"""ElevenLabs TTS backend.

Uses the multilingual `eleven_v3` model with curated premade voices that
work for any supported target language. Output is fetched as MP3 and
converted in one ffmpeg pass to PCM WAV (with optional tail silence) so the
downstream merger sees the same file format as the Cartesia backend.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from elevenlabs.client import ElevenLabs

from . import config as _conf
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment

# Premade voices ship with every ElevenLabs account and are multilingual —
# the same voice can speak any language the chosen model supports. Metadata
# comes verbatim from the ElevenLabs `/v1/voices` API response (name, id,
# labels, description) so the LLM voice matcher gets accurate cues.
_PREMADE_VOICES: dict[str, dict] = {
    "Adam":     {"id": "pNInz6obpgDQGcFmaJgB", "gender": "male",    "accent": "american",  "description": "Bright tenor that cuts through; brash, openly confident, with unwavering certainty."},
    "Brian":    {"id": "nPczCjzI2devNBz1zQrb", "gender": "male",    "accent": "american",  "description": "Middle-aged man with a deep, resonant, comforting tone. Great for narrations and ads."},
    "Bill":     {"id": "pqHfZKP75CvOlQylNhV4", "gender": "male",    "accent": "american",  "description": "Older, friendly and comforting voice — wise, mature, balanced; ready to narrate stories."},
    "Roger":    {"id": "CwhRBWXzGAHq8TQ4Fs17", "gender": "male",    "accent": "american",  "description": "Easy-going, casual, resonant — perfect for laid-back conversations."},
    "Eric":     {"id": "cjVigY5qzO86Huf0OWal", "gender": "male",    "accent": "american",  "description": "Smooth tenor from a man in his 40s — trustworthy, classy, agentic feel."},
    "Chris":    {"id": "iP95p4xoKVk53GoZ742B", "gender": "male",    "accent": "american",  "description": "Natural, real, charming, down-to-earth — versatile for many use cases."},
    "Will":     {"id": "bIHbv24MWmeRgasZH58o", "gender": "male",    "accent": "american",  "description": "Conversational and laid back; relaxed optimist."},
    "Liam":     {"id": "TX3LPaxmHKxFdv7VOQHJ", "gender": "male",    "accent": "american",  "description": "Young adult with energy and warmth — suitable for reels and shorts."},
    "Harry":    {"id": "SOYHLrjzK2X1ezoPC6cr", "gender": "male",    "accent": "american",  "description": "Animated warrior — rough, fierce, ready to charge forward."},
    "Callum":   {"id": "N2lVS1w4EtoT3dr4eOWO", "gender": "male",    "accent": "american",  "description": "Deceptively gravelly with an unsettling edge — husky trickster."},
    "Charlie":  {"id": "IKne3meq5aSn9XLyUdCD", "gender": "male",    "accent": "australian","description": "Young Australian male — confident, energetic, hyped."},
    "George":   {"id": "JBFqnCBsd6RMkjVDRZzb", "gender": "male",    "accent": "british",   "description": "Warm, mature British male — captivating storyteller resonance."},
    "Daniel":   {"id": "onwK4e9ZLuTAKqWW03F9", "gender": "male",    "accent": "british",   "description": "Strong, formal British male — perfect for professional broadcast or news."},
    "Sarah":    {"id": "EXAVITQu4vr4xnSDxMaL", "gender": "female",  "accent": "american",  "description": "Young adult woman — confident, warm, mature; reassuring, professional tone."},
    "Bella":    {"id": "hpp4J3VqNfWAUOO0d1Us", "gender": "female",  "accent": "american",  "description": "Warm, bright, professional middle-aged American female — polished narrative quality."},
    "Matilda":  {"id": "XrExE9yKIg1WjnnlVkGX", "gender": "female",  "accent": "american",  "description": "Professional middle-aged American woman; pleasing alto, upbeat, knowledgeable."},
    "Jessica":  {"id": "cgSgspJ2msm6clMCkdW9", "gender": "female",  "accent": "american",  "description": "Young, playful, bright, warm American female — perfect for trendy content."},
    "Laura":    {"id": "FGY2WhTYpPnrIDTdsKH5", "gender": "female",  "accent": "american",  "description": "Young adult American female — sunny enthusiasm with a quirky, sassy attitude."},
    "Alice":    {"id": "Xb7hH8MSUJpSbSDYk0k2", "gender": "female",  "accent": "british",   "description": "Clear, engaging middle-aged British female — friendly, professional, suitable for e-learning."},
    "Lily":     {"id": "pFZP5JQG7iQjIQuC4Bku", "gender": "female",  "accent": "british",   "description": "Velvety middle-aged British female — confident; news and narrations with warmth and clarity."},
    "River":    {"id": "SAz9YHcvj6GT2YYXdXww", "gender": "neutral", "accent": "american",  "description": "Relaxed, neutral, calm — informative voice ready for narrations or conversational projects."},
}

# Default ordering [primary_male, primary_female] used when caller doesn't
# specify a voice. Multilingual — same defaults regardless of target language.
_DEFAULT_PAIR = ["Adam", "Sarah"]


def native_voices_for(language_code: str) -> list[str]:
    """Multilingual premade voices — language code is ignored."""
    return list(_DEFAULT_PAIR)


def all_voices() -> dict[str, list[str]]:
    """Return all premade voices under a single 'multi' bucket (multilingual)."""
    return {"multi": list(_PREMADE_VOICES.keys())}


def voice_descriptions() -> dict[str, str]:
    """Map voice name → human-readable description (for LLM voice matching)."""
    return {
        name: f"{meta['gender']}, {meta['accent']} accent — {meta['description']}"
        for name, meta in _PREMADE_VOICES.items()
    }


def _resolve_voice_id(voice: str) -> str:
    """Accept either a premade voice name or a raw voice ID."""
    entry = _PREMADE_VOICES.get(voice)
    if entry:
        return entry["id"]
    return voice


def _to_wav(mp3_path: str, wav_path: str, tail_ms: int) -> None:
    """Convert MP3 to mono 44100 PCM WAV, optionally appending silence."""
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
    client: ElevenLabs,
    language: str = "en",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    """Synthesize one segment to a WAV file via ElevenLabs."""
    cfg = _conf.get()
    tts_entry = cfg["models"]["tts"]
    model_id = tts_entry["model"] if isinstance(tts_entry, dict) else "eleven_v3"

    voice_settings = {}
    # ElevenLabs `speed` accepts 0.7–1.2 on supported models; ignore extreme values.
    if speed is not None and 0.7 <= speed <= 1.2:
        voice_settings["speed"] = speed

    kwargs = dict(
        voice_id=_resolve_voice_id(voice),
        text=text,
        model_id=model_id,
        output_format="mp3_44100_128",
    )
    if voice_settings:
        kwargs["voice_settings"] = voice_settings

    audio = client.text_to_speech.convert(**kwargs)

    mp3_path = output_path + ".tmp.mp3"
    with open(mp3_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)

    tail_ms = cfg.get("tts", {}).get("tail_silence_ms", 0)
    _to_wav(mp3_path, output_path, tail_ms)
    Path(mp3_path).unlink(missing_ok=True)
    return output_path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: ElevenLabs,
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
