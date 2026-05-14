"""Together AI TTS backend.

Currently supports `cartesia/sonic-3` only. Together's serverless catalog also
includes Kokoro and Orpheus (and other Cartesia versions); adding them needs a
model-specific voice catalog — see ``_NATIVE_VOICES`` below for the shape.
"""

import re
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

_EN_VOICES = ["tutorial man", "helpful woman", "nonfiction man", "reading man"]


def native_voices_for(language_code: str) -> list[str]:
    return _NATIVE_VOICES.get(language_code, _EN_VOICES)


def all_voices() -> dict[str, list[str]]:
    result = dict(_NATIVE_VOICES)
    result["en"] = list(_EN_VOICES)
    return result


def voice_descriptions() -> dict[str, str]:
    """Map voice name → description. Cartesia names are self-describing
    (e.g. 'german conversational woman'), so the name itself is the description."""
    out: dict[str, str] = {}
    for names in _NATIVE_VOICES.values():
        for n in names:
            out[n] = n
    for n in _EN_VOICES:
        out[n] = n
    return out


def _apply_ssml(text: str, speed: float | None, emotion: str | None) -> str:
    """Prepend Cartesia SSML tags for speed and emotion when set."""
    prefix = ""
    if speed is not None:
        prefix += f'<speed ratio="{speed}"/> '
    if emotion:
        prefix += f'<emotion value="{emotion}"/> '
    return prefix + text if prefix else text


def _append_silence(path: str, ms: int) -> None:
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
    response = client.audio.speech.create(
        model=_conf.get()["models"]["tts"]["model"],
        input=_apply_ssml(text, speed, emotion),
        voice=voice,
        response_format="wav",
        language=language,
    )
    response.write_to_file(output_path)
    tcfg = _conf.get().get("tts", {})
    # Longer pause after a sentence-ending mark (period / !? / 。！？) so the
    # next segment doesn't feel hard-cut against the previous one. Mid-clause
    # boundaries (commas etc.) keep the short tail to preserve flow.
    if re.search(r'[.!?。！？]\s*$', text):
        tail_ms = tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    else:
        tail_ms = tcfg.get("tail_silence_ms", 0)
    _append_silence(output_path, tail_ms)
    return output_path


def _is_rate_limit(exc: BaseException) -> bool:
    """Best-effort 429 detection across Together SDK versions."""
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


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

    workers = _conf.get()["tts"]["workers"]
    done_count = 0

    def _record(idx: int, path: str) -> None:
        nonlocal done_count
        paths[idx] = path
        done_count += 1
        if done_count % 10 == 0 or done_count == total:
            print(f"      TTS progress: {done_count}/{total} segments done")

    def _run_serial(items: list[tuple[int, Segment]]) -> None:
        """Plain sequential loop. Together's entry tier is 1 QPS — TTS request
        latency (~1-2 s) already naturally stays under the limit, no sleep needed."""
        for i, seg in items:
            idx, path = _do(i, seg)
            _record(idx, path)

    if workers <= 1:
        _run_serial(list(enumerate(segments)))
        return paths

    # Parallel path: try the configured worker count, then fall back to serial
    # for any segments that hit 429 (typical when a user provides a BYOK key
    # rate-limited to ~1 RPS while the server is configured for 8 parallel).
    rate_limited: list[tuple[int, Segment]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_do, i, seg): (i, seg) for i, seg in enumerate(segments)}
        for future in as_completed(futures):
            i, seg = futures[future]
            try:
                idx, path = future.result()
                _record(idx, path)
            except Exception as e:
                if _is_rate_limit(e):
                    rate_limited.append((i, seg))
                else:
                    raise

    if rate_limited:
        print(f"      Rate-limited — falling back to serial for {len(rate_limited)} segments")
        _run_serial(rate_limited)

    return paths
