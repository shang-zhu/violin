"""Transcribe audio using Together AI's Whisper Large v3."""

import re
from dataclasses import dataclass
from pathlib import Path

from together import Together

from . import config as _conf

# Whisper hallucinates these patterns on music, silence, and noise.
_HALLUCINATION_RE = re.compile(
    r"^\s*[\[\(\*]"          # starts with [, (, or *
    r"|^\s*$"                 # empty
    r"|\bmusic\b"             # background music markers
    r"|\bapplause\b"
    r"|\blaughter\b"
    r"|\bsilence\b"
    r"|\binaudible\b"
    r"|\buntranscribed\b",
    re.IGNORECASE,
)

_SENTENCE_END_RE = re.compile(r'[.!?。！？…]\s*$')

# Minimum speech duration — shorter segments are almost always noise
_MIN_DURATION = 0.8  # seconds

# Minimum characters in a segment (filters single-word/single-char hallucinations)
_MIN_CHARS = 4

# Whisper's no_speech_prob threshold — above this, treat as non-speech
_MAX_NO_SPEECH_PROB = 0.6


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"


def merge_continuous_segments(
    segments: list["Segment"],
    max_gap: float | None = None,
    max_duration: float | None = None,
) -> list["Segment"]:
    """Merge consecutive same-speaker segments that don't end at sentence boundaries.

    This prevents TTS from restarting prosody mid-sentence, producing much more
    natural-sounding dubbed audio.
    """
    cfg = _conf.get()["merge"]
    if max_gap is None:
        max_gap = cfg["max_gap"]
    if max_duration is None:
        max_duration = cfg["max_duration"]

    if not segments:
        return []

    merged: list[Segment] = []
    current = segments[0]

    for seg in segments[1:]:
        gap = seg.start - current.end
        same_speaker = seg.speaker == current.speaker
        ends_sentence = bool(_SENTENCE_END_RE.search(current.text))
        would_be_too_long = (seg.end - current.start) > max_duration

        if same_speaker and gap <= max_gap and not ends_sentence and not would_be_too_long:
            current = Segment(
                id=current.id,
                start=current.start,
                end=seg.end,
                text=current.text + " " + seg.text,
                speaker=current.speaker,
            )
        else:
            merged.append(current)
            current = seg

    merged.append(current)

    for i, seg in enumerate(merged):
        seg.id = i

    return merged


def _is_valid(s: dict | object) -> bool:
    def g(key, default=None):
        if isinstance(s, dict):
            return s.get(key, default)
        return getattr(s, key, default)

    text = (g("text") or "").strip()
    duration = g("end") - g("start")
    no_speech_prob = g("no_speech_prob", 0.0) or 0.0

    if not text:
        return False
    if duration < _MIN_DURATION:
        return False
    if len(text) < _MIN_CHARS:
        return False
    if no_speech_prob > _MAX_NO_SPEECH_PROB:
        return False
    if _HALLUCINATION_RE.search(text):
        return False
    return True


def transcribe(audio_path: str, client: Together) -> list[Segment]:
    """Return clean, timestamped segments from audio file."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(Path(audio_path).name, f),
            model=_conf.get()["models"]["transcription"],
            response_format="verbose_json",
        )

    def g(s, key, default=None):
        if isinstance(s, dict):
            return s.get(key, default)
        return getattr(s, key, default)

    valid = [s for s in response.segments if _is_valid(s)]

    return [
        Segment(id=i, start=g(s, "start"), end=g(s, "end"), text=g(s, "text").strip())
        for i, s in enumerate(valid)
    ]
