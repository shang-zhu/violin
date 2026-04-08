"""Transcribe audio using Together AI's Whisper Large v3."""

import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from together import Together

from . import config as _conf

_MAX_RETRIES = 3
_RETRY_BACKOFF = [5, 15, 30]  # seconds — generous for long diarization requests
_TIMEOUT = 600  # 10 minutes — diarization on long audio can take a while

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


def _g(s: dict | object, key: str, default=None):
    """Attribute-or-dict accessor for API response objects."""
    if isinstance(s, dict):
        return s.get(key, default)
    return getattr(s, key, default)


def transcribe(
    audio_path: str,
    client: Together,
    diarize: bool = False,
) -> list[Segment]:
    """Return clean, timestamped segments from audio file.

    When *diarize* is True, passes ``diarize="true"`` to Together's Whisper API
    and labels each segment with the dominant speaker derived from the returned
    ``speaker_segments``.
    """
    extra_kwargs: dict = {}
    if diarize:
        extra_kwargs["diarize"] = "true"

    model = _conf.get()["models"]["transcription"]
    response = None
    for attempt in range(_MAX_RETRIES):
        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=(Path(audio_path).name, f),
                    model=model,
                    response_format="verbose_json",
                    timeout=_TIMEOUT,
                    **extra_kwargs,
                )
            break
        except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            if attempt < _MAX_RETRIES - 1:
                print(f"      Transcription timed out (attempt {attempt + 1}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Transcription timed out after {_MAX_RETRIES} attempts"
                ) from exc
        except Exception as exc:
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            if attempt < _MAX_RETRIES - 1:
                print(f"      Transcription error (attempt {attempt + 1}): {exc}, "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    assert response is not None

    valid = [s for s in response.segments if _is_valid(s)]
    segments = [
        Segment(id=i, start=_g(s, "start"), end=_g(s, "end"), text=_g(s, "text").strip())
        for i, s in enumerate(valid)
    ]

    if diarize:
        speaker_segs = getattr(response, "speaker_segments", None) or []
        turns = [
            (_g(ss, "start"), _g(ss, "end"), _g(ss, "speaker_id", "SPEAKER_00"))
            for ss in speaker_segs
        ]
        if turns:
            for seg in segments:
                overlap: dict[str, float] = {}
                for t_start, t_end, spk in turns:
                    o = min(seg.end, t_end) - max(seg.start, t_start)
                    if o > 0:
                        overlap[spk] = overlap.get(spk, 0) + o
                if overlap:
                    seg.speaker = max(overlap, key=overlap.get)

    return segments


def find_main_speaker(segments: list[Segment]) -> str:
    """Return the speaker_id with the longest total speaking duration."""
    durations: dict[str, float] = {}
    for seg in segments:
        durations[seg.speaker] = durations.get(seg.speaker, 0) + (seg.end - seg.start)
    if not durations:
        return "SPEAKER_00"
    return max(durations, key=durations.get)
