"""Transcribe audio using Together AI's Whisper Large v3."""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from together import Together

from . import config as _conf
from .extractor import split_audio

_MAX_RETRIES = 3
_RETRY_BACKOFF = [5, 15, 30]
_TIMEOUT = 600
_DEFAULT_TRANSCRIBE_WORKERS = 2

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


def _transcribe_single(
    audio_path: str,
    client: Together,
    model: str,
) -> list[Segment]:
    """Transcribe a single audio file (must be small enough for the API)."""
    response = None
    for attempt in range(_MAX_RETRIES):
        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=(Path(audio_path).name, f),
                    model=model,
                    response_format="verbose_json",
                    timeout=_TIMEOUT,
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

    return segments


def _dedup_overlap(segments: list[Segment]) -> list[Segment]:
    """Remove near-duplicate segments from chunk boundaries.

    When chunks overlap, the same speech can appear at the end of one chunk
    and the start of the next.  Drop a segment if it overlaps heavily with
    the previous one and has similar text.
    """
    if len(segments) < 2:
        return segments
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        time_overlap = max(0, prev.end - seg.start)
        seg_dur = seg.end - seg.start
        if seg_dur > 0 and time_overlap / seg_dur > 0.5:
            continue
        out.append(seg)
    return out


def transcribe(
    audio_path: str,
    client: Together,
) -> list[Segment]:
    """Return clean, timestamped segments from audio file.

    Long audio files are automatically split into ~10-minute chunks,
    transcribed in parallel, and stitched back together.
    """
    cfg = _conf.get()
    model = cfg["models"]["transcription"]
    tcfg = cfg.get("transcription", {})
    chunk_seconds = tcfg.get("chunk_seconds", 600)
    workers = tcfg.get("parallel_workers", _DEFAULT_TRANSCRIBE_WORKERS)

    chunks = split_audio(audio_path, chunk_seconds=chunk_seconds)

    if len(chunks) == 1:
        print(f"      Transcribing single file…")
        return _transcribe_single(audio_path, client, model)

    print(f"      Audio split into {len(chunks)} chunks, transcribing in parallel…")
    results: dict[int, list[Segment]] = {}

    def _do(idx: int, chunk_path: str, offset: float) -> tuple[int, list[Segment]]:
        segs = _transcribe_single(chunk_path, client, model)
        for s in segs:
            s.start += offset
            s.end += offset
        print(f"      Chunk {idx + 1}/{len(chunks)} transcribed ({len(segs)} segments)")
        return idx, segs

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do, i, path, offset)
                   for i, (path, offset) in enumerate(chunks)]
        for f in as_completed(futures):
            idx, segs = f.result()
            results[idx] = segs

    all_segments: list[Segment] = []
    for i in range(len(chunks)):
        all_segments.extend(results[i])

    all_segments.sort(key=lambda s: s.start)
    all_segments = _dedup_overlap(all_segments)

    for i, seg in enumerate(all_segments):
        seg.id = i

    return all_segments


