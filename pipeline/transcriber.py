"""Transcribe audio with Whisper — provider chosen via config (Together or OpenAI)."""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from . import config as _conf
from .extractor import split_audio
from .llm_client import get_transcription_model

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

_SENTENCE_END_RE = re.compile(r'[.!?。！？]\s*$')

# A segment starting with a lowercase letter or a single uppercase letter followed
# by a space/CJK character is almost certainly a mid-word fragment from a Whisper
# internal split (e.g. "B model" from "120B model").
_FRAGMENT_START_RE = re.compile(r'^[a-z]|^[A-Z][\s一-鿿]')

_SENTENCE_SPLIT_RE = re.compile(r'[^.!?。！？]*[.!?。！？]+(?!\d)\s*')
_PROTECTED_PERIOD = re.compile(r'(?<=\w)\.(?=\w)')  # decimal / abbreviation: 2.0, e.g, U.S

# Clause-level punctuation for soft-splitting long sub-sentences (Chinese commas
# 、，；, English , ;).  We only break here when a sentence-level part exceeds
# max_subtitle_chars, to keep individual subtitle lines short and readable.
_SOFT_SPLIT_RE = re.compile(r'[,，;；、]')


def _soft_split_long(text: str, max_chars: int) -> list[str]:
    """Recursively split *text* at clause punctuation when it exceeds max_chars.

    Picks the punctuation closest to the midpoint each time so the resulting
    pieces stay roughly balanced. If no soft punctuation is available, returns
    the text unchanged.
    """
    if len(text) <= max_chars:
        return [text]
    matches = list(_SOFT_SPLIT_RE.finditer(text))
    if not matches:
        return [text]
    mid = len(text) / 2
    best = min(matches, key=lambda m: abs(m.end() - mid))
    left = text[:best.end()].rstrip()
    right = text[best.end():].lstrip()
    if not left or not right:
        return [text]
    return _soft_split_long(left, max_chars) + _soft_split_long(right, max_chars)

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


def _deduplicate_fragment(prev_text: str, frag_text: str) -> str:
    """Remove the overlapping prefix from frag_text that already appears at the end of prev_text.

    Whisper repeats the last few words of an internal chunk at the start of the
    next chunk. Detect the overlap via case-insensitive character matching and
    strip the duplicate prefix before merging.
    """
    prev = prev_text.rstrip('.!?。！？ ')
    frag = frag_text.lstrip()
    prev_lower = prev.lower()
    frag_lower = frag.lower()

    for length in range(min(len(prev), len(frag)), 1, -1):
        if prev_lower.endswith(frag_lower[:length]):
            remainder = frag[length:].lstrip(' ,，;；')
            return remainder if remainder.strip() else frag_text

    return frag_text


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
        next_is_fragment = bool(_FRAGMENT_START_RE.match(seg.text.strip()))
        would_be_too_long = (seg.end - current.start) > max_duration

        if same_speaker and gap <= max_gap and (not ends_sentence or next_is_fragment) and not would_be_too_long:
            seg_text = _deduplicate_fragment(current.text, seg.text) if next_is_fragment else seg.text
            merged_text = (current.text + " " + seg_text).strip() if seg_text.strip() else current.text
            current = Segment(
                id=current.id,
                start=current.start,
                end=seg.end,
                text=merged_text,
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


_ABBREVIATION_WHITELIST = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "u.s.", "u.s.a.", "u.k.", "e.u.",
    "etc.", "vs.", "i.e.", "e.g.", "approx.", "incl.", "cf.", "ca.",
    "inc.", "ltd.", "corp.", "co.", "no.", "vol.",
    "fig.", "eq.", "ref.", "ed.", "p.m.", "a.m.",
})


def _is_sentence_end(word: str) -> bool:
    """True when *word* ends with terminal punctuation that isn't an abbreviation.

    Avoids treating "U.S.", "etc.", "Mr.", "vs." etc. as sentence boundaries —
    otherwise the word-level segmenter would split mid-thought, leaving the
    next sub-segment with an unrealistically short orig_dur and forcing the
    aligner into freeze/atempo territory.
    """
    stripped = word.rstrip()
    if not re.search(r'[.!?。！？]\s*$', stripped):
        return False
    # CJK terminal punctuation is unambiguous.
    if stripped.endswith(("。", "！", "？")):
        return True
    if stripped.endswith(("!", "?")):
        return True
    # English period: only a sentence end if the trailing token isn't a
    # known abbreviation. Compare lowercased to handle "U.S." vs "u.s.".
    last_token = stripped.split()[-1].lower() if stripped.split() else ""
    return last_token not in _ABBREVIATION_WHITELIST


def _split_words_into_sentences(words: list, offset: float = 0.0) -> list[Segment]:
    """Build sentence-level Segment objects from word-level timestamps.

    Words are grouped into sentences by detecting sentence-ending punctuation.
    Each sentence gets the exact start/end timestamp from the word data,
    eliminating the need for character-proportional estimation.
    """
    if not words:
        return []

    sentences: list[Segment] = []
    current_words: list = []

    for w in words:
        word = _g(w, "word") or ""
        if not word.strip():
            continue
        current_words.append(w)
        # Sentence ends when the word ends with terminal punctuation,
        # excluding common abbreviations (Mr., U.S., etc.).
        if _is_sentence_end(word):
            start = _g(current_words[0], "start") + offset
            end = _g(current_words[-1], "end") + offset
            text = " ".join((_g(w2, "word") or "").strip() for w2 in current_words).strip()
            if text:
                sentences.append(Segment(id=len(sentences), start=start, end=end, text=text))
            current_words = []

    # Flush any trailing words that didn't end with punctuation.
    if current_words:
        start = _g(current_words[0], "start") + offset
        end = _g(current_words[-1], "end") + offset
        text = " ".join((_g(w2, "word") or "").strip() for w2 in current_words).strip()
        if text:
            sentences.append(Segment(id=len(sentences), start=start, end=end, text=text))

    return sentences


def _transcribe_single(
    audio_path: str,
    client: Any,
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
                    timestamp_granularities=["word", "segment"],
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

    words = getattr(response, "words", None) or []
    valid_segments = [s for s in response.segments if _is_valid(s)]

    if words:
        # Use the segment-level confidence signal (no_speech_prob etc. via
        # _is_valid) to drop words that fall inside hallucinated time ranges
        # — Whisper invents tokens like "你 你 你 你..." or "thank you. thank
        # you..." over silence/music at the ends of clips.
        valid_ranges = [(_g(s, "start"), _g(s, "end")) for s in valid_segments]
        if valid_ranges:
            def _in_valid_range(w: object) -> bool:
                ws = _g(w, "start") or 0.0
                return any(rs <= ws <= re for rs, re in valid_ranges)
            words = [w for w in words if _in_valid_range(w)]
        return _split_words_into_sentences(words)

    # Fallback: no word timestamps available, use segment-level.
    return [
        Segment(id=i, start=_g(s, "start"), end=_g(s, "end"), text=_g(s, "text").strip())
        for i, s in enumerate(valid_segments)
    ]


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
    client: Any,
) -> list[Segment]:
    """Return clean, timestamped segments from audio file.

    Long audio files are automatically split into ~10-minute chunks,
    transcribed in parallel, and stitched back together.
    """
    cfg = _conf.get()
    model = get_transcription_model(cfg)
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


def split_into_sentences(segments: list[Segment]) -> list[Segment]:
    """Split each segment into sentence-level sub-segments for precise subtitle timing.

    Time is distributed across sentences proportionally by character count — a
    reasonable proxy for speech duration. After TTS, each sentence chunk gets its
    own speed-adjusted video slice, giving subtitles that match actual speech.

    When ``merge.max_subtitle_chars`` is set in config, sub-sentences longer
    than that threshold are further split at clause punctuation (、，；,;) so
    individual subtitle lines stay readable.
    """
    max_chars = _conf.get()["merge"].get("max_subtitle_chars", 0) or 0

    result: list[Segment] = []

    _PLACEHOLDER = "\x00"

    for seg in segments:
        # Protect periods that are NOT sentence boundaries (decimals, abbreviations).
        protected = _PROTECTED_PERIOD.sub(_PLACEHOLDER, seg.text)
        parts = _SENTENCE_SPLIT_RE.findall(protected)
        covered = "".join(parts)
        remainder = protected[len(covered):].strip()
        if remainder:
            parts.append(remainder)
        # Restore protected periods and strip whitespace.
        parts = [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]

        # Soft-split any part that's still too long for a single subtitle line.
        if max_chars > 0:
            soft_parts: list[str] = []
            for p in parts:
                soft_parts.extend(_soft_split_long(p, max_chars))
            parts = soft_parts

        if len(parts) <= 1:
            result.append(seg)
            continue

        total_chars = sum(len(p) for p in parts)
        dur = seg.end - seg.start
        t = seg.start
        for part in parts:
            part_dur = dur * (len(part) / total_chars)
            result.append(Segment(
                id=0,
                start=t,
                end=t + part_dur,
                text=part,
                speaker=seg.speaker,
            ))
            t += part_dur

    for i, seg in enumerate(result):
        seg.id = i

    return result

