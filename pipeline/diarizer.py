"""Speaker diarization using pyannote.audio."""

import torch
import soundfile as sf
from pyannote.audio import Pipeline

from .transcriber import Segment
from .tts import native_voices_for


def load_pipeline(hf_token: str) -> Pipeline:
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )


def _load_audio(audio_path: str) -> dict:
    """Load audio as a torch tensor dict — bypasses torchcodec/ffmpeg issues in pyannote."""
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    # soundfile returns (frames, channels) — pyannote wants (channels, frames)
    waveform = torch.from_numpy(waveform.T)
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize(audio_path: str, hf_token: str) -> dict[str, str]:
    """
    Run speaker diarization and return a mapping of speaker label → voice name.
    e.g. {"SPEAKER_00": "tutorial man", "SPEAKER_01": "helpful woman"}
    """
    pipeline = load_pipeline(hf_token)
    diarization = pipeline(_load_audio(audio_path)).speaker_diarization

    # Collect speakers ordered by their first appearance
    seen: list[str] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in seen:
            seen.append(speaker)

    return {
        speaker: SPEAKER_VOICES[i % len(SPEAKER_VOICES)]
        for i, speaker in enumerate(seen)
    }


def assign_speakers(
    segments: list[Segment],
    audio_path: str,
    hf_token: str,
    language_code: str = "en",
) -> tuple[list[Segment], dict[str, str]]:
    """
    Label each segment with the dominant speaker in its time window.
    Returns updated segments and the speaker→voice mapping.
    """
    pipeline = load_pipeline(hf_token)
    diarization = pipeline(_load_audio(audio_path)).speaker_diarization

    # Build list of (start, end, speaker) turns
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    # Collect speakers ordered by first appearance for voice assignment
    seen: list[str] = []
    for _, _, speaker in turns:
        if speaker not in seen:
            seen.append(speaker)
    voices = native_voices_for(language_code)
    voice_map = {
        speaker: voices[i % len(voices)]
        for i, speaker in enumerate(seen)
    }

    # For each segment, find the speaker with the most overlap
    labeled: list[Segment] = []
    for seg in segments:
        overlap: dict[str, float] = {}
        for t_start, t_end, speaker in turns:
            o = min(seg.end, t_end) - max(seg.start, t_start)
            if o > 0:
                overlap[speaker] = overlap.get(speaker, 0) + o
        dominant = max(overlap, key=overlap.get) if overlap else (seen[0] if seen else "SPEAKER_00")
        labeled.append(Segment(id=seg.id, start=seg.start, end=seg.end, text=seg.text, speaker=dominant))

    return labeled, voice_map
