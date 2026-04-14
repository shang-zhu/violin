"""Extract audio from video using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path

import ffmpeg

from .ffmpeg_utils import FFMPEG_EXE, get_duration_video

_OVERLAP_SECONDS = 1  # small overlap to avoid cutting mid-word


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio from video as 16kHz mono WAV — optimal for Whisper."""
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(tempfile.mkdtemp()) / f"{stem}_audio.wav")

    (
        ffmpeg.input(video_path)
        .output(output_path, ar=16000, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True, cmd=FFMPEG_EXE)
    )
    return output_path


def split_audio(audio_path: str, output_dir: str | None = None,
                chunk_seconds: float = 600) -> list[tuple[str, float]]:
    """Split a WAV file into chunks of *chunk_seconds*.

    Returns a list of (chunk_path, offset_seconds) tuples.  If the file is
    shorter than one chunk, returns a single-element list with offset 0.
    """
    duration = get_duration_video(audio_path)
    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audiochunk_")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[str, float]] = []
    offset = 0.0
    idx = 0
    while offset < duration:
        chunk_path = str(out / f"chunk_{idx:04d}.wav")
        length = min(chunk_seconds + _OVERLAP_SECONDS, duration - offset)
        subprocess.run([
            FFMPEG_EXE,
            "-ss", str(offset),
            "-t", str(length),
            "-i", audio_path,
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", chunk_path,
        ], check=True, capture_output=True)
        chunks.append((chunk_path, offset))
        offset += chunk_seconds
        idx += 1

    return chunks


def get_video_duration(video_path: str) -> float:
    return get_duration_video(video_path)
