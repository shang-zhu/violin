"""Extract audio from video using ffmpeg."""

import tempfile
from pathlib import Path

import ffmpeg

from .ffmpeg_utils import FFMPEG_EXE, get_duration_video


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


def get_video_duration(video_path: str) -> float:
    return get_duration_video(video_path)
