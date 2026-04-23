"""FFmpeg utilities."""

import os
import re
import shutil
import subprocess

import soundfile as sf

def _find_ffmpeg() -> str:
    """Return the path to an ffmpeg binary, preferring system ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return os.environ.get("FFMPEG_PATH", "ffmpeg")

FFMPEG_EXE = _find_ffmpeg()


def get_duration_video(path: str) -> float:
    """Get duration of any audio/video file.

    Tries soundfile first (fast, handles all WAV variants including float PCM).
    Falls back to parsing ffmpeg stderr for video files or unsupported formats.
    """
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        pass

    # Fallback: parse ffmpeg stderr
    result = subprocess.run(
        [FFMPEG_EXE, "-i", path],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
    if not match:
        raise ValueError(f"Could not determine duration of {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)
