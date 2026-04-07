"""
Creates a short test video with English speech for pipeline testing.
Steps:
  1. Generate English speech via Minimax TTS (Together AI)
  2. Combine with a color-bar video pattern using ffmpeg
Output: test_input.mp4
"""

import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

from pipeline.ffmpeg_utils import FFMPEG_EXE

load_dotenv()

TEXT = (
    "Welcome to this educational video about the water cycle. "
    "Water evaporates from oceans and lakes, rises into the atmosphere, "
    "forms clouds, and eventually falls back to Earth as rain or snow. "
    "This continuous process is essential for all life on our planet."
)

def create_test_video(output_path: str = "test_input.mp4") -> None:
    api_key = os.environ["TOGETHER_API_KEY"]
    base_url = os.environ.get("TOGETHER_TTS_BASE_URL", "https://api.together.xyz/v1").rstrip("/")

    tmp_dir = Path(tempfile.mkdtemp())
    audio_path = str(tmp_dir / "speech.wav")

    print("Generating English speech via Minimax TTS...")
    response = httpx.post(
        f"{base_url}/audio/speech",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "minimax/speech-2.6-turbo",
            "input": TEXT,
            "voice": "English_expressive_narrator",
            "response_format": "wav",
        },
        timeout=60,
    )
    response.raise_for_status()
    Path(audio_path).write_bytes(response.content)
    print(f"  Audio saved ({len(response.content) // 1024} KB)")

    print("Creating test video with color-bar pattern...")
    subprocess.run(
        [
            FFMPEG_EXE,
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25",
            "-i", audio_path,
            "-shortest",
            "-c:v", "libx264", "-c:a", "aac",
            "-y", output_path,
        ],
        check=True,
        capture_output=True,
    )
    print(f"Test video created: {output_path}")


if __name__ == "__main__":
    create_test_video()
