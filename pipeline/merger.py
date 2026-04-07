"""Merge translated audio back into video and generate subtitle file."""

import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as _conf
from .ffmpeg_utils import FFMPEG_EXE, get_duration_video
from .transcriber import Segment

_SAMPLE_RATE = 44100
_SEEK_PAD = 2.0  # seconds before target to land keyframe seek, then trim precisely


def _probe_fps(video_path: str) -> float:
    """Get the frame rate of the source video."""
    try:
        result = subprocess.run(
            [FFMPEG_EXE, "-i", video_path],
            capture_output=True, text=True,
        )
        match = re.search(r"(\d+(?:\.\d+)?)\s*fps", result.stderr)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return _conf.get()["merge_video"]["output_fps"]


def _make_speech_chunk(
    video_path: str, start: float, end: float,
    tts_path: str, out_path: str, fps: float | None = None,
) -> None:
    """Extract video segment, speed-adjust to match TTS duration, mux with TTS audio."""
    vcfg = _conf.get()["merge_video"]
    if fps is None:
        fps = vcfg["output_fps"]
    orig_dur = end - start
    tts_dur = get_duration_video(tts_path)
    if orig_dur < 0.01 or tts_dur < 0.01:
        speed = 1.0
    else:
        speed = max(vcfg["speed_clamp_min"], min(vcfg["speed_clamp_max"], orig_dur / tts_dur))
    target_dur = orig_dur / speed

    coarse = max(0, start - _SEEK_PAD)
    fine = start - coarse

    subprocess.run([
        FFMPEG_EXE,
        "-ss", str(coarse), "-t", str(orig_dur + fine + 0.5), "-i", video_path,
        "-i", tts_path,
        "-filter_complex",
        f"[0:v]trim=start={fine}:duration={orig_dur},setpts=(PTS-STARTPTS)/{speed},fps=fps={fps}[v]",
        "-map", "[v]", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", vcfg["preset"], "-crf", str(vcfg["crf"]),
        "-c:a", "aac", "-ar", str(_SAMPLE_RATE), "-ac", "1",
        "-t", str(target_dur),
        "-f", "mpegts",
        "-y", out_path,
    ], check=True, capture_output=True)


def _make_gap_chunk(
    video_path: str, start: float, end: float,
    out_path: str, fps: float | None = None,
) -> None:
    """Extract gap video at original speed with silence."""
    vcfg = _conf.get()["merge_video"]
    if fps is None:
        fps = vcfg["output_fps"]
    dur = end - start
    coarse = max(0, start - _SEEK_PAD)
    fine = start - coarse

    subprocess.run([
        FFMPEG_EXE,
        "-ss", str(coarse), "-t", str(dur + fine + 0.5), "-i", video_path,
        "-f", "lavfi", "-i", f"anullsrc=r={_SAMPLE_RATE}:cl=mono",
        "-filter_complex",
        f"[0:v]trim=start={fine}:duration={dur},setpts=PTS-STARTPTS,fps=fps={fps}[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", vcfg["preset"], "-crf", str(vcfg["crf"]),
        "-c:a", "aac", "-ar", str(_SAMPLE_RATE), "-ac", "1",
        "-t", str(dur),
        "-f", "mpegts",
        "-y", out_path,
    ], check=True, capture_output=True)


def build_aligned_video(
    video_path: str,
    segments: list[Segment],
    tts_paths: list[str],
    total_duration: float,
    output_path: str,
) -> list[Segment]:
    """Build video with per-segment speed adjustment so video matches TTS audio timing.

    Returns segments with updated timestamps matching the new timeline.
    """
    vcfg = _conf.get()["merge_video"]
    min_gap = vcfg["min_gap"]
    chunk_workers = vcfg["workers"]

    tmp_dir = Path(tempfile.mkdtemp(prefix="vidmerge_"))
    fps = _probe_fps(video_path)
    print(f"      Source fps: {fps}")

    chunks: list[tuple[str, tuple]] = []
    new_segments: list[Segment] = []
    new_time = 0.0
    prev_end = 0.0

    for i, (seg, tts_path) in enumerate(zip(segments, tts_paths)):
        gap = seg.start - prev_end
        if gap > min_gap:
            gap_path = str(tmp_dir / f"gap_{i:05d}.ts")
            chunks.append(("gap", (video_path, prev_end, seg.start, gap_path, fps)))
            new_time += gap

        tts_dur = get_duration_video(tts_path)
        speech_path = str(tmp_dir / f"seg_{i:05d}.ts")
        chunks.append(("speech", (video_path, seg.start, seg.end, tts_path, speech_path, fps)))

        new_start = new_time
        new_time += tts_dur
        new_segments.append(Segment(
            id=seg.id, start=new_start, end=new_time,
            text=seg.text, speaker=seg.speaker,
        ))
        prev_end = seg.end

    trail = total_duration - prev_end
    if trail > min_gap:
        trail_path = str(tmp_dir / "trail.ts")
        chunks.append(("gap", (video_path, prev_end, total_duration, trail_path, fps)))

    # Process chunks in parallel
    total_chunks = len(chunks)
    print(f"      Building {total_chunks} video chunks ({chunk_workers} workers)...")

    def _process(chunk: tuple[str, tuple]) -> None:
        ctype, args = chunk
        if ctype == "gap":
            _make_gap_chunk(*args)
        else:
            _make_speech_chunk(*args)

    done = 0
    with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
        futures = {pool.submit(_process, c): c for c in chunks}
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 25 == 0 or done == total_chunks:
                print(f"      Chunk progress: {done}/{total_chunks}")

    # Concatenate all chunks (output path is second-to-last arg, fps is last)
    concat_file = str(tmp_dir / "concat.txt")
    with open(concat_file, "w") as f:
        for _, args in chunks:
            f.write(f"file '{args[-2]}'\n")

    print("      Concatenating chunks...")
    subprocess.run([
        FFMPEG_EXE,
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c", "copy",
        "-movflags", "+faststart",
        "-y", output_path,
    ], check=True, capture_output=True)

    return new_segments


def generate_srt(segments: list[Segment], output_path: str) -> str:
    def fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg.id + 1}\n")
            f.write(f"{fmt(seg.start)} --> {fmt(seg.end)}\n")
            f.write(f"{seg.text}\n\n")

    return output_path
