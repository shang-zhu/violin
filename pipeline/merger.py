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


def _atempo_chain(speed: float) -> str:
    """Build an ffmpeg atempo filter chain for arbitrary speed values.

    Each atempo instance supports 0.5–100.0, but best quality is 0.5–2.0.
    """
    if 0.5 <= speed <= 2.0:
        return f"atempo={speed}"
    parts: list[str] = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining}")
    return ",".join(parts)


def _make_speech_chunk(
    video_path: str, start: float, end: float,
    tts_path: str, out_path: str, fps: float | None = None,
    tts_dur: float | None = None,
    voiceover: bool = False,
    mix_volume: float = 0.0,
) -> None:
    """Extract video segment, speed-adjust to match TTS duration, mux with TTS audio.

    *voiceover*: TTS-only in the video; caller builds original audio separately.
    *mix_volume*: bake original audio into the video at this level (0.0–1.0).
    Only one of voiceover / mix_volume should be used.
    """
    vcfg = _conf.get()["merge_video"]
    if fps is None:
        fps = vcfg["output_fps"]
    orig_dur = end - start
    if tts_dur is None:
        tts_dur = get_duration_video(tts_path)
    if orig_dur < 0.01 or tts_dur < 0.01:
        speed = 1.0
    else:
        speed = max(vcfg["speed_clamp_min"], min(vcfg["speed_clamp_max"], orig_dur / tts_dur))
    target_dur = orig_dur / speed

    coarse = max(0, start - _SEEK_PAD)
    fine = start - coarse

    vol = max(0.0, min(1.0, mix_volume))
    if vol > 0 and not voiceover:
        atempo = _atempo_chain(speed) if speed != 1.0 else ""
        atempo_part = f",{atempo}" if atempo else ""
        filter_complex = (
            f"[0:v]trim=start={fine}:duration={orig_dur},setpts=(PTS-STARTPTS)/{speed},fps=fps={fps}[v];"
            f"[0:a]atrim=start={fine}:duration={orig_dur},asetpts=PTS-STARTPTS"
            f"{atempo_part},loudnorm=I=-23:TP=-1.5:LRA=11,volume={vol}[orig];"
            f"[1:a:0]apad[tts];"
            f"[orig][tts]amix=inputs=2:duration=shortest[a]"
        )
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        filter_complex = (
            f"[0:v]trim=start={fine}:duration={orig_dur},setpts=(PTS-STARTPTS)/{speed},fps=fps={fps}[v]"
        )
        maps = ["-map", "[v]", "-map", "1:a:0"]

    subprocess.run([
        FFMPEG_EXE,
        "-ss", str(coarse), "-t", str(orig_dur + fine + 0.5), "-i", video_path,
        "-i", tts_path,
        "-filter_complex", filter_complex,
        *maps,
        "-c:v", "libx264", "-preset", vcfg["preset"], "-crf", str(vcfg["crf"]),
        "-c:a", "aac", "-ar", str(_SAMPLE_RATE), "-ac", "1",
        "-t", str(target_dur),
        "-f", "mpegts",
        "-y", out_path,
    ], check=True, capture_output=True)


def _make_orig_audio_chunk(
    video_path: str, start: float, end: float,
    out_path: str, speed: float = 1.0,
) -> None:
    """Extract original audio from a video segment, speed-adjusted, as raw WAV."""
    orig_dur = end - start
    coarse = max(0, start - _SEEK_PAD)
    fine = start - coarse

    atempo = _atempo_chain(speed) if speed != 1.0 else ""
    af_parts = [f"atrim=start={fine}:duration={orig_dur}", "asetpts=PTS-STARTPTS"]
    if atempo:
        af_parts.append(atempo)
    af = ",".join(af_parts)

    subprocess.run([
        FFMPEG_EXE,
        "-ss", str(coarse), "-t", str(orig_dur + fine + 0.5), "-i", video_path,
        "-af", af,
        "-c:a", "pcm_s16le", "-ar", str(_SAMPLE_RATE), "-ac", "1",
        "-vn",
        "-y", out_path,
    ], check=True, capture_output=True)


def _make_silence_audio_chunk(duration: float, out_path: str) -> None:
    """Create a silent audio-only WAV chunk of given duration."""
    subprocess.run([
        FFMPEG_EXE,
        "-f", "lavfi", "-i", f"anullsrc=r={_SAMPLE_RATE}:cl=mono",
        "-c:a", "pcm_s16le", "-ar", str(_SAMPLE_RATE), "-ac", "1",
        "-t", str(duration),
        "-y", out_path,
    ], check=True, capture_output=True)


def _make_gap_chunk(
    video_path: str, start: float, end: float,
    out_path: str, fps: float | None = None,
    preserve_audio: bool = False,
) -> None:
    """Extract gap video at original speed.

    When *preserve_audio* is True the original audio track is kept (used to
    retain non-translated speakers).  Otherwise the audio is replaced with
    silence.
    """
    vcfg = _conf.get()["merge_video"]
    if fps is None:
        fps = vcfg["output_fps"]
    dur = end - start
    coarse = max(0, start - _SEEK_PAD)
    fine = start - coarse

    if preserve_audio:
        subprocess.run([
            FFMPEG_EXE,
            "-ss", str(coarse), "-t", str(dur + fine + 0.5), "-i", video_path,
            "-filter_complex",
            (f"[0:v]trim=start={fine}:duration={dur},setpts=PTS-STARTPTS,fps=fps={fps}[v];"
             f"[0:a]atrim=start={fine}:duration={dur},asetpts=PTS-STARTPTS[a]"),
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", vcfg["preset"], "-crf", str(vcfg["crf"]),
            "-c:a", "aac", "-ar", str(_SAMPLE_RATE), "-ac", "1",
            "-t", str(dur),
            "-f", "mpegts",
            "-y", out_path,
        ], check=True, capture_output=True)
    else:
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


def prepare_merge(
    video_path: str,
    segments: list[Segment],
    total_duration: float,
    preserve_gap_audio: bool = False,
    original_audio_volume: float = 0.0,
    mix_volume: float = 0.0,
) -> "MergePlan":
    """Plan the merge: probe fps, create temp dir, and identify gap chunks.

    When *preserve_gap_audio* is True, gap chunks keep the original audio track
    from the source video (for main-speaker-only translation).

    When *original_audio_volume* > 0 the caller intends to build a separate
    original audio track (webapp mode — browser-side mixing).

    When *mix_volume* > 0 the original audio is baked into each speech chunk
    at that volume (CLI mode — single self-contained file).

    Returns a MergePlan that can be used to pre-build gap chunks before TTS
    finishes (since gaps don't depend on TTS output).
    """
    vcfg = _conf.get()["merge_video"]
    min_gap = vcfg["min_gap"]

    tmp_dir = Path(tempfile.mkdtemp(prefix="vidmerge_"))
    fps = _probe_fps(video_path)

    gap_tasks: list[tuple] = []
    prev_end = 0.0
    for i, seg in enumerate(segments):
        gap = seg.start - prev_end
        if gap > min_gap:
            gap_path = str(tmp_dir / f"gap_{i:05d}.ts")
            gap_tasks.append((video_path, prev_end, seg.start, gap_path, fps, preserve_gap_audio))
        prev_end = seg.end

    trail = total_duration - prev_end
    if trail > min_gap:
        trail_path = str(tmp_dir / "trail.ts")
        gap_tasks.append((video_path, prev_end, total_duration, trail_path, fps, preserve_gap_audio))

    return MergePlan(
        video_path=video_path,
        segments=segments,
        total_duration=total_duration,
        fps=fps,
        tmp_dir=tmp_dir,
        gap_tasks=gap_tasks,
        preserve_gap_audio=preserve_gap_audio,
        original_audio_volume=original_audio_volume,
        mix_volume=mix_volume,
    )


def build_gap_chunks(plan: "MergePlan", workers: int | None = None) -> None:
    """Build all gap .ts files in parallel. Safe to call while TTS is running."""
    if workers is None:
        workers = _conf.get()["merge_video"]["workers"]
    if not plan.gap_tasks:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_make_gap_chunk, *args) for args in plan.gap_tasks]
        for f in as_completed(futures):
            f.result()


class MergePlan:
    """Holds pre-computed merge metadata so gap chunks can be built early."""
    __slots__ = ("video_path", "segments", "total_duration", "fps", "tmp_dir",
                 "gap_tasks", "preserve_gap_audio", "original_audio_volume",
                 "mix_volume")

    def __init__(self, video_path, segments, total_duration, fps, tmp_dir,
                 gap_tasks, preserve_gap_audio=False, original_audio_volume=0.0,
                 mix_volume=0.0):
        self.video_path = video_path
        self.segments = segments
        self.total_duration = total_duration
        self.fps = fps
        self.tmp_dir = tmp_dir
        self.gap_tasks = gap_tasks
        self.preserve_gap_audio = preserve_gap_audio
        self.original_audio_volume = original_audio_volume
        self.mix_volume = mix_volume


def build_aligned_video(
    video_path: str,
    segments: list[Segment],
    tts_paths: list[str],
    total_duration: float,
    output_path: str,
    tts_durations: list[float] | None = None,
    merge_plan: MergePlan | None = None,
    original_audio_path: str | None = None,
) -> list[Segment]:
    """Build video with per-segment speed adjustment so video matches TTS audio timing.

    Args:
        tts_durations: Pre-computed WAV durations — avoids re-probing each file.
        merge_plan: From ``prepare_merge``; if provided, gap chunks are assumed
                    already built (via ``build_gap_chunks``) and are reused.
        original_audio_path: When set, also produce a separate .m4a file with
                    the original audio aligned to the new timeline (for
                    browser-side voice-over mixing).

    Returns segments with updated timestamps matching the new timeline.
    """
    vcfg = _conf.get()["merge_video"]
    min_gap = vcfg["min_gap"]
    chunk_workers = vcfg["workers"]

    if merge_plan is not None:
        tmp_dir = merge_plan.tmp_dir
        fps = merge_plan.fps
        gaps_already_built = True
        preserve_audio = merge_plan.preserve_gap_audio
        voiceover = merge_plan.original_audio_volume > 0
        mix_vol = merge_plan.mix_volume
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="vidmerge_"))
        fps = _probe_fps(video_path)
        gaps_already_built = False
        preserve_audio = False
        voiceover = False
        mix_vol = 0.0

    print(f"      Source fps: {fps}")

    chunks: list[tuple[str, tuple]] = []
    # (type, out_path, duration, speed) — metadata for building the original audio track
    audio_plan: list[tuple[str, float, float, float, float]] = []
    new_segments: list[Segment] = []
    new_time = 0.0
    prev_end = 0.0

    for i, (seg, tts_path) in enumerate(zip(segments, tts_paths)):
        gap = seg.start - prev_end
        if gap > min_gap:
            gap_path = str(tmp_dir / f"gap_{i:05d}.ts")
            chunks.append(("gap", (video_path, prev_end, seg.start, gap_path, fps, preserve_audio)))
            audio_plan.append(("silence", prev_end, seg.start, gap, 1.0))
            new_time += gap

        tts_dur = tts_durations[i] if tts_durations else get_duration_video(tts_path)
        speech_path = str(tmp_dir / f"seg_{i:05d}.ts")
        chunks.append(("speech", (video_path, seg.start, seg.end, tts_path, speech_path, fps, tts_dur, voiceover, mix_vol)))

        orig_dur = seg.end - seg.start
        if orig_dur < 0.01 or tts_dur < 0.01:
            clamped_speed = 1.0
        else:
            clamped_speed = max(vcfg["speed_clamp_min"],
                                min(vcfg["speed_clamp_max"], orig_dur / tts_dur))
        chunk_dur = orig_dur / clamped_speed

        audio_plan.append(("speech", seg.start, seg.end, chunk_dur, clamped_speed))

        new_start = new_time
        new_time += chunk_dur
        new_segments.append(Segment(
            id=seg.id, start=new_start, end=new_time,
            text=seg.text, speaker=seg.speaker,
        ))
        prev_end = seg.end

    trail = total_duration - prev_end
    if trail > min_gap:
        trail_path = str(tmp_dir / "trail.ts")
        chunks.append(("gap", (video_path, prev_end, total_duration, trail_path, fps, preserve_audio)))
        audio_plan.append(("silence", prev_end, total_duration, trail, 1.0))

    chunks_to_build = (
        [c for c in chunks if c[0] != "gap"] if gaps_already_built else chunks
    )

    total_chunks = len(chunks)
    build_count = len(chunks_to_build)
    if gaps_already_built:
        print(f"      {total_chunks - build_count} gap chunks pre-built; "
              f"building {build_count} speech chunks ({chunk_workers} workers)...")
    else:
        print(f"      Building {total_chunks} video chunks ({chunk_workers} workers)...")

    def _process(chunk: tuple[str, tuple]) -> None:
        ctype, args = chunk
        if ctype == "gap":
            _make_gap_chunk(*args)
        else:
            _make_speech_chunk(*args)

    done = 0
    with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
        futures = {pool.submit(_process, c): c for c in chunks_to_build}
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 25 == 0 or done == build_count:
                print(f"      Chunk progress: {done}/{build_count}")

    concat_file = str(tmp_dir / "concat.txt")
    with open(concat_file, "w") as f:
        for ctype, args in chunks:
            if ctype == "gap":
                f.write(f"file '{args[3]}'\n")
            else:
                f.write(f"file '{args[4]}'\n")

    print("      Concatenating chunks...")
    subprocess.run([
        FFMPEG_EXE,
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c", "copy",
        "-movflags", "+faststart",
        "-y", output_path,
    ], check=True, capture_output=True)

    if original_audio_path and (voiceover or mix_vol > 0):
        _build_original_audio_track(video_path, audio_plan, tmp_dir,
                                    original_audio_path, chunk_workers)

    return new_segments


def _build_original_audio_track(
    video_path: str,
    audio_plan: list[tuple],
    tmp_dir: Path,
    output_path: str,
    workers: int,
) -> None:
    """Build a separate .m4a with original audio aligned to the new timeline.

    Speech segments get the original audio (speed-adjusted); gaps are silent
    (since the main video already carries original audio during gaps).
    """
    print("      Building original audio track for voice-over…")

    def _build_audio_chunk(idx: int, entry: tuple) -> str:
        atype, start, end, duration, speed = entry
        out = str(tmp_dir / f"oa_{idx:05d}.wav")
        if atype == "silence":
            _make_silence_audio_chunk(duration, out)
        else:
            _make_orig_audio_chunk(video_path, start, end, out, speed)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_audio_chunk, i, e): i
                   for i, e in enumerate(audio_plan)}
        results: dict[int, str] = {}
        for f in as_completed(futures):
            idx = futures[f]
            results[idx] = f.result()

    audio_chunks = [results[i] for i in range(len(audio_plan))]

    concat_file = str(tmp_dir / "oa_concat.txt")
    with open(concat_file, "w") as f:
        for path in audio_chunks:
            f.write(f"file '{path}'\n")

    subprocess.run([
        FFMPEG_EXE,
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c:a", "aac", "-ar", str(_SAMPLE_RATE), "-ac", "1",
        "-movflags", "+faststart",
        "-vn",
        "-y", output_path,
    ], check=True, capture_output=True)
    print("      Original audio track ready.")


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
