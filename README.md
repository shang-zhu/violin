# Violin

Translate educational videos from one language to another using [Together AI](https://www.together.ai). Replaces the audio track with a dubbed voice in the target language and generates a subtitle file — available as a CLI or a REST API.

```bash
uv run main.py lecture.mp4 lecture_zh.mp4 --language Chinese
#example video can be downloaded here: https://html5.stanford.edu/videos/courses/see/CS229/CS229-lecture01.mp4 (Note that you need to trim it to reduce the cost)
#keep the logs
#PYTHONUNBUFFERED=1 uv run main.py examples/CS229-lecture01.mp4 examples/CS229-lecture01_zh.mp4 --language Chinese 2>&1 | tee out.log
#yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" ""
```

## Pipeline

```
Video
  │
  ├─ ffmpeg ──────────────────► Extract audio (16kHz WAV)
  │
  ├─ Together AI Whisper v3 ──► Transcribe → timestamped segments
  │
  ├─ pyannote.audio ──────────► Speaker diarization → per-speaker voice assignment
  │
  ├─ Together AI Qwen3.5-397B ► Translate segments → target language
  │
  ├─ Cartesia Sonic 3 ────────► Synthesize dubbed audio (native-language voices)
  │
  └─ ffmpeg ──────────────────► Merge dubbed audio + original video → output
                                 + SRT subtitle file
```

| Step | Model | Provider |
|------|-------|----------|
| Transcription | `openai/whisper-large-v3` | Together AI |
| Translation | `Qwen/Qwen3.5-397B-A17B` | Together AI |
| TTS | `cartesia/sonic-3` | Together AI |
| Diarization | `pyannote/speaker-diarization-3.1` | Local (HuggingFace) |

## Features

- **42 target languages** via Cartesia Sonic 3
- **Native-language voices** — automatically selects a language-matched voice (e.g. `chinese commercial man` for Chinese, `korean narrator man` for Korean)
- **Speaker diarization** — multiple speakers in the same video each get their own consistent voice
- **Hallucination filtering** — removes Whisper noise segments (`[Music]`, single-word fragments, sub-0.8s clips)
- **Natural dubbing speed** — no aggressive pitch/tempo distortion; segments play at natural TTS speed
- **SRT subtitles** generated alongside every output video
- **Video chat UI** — watch the translated video and ask in-context questions against nearby subtitles plus sampled frames
- **REST API** — submit jobs, poll status, download results over HTTP
- **Single provider** — everything runs through Together AI except local diarization

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- A [Together AI](https://www.together.ai) API key
- A [HuggingFace](https://huggingface.co) token (for speaker diarization)

## Installation

```bash
git clone https://github.com/shang-zhu/Violin
cd Violin
uv sync
cp .env.example .env
# fill in your API keys in .env
```

### HuggingFace model access

Speaker diarization uses gated models. Accept terms at each of these (one-time, free):

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1

## CLI Usage

```bash
# Basic — translate to Spanish
uv run main.py lecture.mp4 lecture_es.mp4 --language Spanish

# Skip diarization (single-speaker video, faster)
uv run main.py lecture.mp4 lecture_zh.mp4 --language Chinese --diarize

# Custom voice
uv run main.py lecture.mp4 lecture_fr.mp4 --language French --voice "french narrator man"

# No subtitles
uv run main.py lecture.mp4 lecture_ja.mp4 --language Japanese --no-subtitles

# Hint the source language for better translation
uv run main.py lecture.mp4 lecture_ko.mp4 --language Korean --source-language English
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--language`, `-l` | *(required)* | Target language name (e.g. `Spanish`, `Japanese`) |
| `--voice`, `-v` | auto | Cartesia Sonic 3 voice. Defaults to the primary native voice for the target language |
| `--source-language` | `auto-detect` | Source language hint for translation |
| `--diarize` | off | Conduct speaker diarization (faster, single voice) |
| `--no-subtitles` | off | Skip SRT generation |

## REST API

Start the server:

```bash
uv run run_api.py
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs  (interactive API docs)
```

Options: `--host`, `--port`, `--reload` (dev mode).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Upload a video and start a translation job |
| `GET` | `/jobs/{id}` | Poll job status and progress |
| `GET` | `/jobs/{id}/video` | Download the dubbed video (when done) |
| `GET` | `/jobs/{id}/srt` | Download the SRT subtitle file (when done) |
| `GET` | `/jobs/{id}/segments` | Fetch aligned subtitle segments for player/chat context |
| `POST` | `/jobs/{id}/chat` | Ask a question about the current video moment using subtitle + visual context |
| `DELETE` | `/jobs/{id}` | Delete a job and its files |
| `GET` | `/languages` | List all supported language names |
| `GET` | `/voices` | List all native voices by language code |
| `GET` | `/voices/{language}` | Voices for a specific language |

### Example

```bash
# Submit a job
JOB=$(curl -s -X POST http://localhost:8000/jobs \
  -F "file=@lecture.mp4" \
  -F "language=Spanish" | jq -r .id)

# Poll until done
curl -s http://localhost:8000/jobs/$JOB | jq '{status, progress}'

# Download results
curl -OJ http://localhost:8000/jobs/$JOB/video
curl -OJ http://localhost:8000/jobs/$JOB/srt
```

Job data is stored under `jobs/{id}/` and persists across server restarts. Use `DELETE /jobs/{id}` to clean up.

## Supported Languages

All 42 languages supported by Cartesia Sonic 3, with native-matched voices where available:

| Language | Native Voice (male / female) |
|----------|------------------------------|
| Chinese | chinese commercial man / chinese female conversational |
| Japanese | japanese male conversational / japanese woman conversational |
| Korean | korean narrator man / korean calm woman |
| Spanish | spanish narrator man / spanish narrator lady |
| French | french narrator man / french narrator lady |
| German | german reporter man / german conversational woman |
| Italian | italian narrator man / italian narrator woman |
| Dutch | dutch confident man / dutch man |
| Russian | russian narrator man 1 / russian narrator woman |
| Portuguese | friendly brazilian man / pleasant brazilian lady |
| Hindi | hindi narrator man / hindi narrator woman |
| Turkish | turkish narrator man / turkish calm man |
| Polish | polish confident man / polish narrator woman |
| Swedish | swedish narrator man / swedish calm lady |
| Arabic | middle eastern woman |
| English + 27 more | tutorial man / helpful woman |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TOGETHER_API_KEY` | Yes | Together AI API key |
| `HF_TOKEN` | Yes (diarization) | HuggingFace token for pyannote models |
| `TOGETHER_TTS_BASE_URL` | No | Custom base URL for Together AI dedicated endpoints |
| `JOBS_DIR` | No | Directory for API job storage (default: `./jobs`) |
| `MAX_WORKERS` | No | Max concurrent API translation jobs (default: `2`) |

## Project Structure

```
Violin/
├── main.py                  # CLI entry point
├── run_api.py               # API server entry point
├── pipeline/
│   ├── extractor.py         # Audio extraction (ffmpeg)
│   ├── transcriber.py       # Whisper transcription + hallucination filtering
│   ├── translator.py        # Qwen3.5-397B translation (batched)
│   ├── tts.py               # Cartesia Sonic 3 TTS + native voice selection
│   ├── diarizer.py          # pyannote speaker diarization
│   ├── merger.py            # Audio assembly + SRT generation + video merge
│   ├── languages.py         # BCP-47 language code mapping
│   └── ffmpeg_utils.py      # Bundled ffmpeg helpers (no system ffmpeg needed)
├── api/
│   ├── app.py               # FastAPI application
│   ├── config.py            # Configuration (JOBS_DIR, MAX_WORKERS)
│   ├── models.py            # Pydantic schemas
│   ├── storage.py           # File-based job storage
│   ├── worker.py            # ThreadPoolExecutor job runner
│   └── routes/
│       ├── jobs.py          # Job lifecycle endpoints
│       ├── files.py         # Video/SRT download endpoints
│       └── catalog.py       # Language and voice catalog endpoints
├── .env.example
└── pyproject.toml
```

## Notes

- **No system ffmpeg required** — bundled via `imageio-ffmpeg`
- **Long videos** — translation is batched in chunks of 60 segments; no hard length limit
- **API job storage** — jobs persist in `jobs/` across server restarts; clean up with `DELETE /jobs/{id}` or by removing the directory

## License

MIT
