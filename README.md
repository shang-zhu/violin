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

## Features

- **42 target languages** via Cartesia Sonic 3
- **Native-language voices** — automatically selects a language-matched voice (e.g. `chinese commercial man` for Chinese, `korean narrator man` for Korean)
- **Hallucination filtering** — removes Whisper noise segments (`[Music]`, single-word fragments, sub-0.8s clips)
- **Natural dubbing speed** — no aggressive pitch/tempo distortion; segments play at natural TTS speed
- **SRT subtitles** generated alongside every output video
- **Video chat UI** — watch the translated video and ask in-context questions against nearby subtitles plus sampled frames
- **REST API** — submit jobs, poll status, download results over HTTP
- **Single provider** — everything runs through Together AI

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- A [Together AI](https://www.together.ai) API key

## Installation

```bash
git clone https://github.com/shang-zhu/Violin
cd Violin
uv sync
cp .env.example .env
# fill in your API keys in .env
```

## CLI Usage

```bash
# Basic — translate to Spanish
uv run main.py lecture.mp4 lecture_es.mp4 --language Spanish

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

## Install as a global CLI

The instructions above get you a working dev checkout (`uv run main.py …`). To install `violin` and `violin-api` as system-wide commands you can run from anywhere:

```bash
# From inside the repo
uv tool install .

# Or, while developing — links to your source so edits take effect immediately
uv tool install --editable .
```

After installing:

```bash
violin lecture.mp4 lecture_zh.mp4 --language Chinese
violin-api --port 8000
```

Update after pulling new changes:

```bash
uv tool install . --reinstall
```

Uninstall:

```bash
uv tool uninstall violin
```

The bundled `config/default.yaml` and `config/prod.yaml` are packaged into the install, so `violin` works correctly from any directory.

## Use with Claude Code

This repo ships a [Claude Code skill](https://code.claude.com/docs/en/skills) at `.claude/skills/video-translator/`. After cloning the repo and configuring `.env`, run `claude` in the project directory and describe the task in natural language:

> Translate examples/lecture.mp4 to Chinese with the academic style

Claude loads the skill automatically, picks the right config / style, runs pre-flight checks (input file exists, required API keys are set), invokes `violin` (or `uv run main.py` if not yet installed globally), and reports the cost summary at the end.

### Global skill (any directory)

If you've installed `violin` globally (see above), you can copy the skill to your user-level skills directory so it loads in **any** Claude Code session, not just inside this repo:

```bash
cp -r .claude/skills/video-translator ~/.claude/skills/
```

After that, `claude` running anywhere will recognize requests like "dub this video" or "generate Chinese subtitles for X.mp4" and call `violin` with the right flags.

**API keys outside the repo.** The project-local `.env` is only auto-loaded when you run `violin` from inside the repo. For the global skill to work in any directory, export the keys in your shell rc file instead:

```bash
# ~/.zshrc or ~/.bashrc
export TOGETHER_API_KEY="..."
export OPENAI_API_KEY="..."        # only needed for config/prod.yaml
export ELEVENLABS_API_KEY="..."    # only needed for config/prod.yaml
```

Reload (`source ~/.zshrc`) and `violin` will pick the keys up from anywhere.

## License

MIT
