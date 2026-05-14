# 🎻 Violin

**Open-source Video Translation Tool.**

<p align="left">
  <img src="assets/logo.png" alt="Violin logo" width="256">
</p>

[🌐 Live demo](https://violin-ai.com) · [📜 MIT License](https://github.com/shang-zhu/violin/blob/main/LICENSE)

<!-- ![demo](assets/outcome.png) -->

Upload a video. Violin transcribes the speech, translates it, synthesizes a native-sounding voice-over in the target language, and remuxes it back into the video — fully aligned, with optional SRT subtitles.

Available as a **CLI**, a **FastAPI web app**, and a **Claude Code skill**.

---

## ✨ Features

- **33 target languages** with handpicked native-speaker voices for the 16 most-used ones (Cartesia Sonic 3 + ElevenLabs)
- **In-video Q&A** — ask questions about any moment in the dubbed video; answers use nearby subtitles plus sampled frames
- **Natural-language voice picker** — describe the voice you want, an LLM picks from the catalog
- **6 style profiles** *(experimental)* — standard / kids / academic / casual / storyteller / news
- **Pluggable stack** — Together / OpenAI / ElevenLabs interchangeable for every stage, one YAML

---

## 🚀 Quick start

### Try it without installing anything

The live demo runs at **<https://violin-ai.com>** — drop a short clip in, get a dubbed video out in a few minutes.

### Run locally

Requires **Python 3.10+** and **ffmpeg** on PATH.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if you don't have it
uv tool install violin                            # recommended — faster, isolated
# or: pip install violin                          # if you'd rather install into your current Python env

export TOGETHER_API_KEY=...                       # get one at https://api.together.ai (add to ~/.zshrc to persist)
```

Three ways to use it:

**1. CLI** — translate one file:

```bash
violin lecture.mp4 lecture_zh.mp4 --language Chinese
```

**2. Web app** — full REST API + browser UI:

```bash
violin-api
# → http://127.0.0.1:8000           (browser UI)
# → http://127.0.0.1:8000/docs      (interactive API docs)
```

**3. Claude Code skill** — invoke from any Claude Code session:

```bash
violin --install-skill          # one-time: copies the skill into ~/.claude/skills/
claude
> please use the violin skill to translate path/to/video.mp4 into Chinese
```

<details><summary>Run from source (for hacking on the pipeline)</summary>

```bash
git clone https://github.com/shang-zhu/violin.git
cd violin
uv sync
cp .env.example .env             # then fill in TOGETHER_API_KEY
uv run main.py lecture.mp4 lecture_zh.mp4 --language Chinese
```

To use the `violin` / `violin-api` commands globally while edits to your local source reflect immediately, install editable:

```bash
uv tool uninstall violin     # if you've installed the PyPI version
uv tool install --editable .
```

After this, `violin` / `violin-api` run from your local checkout — edit any file and the next invocation picks it up; no rebuild needed. To switch back to PyPI: `uv tool uninstall violin && uv tool install violin`.

</details>

---

## 🎬 How Violin works

```
Video
  │
  ├─ ffmpeg ─────────────────────► Extract audio (16 kHz WAV)
  │
  ├─ Whisper Large v3 ────────────► Word-level timestamps → sentence segments
  │
  ├─ LLM (DeepSeek V4 Pro by default) ──► Translate each segment, respecting style profile
  │
  ├─ TTS (Cartesia Sonic 3 by default) ─► Synthesize dubbed audio per segment
  │
  └─ ffmpeg ─────────────────────► Speed-align video to dubbed audio,
                                    concat with freeze-frame fallback,
                                    single-pass AAC encode the audio track,
                                    write output mp4 + optional SRT
```

---

## ⚙️ Configuration

Override any default by writing your own YAML and passing it with `--config my.yaml` — only the keys you want to change need to appear; values deep-merge with the [built-in defaults](https://github.com/shang-zhu/violin/blob/main/config/default.yaml).

### Switch providers

```yaml
# config/default.yaml — pick the stack you want
models:
  transcription:
    provider: together                  # together | openai
    model: openai/whisper-large-v3      # together → openai/whisper-large-v3 | openai → whisper-1
  translation:
    provider: together                  # together | openai
    model: deepseek-ai/DeepSeek-V4-Pro  # together → deepseek-ai/DeepSeek-V4-Pro | openai → gpt-5.5
  tts:
    provider: together                  # together | elevenlabs | openai
    model: cartesia/sonic-3             # together → cartesia/sonic-3 | elevenlabs → eleven_v3 | openai → tts-1-hd
```

### Production overrides

A starter `config/prod.yaml` is included for public deployments. It adds upload limits, serializes jobs, and caps ffmpeg concurrency. The included `Dockerfile` + `docker-compose.yml` + `Caddyfile` are how the live demo is hosted — `docker compose up -d --build` after filling `.env` is enough to put a copy of Violin behind auto-HTTPS on any Docker host.

### Environment variables

| Variable | When required | Description |
|----------|---------------|-------------|
| `TOGETHER_API_KEY` | **Recommended** — covers every stage with the default config | Together AI API key |
| `OPENAI_API_KEY` | Any stage uses `provider: openai` | Covers `whisper-1`, GPT models, and `tts-1` |
| `ELEVENLABS_API_KEY` | TTS uses `provider: elevenlabs` | ElevenLabs API key |
| `CORS_ORIGINS` | Optional | Comma-separated allowed origins (default: `*`) |

> You only need keys for the providers you actually pick. Pure-OpenAI deployments (all stages on `openai`) work too — `OPENAI_API_KEY` alone is enough. Same idea for ElevenLabs.

---

## 🎭 Style profiles

Six built-in profiles tune both the translation LLM prompt and the TTS delivery. Use `--style <name>` on the CLI or pass `style` in API requests.

| Style | Tone | TTS speed | Emotion |
|-------|------|-----------|---------|
| `standard` | Faithful translation, natural voice | 1.0× | — |
| `kids` | Rewritten for a 7-year-old, plain language | 1.0× | excited |
| `academic` | Formal register, preserves jargon and honorifics | 0.95× | calm |
| `casual` | Spoken slang, contractions, friendly | 1.1× | content |
| `storyteller` | Vivid, dramatic narration | 0.9× | enthusiastic |
| `news` | Concise, declarative, broadcast-style | 1.0× | neutral |

Add your own by editing `prompts/styles.yaml`.

See all available styles: `violin --style list`.

---

## 💻 CLI usage

> Examples use the PyPI-installed `violin` command. If you're running from a git checkout, substitute `uv run main.py` for `violin` (and `uv run run_api.py` for `violin-api`).

```bash
# Basic
violin lecture.mp4 lecture_es.mp4 --language Spanish

# Pick a style
violin talk.mp4 talk_zh.mp4 --language Chinese --style kids

# Pick a specific voice
violin lecture.mp4 lecture_fr.mp4 --language French --voice "french narrator man"

# Skip SRT
violin lecture.mp4 lecture_ja.mp4 --language Japanese --no-subtitles

# Full replacement (no original audio underneath)
violin lecture.mp4 lecture_ko.mp4 --language Korean --no-voiceover

# Custom config (e.g. switch to OpenAI/ElevenLabs)
violin lecture.mp4 lecture_it.mp4 --language Italian --config config/other_api.yaml
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--language` / `-l` | *(required)* | Target language name (e.g. `Spanish`, `Japanese`) |
| `--voice` / `-v` | auto | TTS voice. Defaults to the primary native voice for the target language |
| `--source-language` | `auto-detect` | Source language hint for translation |
| `--no-subtitles` | off | Skip SRT generation |
| `--voiceover` / `--no-voiceover` | voiceover on | Keep original audio underneath the dub, or full replacement |
| `--style` / `-s` | `standard` | Style profile name. Use `--style list` to see all |
| `--config` / `-c` | `config/default.yaml` | Path to a YAML override file |
| `--timings-out` | off | Write per-step wall-clock timings + cost as JSON |

---

## 🛰️ Web app & REST API

```bash
violin-api                              # default dev mode
violin-api --host 0.0.0.0 --port 8080   # bind everywhere
violin-api --config config/prod.yaml    # production overrides (requires a git checkout for config/prod.yaml)
```

Core flow: `POST /jobs` to start, `GET /jobs/{id}` to poll, `GET /jobs/{id}/video` and `/srt` to download, `POST /jobs/{id}/chat` for in-video Q&A. Full list with request/response schemas at **`/docs`**.

### Example

```bash
# Submit
JOB=$(curl -s -X POST http://localhost:8000/jobs \
  -F "file=@lecture.mp4" \
  -F "language=Spanish" \
  -F "style=academic" | jq -r .id)

# Poll
curl -s http://localhost:8000/jobs/$JOB | jq '{status, progress}'

# Download
curl -OJ http://localhost:8000/jobs/$JOB/video
curl -OJ http://localhost:8000/jobs/$JOB/srt
```

Job data lives under `jobs/{id}/`. Set `api.job_ttl_hours` to auto-delete jobs older than N hours (default `0` = disabled; `config/prod.yaml` uses 24h for the public demo).

---

## 🌍 Supported languages

Violin supports **33 target languages**. The 16 below ship with handpicked native-speaker voices for each provider; the rest fall back to the English voice catalog (which is multilingual under both Cartesia Sonic 3 and ElevenLabs `eleven_v3`).

Ordered by native-speaker population.

| Language | Cartesia native voice (M / F) | ElevenLabs native voice (M / F) |
|----------|-------------------------------|---------------------------------|
| Chinese | chinese commercial man / chinese female conversational | Lin / Lingyue |
| Spanish | spanish narrator man / spanish narrator lady | Carlos / Valeria |
| English | tutorial man / helpful woman | Adam / Sarah |
| Hindi | hindi narrator man / hindi narrator woman | Yatin / Madhusmita |
| Arabic | middle eastern woman | Faris / Haneen |
| Portuguese | friendly brazilian man / pleasant brazilian lady | Medeiros / Luna |
| Russian | russian narrator man 1 / russian narrator woman | Ivo / Xenia |
| Japanese | japanese male conversational / japanese woman conversational | Shohei / Maiko |
| Turkish | turkish narrator man / turkish calm man | Sinan / Aura |
| German | german reporter man / german conversational woman | Daniel / Sina |
| Korean | korean narrator man / korean calm woman | Joon-ho / Soo |
| French | french narrator man / french narrator lady | Lior / Virginie |
| Italian | italian narrator man / italian narrator woman | Raffaele / Chiara |
| Polish | polish confident man / polish narrator woman | Gregor / Jola |
| Dutch | dutch confident man / dutch man | Ronald / Jolanda |
| Swedish | swedish narrator man / swedish calm lady | Andreas / Louise |

The 17 fallback languages (using the English voice catalog), also ordered by native speakers: Vietnamese, Tamil, Indonesian, Malay, Ukrainian, Romanian, Thai, Greek, Hungarian, Catalan, Czech, Bulgarian, Danish, Slovak, Croatian, Finnish, Norwegian.

---

## 🤝 Contributing

PRs welcome. Got questions or hit a bug? Email **<heyviolinai@gmail.com>** or open an issue.

---

## ⚠️ Disclaimer

This is a personal open-source project, not a Together AI product. Users are responsible for ensuring they have the right to download and translate any content they process. Designed for Creative Commons, public domain, your own recordings, and other content you have permission to use.

---

## 📜 License

[MIT](https://github.com/shang-zhu/violin/blob/main/LICENSE) — use it freely, including commercially.

---

## 🙏 Acknowledgements

Built on top of [Together AI](https://together.ai), [Whisper](https://github.com/openai/whisper), [Cartesia Sonic 3](https://cartesia.ai), [ElevenLabs](https://elevenlabs.io), [FastAPI](https://fastapi.tiangolo.com/), and [ffmpeg](https://ffmpeg.org).
