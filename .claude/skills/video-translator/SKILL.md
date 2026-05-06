---
name: video-translator
description: Dub a video into another language and generate subtitles using the default Together + Cartesia stack. Trigger when the user wants to translate / dub / voice-over a video file, or generate subtitles for it. Handles `.mp4` / `.mkv` / `.webm`. Installs as the `violin` CLI (and `violin-api` for the FastAPI server) via `uv tool install`. For premium models (OpenAI / ElevenLabs) or custom configs, point the user to the repo.
allowed-tools: Bash, Read
---

# Violin — operating skill

This skill drives the Violin pipeline (audio extract → Whisper → LLM translate → TTS → ffmpeg merge). The README has the full feature list; this file tells you **how to drive the tool, not what it is**.

## When to use

Fire this skill when the user asks for any of:
- "translate this video to <lang>" / "dub <file> in <lang>"
- "generate subtitles for <file>" (SRT only — see "Subtitles only" below)
- "make a Chinese / Spanish / Japanese / etc. version of <file>"
- "voice-over this video"

Do **not** fire it for: audio-only translation, live transcription, or video editing tasks unrelated to language translation.

## Decision tree

Before running anything, decide on each axis. Ask the user only if the signal is genuinely ambiguous — don't interrogate.

This skill always uses the **default config** (Together for translation, Cartesia for TTS). If the user asks for premium quality, OpenAI, ElevenLabs, or custom model configs, **stop and direct them to the Violin repo** — those flows aren't supported through the global CLI.

1. **CLI vs API server**
   - One file, run-and-wait → **CLI** (`violin …`).
   - Multiple jobs, web UI integration, or the user explicitly mentions HTTP / API → **API server** (`violin-api …`). Don't auto-start it; print the command for the user to run (per memory `feedback_running_services`).

2. **Style** (`--style …`)
   - Default `standard` unless user signals otherwise: kids content → `kids`, lecture / formal → `academic`, casual chat → `casual`, dramatic narration → `storyteller`, news clip → `news`. Run `violin --style list` to enumerate if unsure.

3. **Voiceover mode**
   - Default = mix dubbed audio over a quiet original track (`voiceover` on). Keep it on.
   - Switch to `--no-voiceover` only when the user explicitly says "replace audio entirely" / "no original audio".

4. **Subtitles only** (no dubbing)
   - The CLI does not have a "subtitles only" mode — translation requires the full pipeline. If the user wants only an SRT, run the full pipeline anyway and hand them just the `.srt`; warn them of the cost. Don't invent flags that don't exist.

## Pre-flight checks (run these silently before invoking)

```bash
# 1. Confirm `violin` is on PATH
command -v violin
# If missing: tell the user to `uv tool install .` from a Violin checkout
# (or `uv tool install violin` once it's on PyPI). Do NOT auto-install.

# 2. Confirm input exists
test -f "<input>" || abort

# 3. Verify TOGETHER_API_KEY is set (the only key the default stack needs)
printenv TOGETHER_API_KEY
# Source:
# - Inside the Violin repo: `.env` is auto-loaded by python-dotenv
# - Outside the repo: must come from the shell environment (e.g. exported in
#   ~/.zshrc / ~/.bashrc). `.env` will NOT be found.
```

If `TOGETHER_API_KEY` is missing, **stop and tell the user how to set it**:
- If they're inside the Violin repo → populate `.env`
- If they're using `violin` from another directory → `export TOGETHER_API_KEY=...` in `~/.zshrc` / `~/.bashrc` and `source` it

Do not run a doomed command.

## Running the CLI

```bash
PYTHONUNBUFFERED=1 violin <input> <output> --language <Lang> [flags] 2>&1 | tee <output>.log
```

- Always `PYTHONUNBUFFERED=1` + `tee` — runs are minutes long; users want live progress and a saved log.
- Quote paths with spaces.
- Output naming: if user didn't specify, default to `<input-stem>_<lang-code>.<ext>` (e.g. `lecture.mp4` → `lecture_zh.mp4`).

## After the run

Report back:
- Output video path and SRT path (the run prints them).
- Cost summary (printed at end — surface the total, don't hide it).
- If `voiceover` was on, mention the `_original.m4a` sidecar file exists.

If the run failed mid-pipeline, the temp dir under `/tmp/vidtrans_*` is cleaned up automatically — don't try to recover from there.

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `TOGETHER_API_KEY ... is not set` | Key not in env | If in repo: populate `.env`. If global: `export` in `~/.zshrc`/`~/.bashrc` and `source` it. |
| Whisper / Together request 401 / 429 | Bad or rate-limited Together key | Surface the error; don't retry blindly |
| Cartesia TTS error | Together-hosted Cartesia issue (rate limit, transient) | Surface the error; suggest re-running |
| ffmpeg merge fails on `.mkv` with weird codec | Source has an unusual stream layout | Suggest re-encoding to `.mp4` first |
| Output video is silent | `--no-voiceover` chosen but TTS step failed | Re-run without `--no-voiceover` to fall back to mixed audio |

## Don'ts

- Don't auto-run `uv tool install` or `uv sync` — if `violin` isn't on PATH, tell the user the install command and stop.
- Don't run on multi-GB videos without first telling the user the rough cost (audio length × pricing in `config/default.yaml`'s `pricing:` section).
- Don't try to switch to OpenAI or ElevenLabs from this skill. If the user asks for premium models or custom configs, tell them to clone the Violin repo and use it directly with `--config config/prod.yaml`.
- Don't paraphrase the README at the user; if they ask "what does this tool do", just point them at `README.md`.

## Reference

For exhaustive flag docs, supported languages (42), and voice catalog: read `README.md` or run `violin --help`. Style profiles and their LLM directives live in `config/default.yaml` under `styles:`.
