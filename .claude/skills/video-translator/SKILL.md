---
name: video-translator
description: Dub a video into another language and generate subtitles. Trigger when the user wants to translate / dub / voice-over a video file, or generate subtitles for it. Handles `.mp4` / `.mkv` / `.webm`. Installs as the `violin` CLI (and `violin-api` for the FastAPI server) via `uv tool install`.
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

1. **CLI vs API server**
   - One file, run-and-wait → **CLI** (`violin …`).
   - Multiple jobs, web UI integration, or the user explicitly mentions HTTP / API → **API server** (`violin-api …`). Don't auto-start it; print the command for the user to run (per memory `feedback_running_services`).

2. **Config: `default.yaml` vs `prod.yaml`**
   - Default is Together + Cartesia (cheap/fast).
   - `--config config/prod.yaml` is OpenAI + ElevenLabs (premium quality, much higher cost).
   - Pick `prod.yaml` only if the user mentions "best quality", "premium", "ElevenLabs", or "OpenAI". Otherwise default.

3. **Style** (`--style …`)
   - Default `standard` unless user signals otherwise: kids content → `kids`, lecture / formal → `academic`, casual chat → `casual`, dramatic narration → `storyteller`, news clip → `news`. Run `violin --style list` to enumerate if unsure.

4. **Voiceover mode**
   - Default = mix dubbed audio over a quiet original track (`voiceover` on). Keep it on.
   - Switch to `--no-voiceover` only when the user explicitly says "replace audio entirely" / "no original audio".

5. **Subtitles only** (no dubbing)
   - The CLI does not have a "subtitles only" mode — translation requires the full pipeline. If the user wants only an SRT, run the full pipeline anyway and hand them just the `.srt`; warn them of the cost. Don't invent flags that don't exist.

## Pre-flight checks (run these silently before invoking)

```bash
# 1. Confirm `violin` is on PATH
command -v violin || echo "Tell user to run: uv tool install --editable . (from the Violin repo)"

# 2. Confirm input exists
test -f "<input>" || abort

# 3. Read the active config to know which providers are needed
grep -E "provider:|model:" config/<chosen>.yaml

# 4. Verify the required env vars based on config
#    - Always: TOGETHER_API_KEY
#    - If translation.provider=openai → OPENAI_API_KEY
#    - If tts.provider=elevenlabs    → ELEVENLABS_API_KEY
#    .env is loaded automatically; check `printenv` or `grep` the file.
```

If a required key is missing, **stop and tell the user which key to set** — do not run a doomed command.

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
| `TOGETHER_API_KEY ... is not set` | `.env` missing or env not loaded | Tell user to populate `.env` |
| Whisper request 401 / 429 | Bad or rate-limited Together key | Surface the error; don't retry blindly |
| ElevenLabs `quota_exceeded` | Out of credits on premium config | Suggest switching back to default config |
| ffmpeg merge fails on `.mkv` with weird codec | Source has an unusual stream layout | Suggest re-encoding to `.mp4` first |
| Output video is silent | `--no-voiceover` chosen but TTS step failed | Re-run without `--no-voiceover` to fall back to mixed audio |

## Don'ts

- Don't auto-run `uv tool install` or `uv sync` — if `violin` isn't on PATH, tell the user the install command and stop.
- Don't run on multi-GB videos without first telling the user the rough cost (audio length × pricing in `config/default.yaml`'s `pricing:` section).
- Don't suggest editing `config/default.yaml` to change models for a one-off run — pass `--config` instead.
- Don't paraphrase the README at the user; if they ask "what does this tool do", just point them at `README.md`.

## Reference

For exhaustive flag docs, supported languages (42), and voice catalog: read `README.md` or run `violin --help`. Style profiles and their LLM directives live in `config/default.yaml` under `styles:`.
