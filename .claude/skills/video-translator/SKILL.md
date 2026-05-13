---
name: video-translator
description: Dub a video into another language and generate subtitles using the default Together + Cartesia stack. Trigger when the user wants to translate / dub / voice-over a video file, or generate subtitles for it. Handles `.mp4` / `.mkv` / `.webm`. Installs as the `violin` CLI (and `violin-api` for the FastAPI server) via `uv tool install`. For alternative models (OpenAI / ElevenLabs) or custom configs, point the user to the repo: https://github.com/shang-zhu/violin.
allowed-tools: Bash, Read
---

# Violin — operating skill

Always uses the default config (Together for translation, `cartesia/sonic-3` for TTS). If the user asks for OpenAI, ElevenLabs, or custom configs, **stop and point them to the Violin repo** — those flows aren't supported through the global CLI.

## Pre-flight

Run these silently first. Abort if any fails:

```bash
command -v violin                 # 1. CLI on PATH
test -f "<input>"                 # 2. Input exists
printenv TOGETHER_API_KEY         # 3. Key available
```

If `violin` is missing: tell the user to `uv tool install violin`, then `violin --install-skill` to refresh this skill file. Do not auto-install.

If `TOGETHER_API_KEY` is missing:
- Inside the Violin repo → populate `.env` (auto-loaded)
- Elsewhere → `export TOGETHER_API_KEY=...` in `~/.zshrc` / `~/.bashrc`, then `source` it

## Decisions

- **CLI vs API**: single run-and-wait file → CLI (`violin`). Multi-job / HTTP / web UI → API server (`violin-api`); print the command, don't auto-start it.
- **Style** (`--style`): default `standard`. Kids content → `kids`, formal/lecture → `academic`, casual → `casual`, dramatic → `storyteller`, news → `news`. Run `violin --style list` if unsure.
- **Voiceover**: keep default (mix dubbed audio over a quiet original). Use `--no-voiceover` only when the user explicitly says "replace audio entirely".

## Run

```bash
violin <input> <output> --language <Lang> [flags]
```

## Flags

| Flag | Default | When to set |
|------|---------|-------------|
| `--language` / `-l` | *required* | Target language (e.g. `Chinese`, `Spanish`, `Japanese`). |
| `--voice` / `-v` | auto (native voice picked by `preferences.voice_gender`) | Only when the user names a specific voice from the catalog (e.g. `"warm female narrator"`). Otherwise omit and let the default kick in. |
| `--source-language` | `auto-detect` | Only if Whisper mis-detects the source language. |
| `--style` / `-s` | `standard` | See Decisions above. |
| `--no-subtitles` | off | User says "no SRT" / "video only". |
| `--no-voiceover` | off | User says "replace original audio entirely". |
| `--config` / `-c` | `config/default.yaml` | Don't use through this skill — repo-only flow. |
| `--timings-out` | off | Only when the user wants a per-step timing JSON for debugging / benchmarking. |

## Language coverage

33 target languages total. **16** ship with handpicked native-speaker voices: Chinese, Spanish, English, Hindi, Arabic, Portuguese, Russian, Japanese, Turkish, German, Korean, French, Italian, Polish, Dutch, Swedish. The other **17** fall back to the English voice catalog (multilingual under Cartesia Sonic 3) — quality is decent but the voice isn't a native speaker. Mention this caveat only if the user is translating to a fallback language and asks about voice quality.

## Report back

- Output video path + SRT path (printed by the run).
- Total cost (printed at end — surface, don't hide).
- If voiceover was on, mention the `_original.m4a` sidecar.

## Don'ts

- Don't run on multi-GB videos without first quoting the rough cost (audio length × per-provider rates in `pipeline/pricing.py`).
- Don't fabricate a "subtitles-only" mode — the CLI requires the full pipeline. If the user only wants SRT, run the full pipeline and hand them just the `.srt`, warning them of the cost first.
- Don't try to switch to OpenAI or ElevenLabs from this skill. Point the user to the repo + `--config config/other_api.yaml` (or their own override).
- Don't paraphrase the README. For supported languages (33), voice catalog, and full flag docs, point them at `README.md` or `violin --help`.
