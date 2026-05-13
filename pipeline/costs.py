"""Track wall-clock time and estimated API dollar costs for each pipeline step."""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    _steps: list[dict] = field(default_factory=list)
    _t0: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_calls: int = 0
    tts_characters: int = 0
    tts_calls: int = 0
    audio_minutes: float = 0.0

    def start_timer(self) -> None:
        self._t0 = time.time()

    def record_step(self, name: str) -> float:
        elapsed = time.time() - self._t0
        self._steps.append({"name": name, "elapsed": elapsed})
        self._t0 = time.time()
        return elapsed

    def add_llm_usage(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.llm_input_tokens += input_tokens
            self.llm_output_tokens += output_tokens
            self.llm_calls += 1

    def add_tts_usage(self, characters: int) -> None:
        with self._lock:
            self.tts_characters += characters
            self.tts_calls += 1

    def cost_breakdown(self) -> dict:
        """Return a serializable per-stage cost breakdown for the current run.

        Pricing comes from ``pipeline.pricing`` — informational only; updates
        there are needed when an external provider changes rates.
        """
        from . import config as _conf, pricing as _pricing
        from .llm_client import get_transcription_provider, get_translation_provider
        from .tts import get_tts_provider

        cfg = _conf.get()

        whisper_provider = get_transcription_provider(cfg)
        whisper_per_min = _pricing.whisper_per_minute(whisper_provider)
        whisper_cost = self.audio_minutes * whisper_per_min

        translation_provider = get_translation_provider(cfg)
        llm_rates = _pricing.translation_rates(translation_provider)
        llm_cost = (
            self.llm_input_tokens / 1_000_000 * llm_rates["per_m_input_tokens"]
            + self.llm_output_tokens / 1_000_000 * llm_rates["per_m_output_tokens"]
        )

        tts_provider = get_tts_provider()
        tts_per_m = _pricing.tts_per_m_characters(tts_provider)
        tts_cost = self.tts_characters / 1_000_000 * tts_per_m

        return {
            "total": whisper_cost + llm_cost + tts_cost,
            "whisper": {
                "provider": whisper_provider,
                "audio_minutes": self.audio_minutes,
                "per_minute_usd": whisper_per_min,
                "cost": whisper_cost,
            },
            "translation": {
                "provider": translation_provider,
                "input_tokens": self.llm_input_tokens,
                "output_tokens": self.llm_output_tokens,
                "calls": self.llm_calls,
                "cost": llm_cost,
            },
            "tts": {
                "provider": tts_provider,
                "characters": self.tts_characters,
                "calls": self.tts_calls,
                "cost": tts_cost,
            },
        }

    def print_summary(self) -> None:
        cb = self.cost_breakdown()
        total_time = sum(s["elapsed"] for s in self._steps)

        print("\n" + "=" * 62)
        print("  COST & TIME SUMMARY")
        print("=" * 62)

        for s in self._steps:
            pct = s["elapsed"] / total_time * 100 if total_time > 0 else 0
            mins, secs = divmod(s["elapsed"], 60)
            print(f"  {s['name']:<30} {int(mins)}m{secs:04.1f}s  ({pct:>4.1f}%)")
        print(f"  {'─' * 56}")
        mins, secs = divmod(total_time, 60)
        print(f"  {'Total wall time':<30} {int(mins)}m{secs:04.1f}s")

        w, t, tts = cb["whisper"], cb["translation"], cb["tts"]
        print()
        print(f"  {'Transcription':<22} {w['audio_minutes']:>7.1f} min"
              f"          ${w['cost']:>8.4f}")
        print(f"  {f'Translation ({t['provider']})':<22} {t['input_tokens']:>7,} in"
              f" / {t['output_tokens']:>7,} out"
              f"  ${t['cost']:>8.4f}  ({t['calls']} calls)")
        print(f"  {'TTS':<22} {tts['characters']:>7,} chars"
              f"        ${tts['cost']:>8.4f}  ({tts['calls']} calls)")
        print(f"  {'─' * 56}")
        print(f"  {'TOTAL API COST':<22}"
              f"                     ${cb['total']:>8.4f}")
        print("=" * 62)
        from . import pricing as _pricing
        print(f"  Estimates use rates from pipeline/pricing.py "
              f"(last updated {_pricing.LAST_UPDATED}). Check the provider's")
        print(f"  dashboard for your real bill.")
