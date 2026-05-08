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

        Pricing is read from the active pipeline config. Falls back gracefully
        when the active provider has no pricing entry (cost reported as 0).
        """
        from . import config as _conf
        from .llm_client import get_transcription_provider, get_translation_provider
        from .tts import get_tts_provider

        cfg = _conf.get()
        pricing = cfg["pricing"]

        # ── whisper ──────────────────────────────────────
        whisper_provider = get_transcription_provider(cfg)
        whisper_pricing = pricing.get("whisper")
        if isinstance(whisper_pricing, dict):
            whisper_per_min = whisper_pricing.get(whisper_provider,
                                                  whisper_pricing.get("together", 0.0))
        else:
            whisper_per_min = pricing.get("whisper_per_minute", 0.0)
        whisper_cost = self.audio_minutes * whisper_per_min

        # ── translation ──────────────────────────────────
        translation_provider = get_translation_provider(cfg)
        llm_pricing = pricing["translation"].get(
            translation_provider, pricing["translation"].get("together", {})
        )
        llm_cost = (
            self.llm_input_tokens / 1_000_000 * llm_pricing.get("per_m_input_tokens", 0.0)
            + self.llm_output_tokens / 1_000_000 * llm_pricing.get("per_m_output_tokens", 0.0)
        )

        # ── tts ──────────────────────────────────────────
        tts_provider = get_tts_provider()
        tts_pricing = pricing.get("tts", {})
        if isinstance(tts_pricing, (int, float)):
            tts_per_m = float(tts_pricing)
        elif "per_m_characters" in tts_pricing:
            tts_per_m = tts_pricing["per_m_characters"]
        else:
            tts_per_m = tts_pricing.get(tts_provider, {}).get("per_m_characters", 0.0)
        if not tts_per_m:
            tts_per_m = pricing.get("tts_per_m_characters", 0.0)
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
