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

    def print_summary(self) -> None:
        from . import config as _conf

        pricing = _conf.get()["pricing"]
        total_time = sum(s["elapsed"] for s in self._steps)

        whisper_cost = self.audio_minutes * pricing["whisper_per_minute"]
        llm_cost = (
            self.llm_input_tokens / 1_000_000 * pricing["llm_per_m_input_tokens"]
            + self.llm_output_tokens / 1_000_000 * pricing["llm_per_m_output_tokens"]
        )
        tts_cost = self.tts_characters / 1_000_000 * pricing["tts_per_m_characters"]
        total_cost = whisper_cost + llm_cost + tts_cost

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

        print()
        print(f"  {'Transcription':<22} {self.audio_minutes:>7.1f} min"
              f"          ${whisper_cost:>8.4f}")
        print(f"  {'Translation':<22} {self.llm_input_tokens:>7,} in"
              f" / {self.llm_output_tokens:>7,} out"
              f"  ${llm_cost:>8.4f}  ({self.llm_calls} calls)")
        print(f"  {'TTS':<22} {self.tts_characters:>7,} chars"
              f"        ${tts_cost:>8.4f}  ({self.tts_calls} calls)")
        print(f"  {'─' * 56}")
        print(f"  {'TOTAL API COST':<22}"
              f"                     ${total_cost:>8.4f}")
        print("=" * 62)
