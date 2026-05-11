"""SQLite-backed stats DB.

Records one row per finished job (done or failed) with metadata, usage and
cost estimates. Local-only — exposed through `/admin` which the Caddy front
explicitly blocks for external traffic, so the DB stays private.

Data captured is non-identifying: target language, model providers, token
counts, dollar cost, timestamps. No video content, no user IPs, no API keys.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import JOBS_DIR

logger = logging.getLogger(__name__)

_DB_PATH = JOBS_DIR / "stats.sqlite"
_lock = threading.Lock()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                       TEXT PRIMARY KEY,
    created_at               INTEGER NOT NULL,
    finished_at              INTEGER NOT NULL,
    status                   TEXT NOT NULL,
    target_language          TEXT,
    source_language          TEXT,
    style                    TEXT,
    voice                    TEXT,
    voiceover                INTEGER,
    transcription_provider   TEXT,
    translation_provider     TEXT,
    tts_provider             TEXT,
    segments_count           INTEGER,
    audio_seconds            REAL,
    tts_characters           INTEGER,
    llm_input_tokens         INTEGER,
    llm_output_tokens        INTEGER,
    whisper_cost_usd         REAL,
    translation_cost_usd     REAL,
    tts_cost_usd             REAL,
    total_cost_usd           REAL,
    duration_seconds         REAL,
    error                    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_target_language ON jobs(target_language);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the table + indexes if they don't exist."""
    with _lock, _connect() as conn:
        conn.executescript(_SCHEMA)


def record_job(row: dict[str, Any]) -> None:
    """Insert one job's metadata. Silent on failure — stats must never break
    the main pipeline."""
    try:
        with _lock, _connect() as conn:
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            conn.execute(
                f"INSERT OR REPLACE INTO jobs ({cols}) VALUES ({placeholders})",
                row,
            )
    except Exception:
        logger.exception("Failed to record job stats")


def summary(days: int = 30) -> dict[str, Any]:
    """Aggregate stats for the last *days* days."""
    cutoff = _now() - days * 86400
    with _lock, _connect() as conn:
        cur = conn.execute(
            """
            SELECT
                COUNT(*)                                       AS jobs,
                SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                COALESCE(SUM(audio_seconds), 0)                AS audio_seconds,
                COALESCE(SUM(tts_characters), 0)               AS tts_characters,
                COALESCE(SUM(llm_input_tokens), 0)             AS llm_input_tokens,
                COALESCE(SUM(llm_output_tokens), 0)            AS llm_output_tokens,
                COALESCE(SUM(whisper_cost_usd), 0)             AS whisper_cost_usd,
                COALESCE(SUM(translation_cost_usd), 0)         AS translation_cost_usd,
                COALESCE(SUM(tts_cost_usd), 0)                 AS tts_cost_usd,
                COALESCE(SUM(total_cost_usd), 0)               AS total_cost_usd,
                COALESCE(SUM(duration_seconds), 0)             AS wall_seconds
            FROM jobs
            WHERE created_at >= ?
            """,
            (cutoff,),
        )
        agg = dict(cur.fetchone())

        # By language
        cur = conn.execute(
            "SELECT target_language, COUNT(*) AS n, COALESCE(SUM(audio_seconds),0) AS audio "
            "FROM jobs WHERE created_at >= ? AND target_language IS NOT NULL "
            "GROUP BY target_language ORDER BY n DESC",
            (cutoff,),
        )
        agg["by_language"] = [dict(r) for r in cur.fetchall()]

        # Recent
        cur = conn.execute(
            "SELECT id, created_at, status, target_language, audio_seconds, "
            "       total_cost_usd, duration_seconds, error "
            "FROM jobs ORDER BY created_at DESC LIMIT 50"
        )
        agg["recent"] = [dict(r) for r in cur.fetchall()]

    return agg


def _now() -> int:
    import time
    return int(time.time())
