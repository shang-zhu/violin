"""FastAPI application factory."""

import asyncio
import logging
import os
import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import catalog, chat, files, jobs

logger = logging.getLogger(__name__)

_STATIC = pathlib.Path(__file__).parent / "static"

_ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="Violin API",
    description=(
        "Translate educational videos into 42 languages using Together AI. "
        "Upload a video, poll for status, then download the dubbed output."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(files.router)
app.include_router(catalog.router)
app.include_router(chat.router)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", include_in_schema=False)
def root():
    # Force the SPA shell to revalidate on every load — otherwise users keep
    # seeing stale UI after upgrading violin-api (esp. on 127.0.0.1 vs localhost,
    # which Chrome caches as separate origins).
    return FileResponse(
        str(_STATIC / "index.html"),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


# Lightweight health probe — accepts both GET and HEAD so external monitors
# (e.g. UptimeRobot's free tier, which only does HEAD) don't get 405s.
@app.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
def health():
    return {"ok": True}


@app.get("/app-config", include_in_schema=False)
def app_config():
    from .config import (
        FREE_TRIAL_JOBS,
        JOB_TTL_HOURS,
        MAX_DURATION_SECONDS,
        MAX_FILE_SIZE_MB,
        URL_UPLOAD,
    )
    return {
        "url_upload": URL_UPLOAD,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        # Used by the footer to hide privacy/auto-delete warnings on local deployments.
        "free_trial_jobs": FREE_TRIAL_JOBS,
        "job_ttl_hours": JOB_TTL_HOURS,
    }


@app.on_event("startup")
async def _init_stats():
    """Create the stats table if it doesn't exist. Idempotent — existing rows
    are preserved across restarts."""
    from . import stats as _stats
    _stats.init_db()


@app.on_event("startup")
async def _start_cleanup_loop():
    from .config import JOB_TTL_HOURS
    if JOB_TTL_HOURS <= 0:
        return

    async def _cleanup_loop():
        from .storage import cleanup_old_jobs
        while True:
            await asyncio.sleep(3600)
            try:
                deleted = cleanup_old_jobs(JOB_TTL_HOURS)
                if deleted:
                    logger.info("Cleaned up %d expired job(s)", deleted)
            except Exception:
                logger.exception("Job cleanup failed")

    asyncio.create_task(_cleanup_loop())
