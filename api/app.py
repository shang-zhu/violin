"""FastAPI application factory."""

import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import catalog, files, jobs

_STATIC = pathlib.Path(__file__).parent / "static"

app = FastAPI(
    title="video-translate API",
    description=(
        "Translate educational videos into 42 languages using Together AI. "
        "Upload a video, poll for status, then download the dubbed output."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(jobs.router)
app.include_router(files.router)
app.include_router(catalog.router)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(_STATIC / "index.html"))
