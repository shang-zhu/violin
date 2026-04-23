"""FastAPI application factory."""

import os
import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import catalog, chat, files, jobs

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
    return FileResponse(str(_STATIC / "index.html"))
