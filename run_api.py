"""Start the Violin API server.

Usage:
    uv run run_api.py
    uv run run_api.py --host 0.0.0.0 --port 8080
    uv run run_api.py --config config/prod.yaml
"""

import argparse
import sys

import uvicorn

from pipeline import config as pipeline_config
from pipeline.llm_client import validate_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Violin API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a YAML config file (overrides config/default.yaml)",
    )
    args = parser.parse_args()

    pipeline_config.load(args.config)

    cfg = pipeline_config.get()
    missing = validate_env(cfg)
    if missing:
        keys = ", ".join(missing)
        free_trial = cfg["api"].get("free_trial_jobs", 0)
        if free_trial > 0:
            # Free-trial requests are served with the server's keys — without
            # them the trial path is guaranteed to fail, so refuse to start.
            print(
                f"ERROR: missing required environment variable(s): {keys}\n"
                f"       Set them in .env or export them before starting,\n"
                f"       or set api.free_trial_jobs: 0 in your config to run\n"
                f"       in BYOK-only mode (users must provide their own keys).",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            # BYOK-only deployment — every request must carry its own key.
            print(
                f"WARNING: no server-side keys for {keys}.\n"
                f"         Running in BYOK-only mode — every request must include\n"
                f"         its own API key.",
                file=sys.stderr,
            )

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
