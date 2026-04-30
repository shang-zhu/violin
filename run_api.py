"""Start the Violin API server.

Usage:
    uv run run_api.py
    uv run run_api.py --host 0.0.0.0 --port 8080
    uv run run_api.py --config config/prod.yaml
"""

import argparse

import uvicorn

from pipeline import config as pipeline_config


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

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
