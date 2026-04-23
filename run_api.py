"""Start the Violin API server.

Usage:
    uv run run_api.py
    uv run run_api.py --host 0.0.0.0 --port 8080 --workers 4
"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Violin API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    args = parser.parse_args()

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
