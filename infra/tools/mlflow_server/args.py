from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch MLflow UI for a local path (run dir / experiment dir / mlruns dir / output dir)."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Local path pointing to a run, experiment, mlruns root, or folder containing mlruns.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind MLflow UI."
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind MLflow UI."
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail if the requested port is already in use (default: auto-select next free port).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sqlite", "file"],
        default="auto",
        help="Backend store mode. auto: sqlite if db exists, otherwise file.",
    )
    parser.add_argument(
        "--sqlite-db-name",
        type=str,
        default="mlflow.db",
        help="SQLite DB filename when backend is sqlite (or auto chooses sqlite).",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not open MLflow UI URL automatically in browser.",
    )
    return parser.parse_args()
