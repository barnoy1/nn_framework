from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from argparse import Namespace
from pathlib import Path

from infra.common.logging import logger
from infra.tracking.service_launchers.shared.network_utils import (
    find_available_port,
    free_port_for_reuse,
    is_port_in_use,
    wait_for_service,
)

from .aggregation import build_aggregate_tracking_dir
from .command_builder import build_mlflow_command, resolve_backend
from .discovery import discover_tracking_dirs


def _build_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    venv_bin = str(Path(sys.executable).resolve().parent)
    existing_path = env.get("PATH", "")
    path_entries = [entry for entry in existing_path.split(os.pathsep) if entry]
    if venv_bin not in path_entries:
        env["PATH"] = os.pathsep.join([venv_bin, *path_entries]) if path_entries else venv_bin
    return env


def run(args: Namespace) -> None:
    source_tracking_dirs = discover_tracking_dirs(args.path)
    root_candidate = Path(args.path).expanduser().resolve()
    tracking_dir = build_aggregate_tracking_dir(root_candidate, source_tracking_dirs, args.sqlite_db_name)
    backend = resolve_backend(tracking_dir, args.backend, args.sqlite_db_name)

    if len(source_tracking_dirs) > 1:
        logger.info(
            "Discovered {} MLflow run stores under root. Aggregated store: {}",
            len(source_tracking_dirs),
            tracking_dir,
        )
        for source_dir in source_tracking_dirs:
            logger.info("Source store: {}", source_dir)

    selected_port = int(args.port)
    if is_port_in_use(args.host, selected_port):
        reused = free_port_for_reuse(args.host, selected_port, logger)
        if not reused:
            if args.strict_port:
                raise RuntimeError(
                    f"Requested port {selected_port} is already in use on {args.host} and could not be reclaimed. "
                    "Use a different --port or omit --strict-port."
                )
            fallback = find_available_port(args.host, selected_port + 1)
            logger.warning("Port {} is busy, using {} instead.", selected_port, fallback)
            selected_port = int(fallback)

    url = f"http://{args.host}:{selected_port}"
    cmd = build_mlflow_command(
        tracking_dir=tracking_dir,
        backend=backend,
        sqlite_db_name=args.sqlite_db_name,
        host=args.host,
        port=selected_port,
    )

    logger.info("Launching MLflow UI for {}", tracking_dir)
    logger.info("Artifact root={} ", (tracking_dir / "mlruns").resolve())
    logger.info("Backend={} host={} port={}", backend, args.host, selected_port)
    logger.info("Command: {}", " ".join(cmd))

    process = subprocess.Popen(cmd, cwd=str(tracking_dir), env=_build_subprocess_env())
    if wait_for_service(host=args.host, port=selected_port, timeout_seconds=15.0):
        logger.info("MLflow UI started (pid={})", process.pid)
        logger.info("Open: {}", url)
        if not args.no_open_browser:
            try:
                webbrowser.open(url, new=0)
                logger.info("Browser opened: {}", url)
            except Exception as error:
                logger.warning("Failed to open browser automatically: {}", error)
    else:
        if process.poll() is not None and process.returncode not in (None, 0):
            raise RuntimeError("MLflow UI process exited before becoming ready.")
        logger.warning("MLflow UI is still starting. Try opening {} in a few seconds.", url)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
