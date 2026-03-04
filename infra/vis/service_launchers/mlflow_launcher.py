from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from .log_utils import read_log_tail
from .network_utils import find_available_port, free_port_for_reuse, is_port_in_use, wait_for_service
from .process_state import get_running_process, register_process


def start_mlflow_ui_service(
    *,
    tracking_dir: Path,
    host: str,
    port: int,
    logger_port,
    tracking_backend: str = "sqlite",
    sqlite_db_name: str = "mlflow.db",
) -> str:
    resolved_tracking_dir = tracking_dir.resolve()
    backend = str(tracking_backend).strip().lower()
    requested_port = int(port)
    url = f"http://{host}:{requested_port}"
    service_key = f"mlflow:{host}:{requested_port}:{resolved_tracking_dir}:{backend}:{sqlite_db_name}"
    existing = get_running_process(service_key)
    if existing is not None:
        logger_port.info("MLflow UI is up: {}", url)
        return url

    selected_port = requested_port
    if is_port_in_use(host, requested_port):
        if not free_port_for_reuse(host, requested_port, logger_port):
            selected_port = find_available_port(host, requested_port)
            if selected_port != requested_port:
                logger_port.warning(
                    "MLflow UI port {} still in use; using {} instead.",
                    requested_port,
                    selected_port,
                )

    url = f"http://{host}:{selected_port}"
    service_key = f"mlflow:{host}:{selected_port}:{resolved_tracking_dir}:{backend}:{sqlite_db_name}"

    if backend == "sqlite":
        sqlite_path = (resolved_tracking_dir / str(sqlite_db_name)).resolve()
        backend_store_uri = f"sqlite:///{sqlite_path}"
    else:
        backend_store_uri = resolved_tracking_dir.as_uri()

    service_log = resolved_tracking_dir / "mlflow_ui.log"
    log_handle = service_log.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        resolved_tracking_dir.as_uri(),
        "--host",
        str(host),
        "--port",
        str(selected_port),
    ]
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        cwd=str(resolved_tracking_dir),
    )
    register_process(service_key, process)

    if wait_for_service(host=host, port=selected_port, timeout_seconds=15.0):
        logger_port.info("MLflow UI is up: {}", url)
        logger_port.info("MLflow tracking store: {} ({})", resolved_tracking_dir, backend)
    elif process.poll() is not None:
        tail = read_log_tail(service_log)
        if tail:
            logger_port.warning("MLflow UI failed to start: {}", tail)
        logger_port.warning("MLflow UI failed to start. Check {}", service_log)
    else:
        logger_port.info("MLflow UI starting: {}", url)
        logger_port.info("MLflow UI logs: {}", service_log)
    return url
