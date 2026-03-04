from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from .log_utils import read_log_tail
from .network_utils import find_available_port, free_port_for_reuse, is_port_in_use, wait_for_service
from .process_state import get_running_process, register_process


def start_tensorboard_service(*, log_dir: Path, host: str, port: int, logger_port) -> str:
    resolved_log_dir = log_dir.resolve()
    requested_port = int(port)
    url = f"http://{host}:{requested_port}"
    service_key = f"tensorboard:{host}:{requested_port}:{resolved_log_dir}"
    existing = get_running_process(service_key)
    if existing is not None:
        logger_port.info("TensorBoard is up: {}", url)
        return url

    selected_port = requested_port
    if is_port_in_use(host, requested_port):
        if not free_port_for_reuse(host, requested_port, logger_port):
            selected_port = find_available_port(host, requested_port)
            if selected_port != requested_port:
                logger_port.warning(
                    "TensorBoard port {} still in use; using {} instead.",
                    requested_port,
                    selected_port,
                )

    url = f"http://{host}:{selected_port}"
    service_key = f"tensorboard:{host}:{selected_port}:{resolved_log_dir}"

    service_log = resolved_log_dir / "tensorboard.log"
    log_handle = service_log.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(resolved_log_dir),
        "--host",
        str(host),
        "--port",
        str(selected_port),
    ]
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        cwd=str(resolved_log_dir),
    )
    register_process(service_key, process)

    if wait_for_service(host=host, port=selected_port, timeout_seconds=15.0):
        logger_port.info("TensorBoard is up: {}", url)
        logger_port.info("TensorBoard logdir: {}", resolved_log_dir)
    elif process.poll() is not None:
        tail = read_log_tail(service_log)
        if tail:
            logger_port.warning("TensorBoard failed to start: {}", tail)
            if "pkg_resources" in tail:
                logger_port.warning("TensorBoard requires setuptools (pkg_resources). Install with: pip install setuptools")
        logger_port.warning("TensorBoard failed to start. Check {}", service_log)
    else:
        logger_port.info("TensorBoard starting: {}", url)
        logger_port.info("TensorBoard service logs: {}", service_log)
    return url
