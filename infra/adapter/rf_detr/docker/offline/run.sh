#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/infra/adapter/rf_detr/docker/.env"
RUNTIME_DIR="${PROJECT_ROOT}/runtime/rf_detr"
XAUTH_FILE="${XAUTHORITY:-${HOME}/.Xauthority}"
IMAGE_TAG="${IMAGE_TAG:-local}"
IMAGE_NAME="${IMAGE_NAME:-rf-detr-model:${IMAGE_TAG}}"

mkdir -p "${RUNTIME_DIR}"

if [[ ! -f "${XAUTH_FILE}" ]]; then
  XAUTH_FILE="/dev/null"
fi

docker run --rm -it \
  --gpus all \
  --ipc host \
  --shm-size "${SHM_SIZE:-32g}" \
  --env-file "${ENV_FILE}" \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${XAUTH_FILE}:/tmp/.Xauthority:ro" \
  -v /tmp:/tmp \
  -v "${HOME}:${HOME}:rw" \
  -v "${RUNTIME_DIR}:/workspace/runtime:rw" \
  --name nn-framework-rf-detr-offline \
  "${IMAGE_NAME}" \
  /bin/bash