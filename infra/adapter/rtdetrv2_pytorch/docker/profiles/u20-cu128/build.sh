#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not available." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Error: docker compose plugin is not available." >&2
  exit 1
fi

print_usage() {
  cat <<'EOF'
Usage: infra/adapter/rtdetrv2_pytorch/docker/profiles/u20-cu128/build.sh [options]

Build and start the Ubuntu 20.04 + CUDA 12.8 compatibility profile:
  docker compose -f docker/profiles/u20-cu128/docker-compose.yml build runner
  docker compose -f docker/profiles/u20-cu128/docker-compose.yml \
    -f infra/adapter/rtdetrv2_pytorch/docker/profiles/u20-cu128/docker-compose.yml up -d --build runner

Ensures compatibility base image exists first:
  nn-framework-runner:u20-cu128

Options:
  -h, --help       Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"

CONTAINER_NAME="nn-framework-rtdetrv2-model-u20-cu128"
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Removing stale container name conflict: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker compose \
  -f docker/profiles/u20-cu128/docker-compose.yml \
  build runner

docker compose \
  -f docker/profiles/u20-cu128/docker-compose.yml \
  -f infra/adapter/rtdetrv2_pytorch/docker/profiles/u20-cu128/docker-compose.yml \
  up -d --build --force-recreate runner

echo "Started RTDETRv2 compatibility container: rtdetrv2-model:u20-cu128"
