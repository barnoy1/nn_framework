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
Usage: infra/adapter/rf_detr/docker/profiles/u22-cu128/build.sh [options]

Build and start the Ubuntu 22.04 + CUDA 12.8 profile:
  docker compose -f docker/profiles/u22-cu128/docker-compose.yml build runner
  docker compose -f docker/profiles/u22-cu128/docker-compose.yml \
    -f infra/adapter/rf_detr/docker/profiles/u22-cu128/docker-compose.yml up -d --build runner

Ensures base image exists first:
  nn-framework-runner:local

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

CONTAINER_NAME="nn-framework-rf-detr-model"
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Removing stale container name conflict: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker compose \
  -f docker/profiles/u22-cu128/docker-compose.yml \
  build runner

docker compose \
  -f docker/profiles/u22-cu128/docker-compose.yml \
  -f infra/adapter/rf_detr/docker/profiles/u22-cu128/docker-compose.yml \
  up -d --build --force-recreate runner

echo "Started RF-DETR container with compose build check: rf-detr-model:local"
