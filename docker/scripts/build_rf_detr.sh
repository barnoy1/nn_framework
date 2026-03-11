#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Error: neither 'docker compose' nor 'docker-compose' is available." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${COMPOSE[@]}" \
  -f docker/docker-compose.yml \
  -f docker/compose/docker-compose.rf-detr.yml \
  build runner model

echo "Built RF-DETR flavor images: nn-framework-runner:local + rf-detr-model:local"
