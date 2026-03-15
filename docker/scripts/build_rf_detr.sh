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

print_usage() {
  cat <<'EOF'
Usage: docker/scripts/build_rf_detr.sh [options]

Fast default behavior:
  - Starts/updates the RF-DETR runner container without rebuilding images.

Options:
  --build          Build RF-DETR flavored image before start.
  --rebuild-base   Build base runner + RF-DETR flavored image before start.
  --no-cache       Build without Docker cache (effective with --build/--rebuild-base).
  --force-recreate Force container recreation on up.
  -h, --help       Show this help message.
EOF
}

BUILD_BASE=0
BUILD_FLAVOR=0
NO_CACHE=0
FORCE_RECREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD_FLAVOR=1
      shift
      ;;
    --rebuild-base)
      BUILD_BASE=1
      BUILD_FLAVOR=1
      shift
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --force-recreate)
      FORCE_RECREATE=1
      shift
      ;;
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

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

cd "${PROJECT_ROOT}"

if [[ $BUILD_BASE -eq 0 && $BUILD_FLAVOR -eq 0 ]]; then
  if ! image_exists "rf-detr-model:local"; then
    BUILD_FLAVOR=1
    if ! image_exists "nn-framework-runner:local"; then
      BUILD_BASE=1
    fi
    echo "RF-DETR image missing; building required image layers first..."
  fi
fi

BUILD_ARGS=(--progress=plain)
if [[ $NO_CACHE -eq 1 ]]; then
  BUILD_ARGS+=(--no-cache)
fi

if [[ $BUILD_BASE -eq 1 ]]; then
  "${COMPOSE[@]}" "${BUILD_ARGS[@]}" \
    -f docker/docker-compose.yml \
    build runner
fi

if [[ $BUILD_FLAVOR -eq 1 ]]; then
  "${COMPOSE[@]}" "${BUILD_ARGS[@]}" \
    -f docker/docker-compose.yml \
    -f docker/compose/docker-compose.rf-detr.yml \
    build runner
fi

UP_ARGS=(-d)
if [[ $BUILD_BASE -eq 0 && $BUILD_FLAVOR -eq 0 ]]; then
  UP_ARGS+=(--no-build)
fi
if [[ $FORCE_RECREATE -eq 1 ]]; then
  UP_ARGS+=(--force-recreate)
fi

"${COMPOSE[@]}" \
  -f docker/docker-compose.yml \
  -f docker/compose/docker-compose.rf-detr.yml \
  up "${UP_ARGS[@]}" runner

if [[ $BUILD_BASE -eq 1 ]]; then
  echo "Rebuilt base runner + RF-DETR image and started container: rf-detr-model:local"
elif [[ $BUILD_FLAVOR -eq 1 ]]; then
  echo "Rebuilt RF-DETR image and started container: rf-detr-model:local"
else
  echo "Started RF-DETR container without rebuild: rf-detr-model:local"
fi
