#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace"
PYTHON_BIN="${NN_PYTHON_BIN:-python}"

backend="${MODEL_BACKEND:-rtdetrv2}"
default_config="experiments/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yaml"
if [[ "$backend" == "rf_detr" || "$backend" == "rfdetr" ]]; then
  default_config="experiments/rfdetr_small_coco_instance_seg_rle.yaml"
fi

CONFIG_PATH="${NN_EXPERIMENT_CONFIG:-$default_config}"
OUTPUT_DIR="${NN_OUTPUT_DIR:-${REPO_ROOT}/out}"
MODEL_SOURCE_ROOT="${NN_MODEL_SOURCE_ROOT:-}"

has_arg() {
  local flag="$1"
  shift
  for token in "$@"; do
    if [[ "$token" == "$flag" || "$token" == "$flag="* ]]; then
      return 0
    fi
  done
  return 1
}

has_source_root_override() {
  for token in "$@"; do
    if [[ "$token" == *"model.source_root="* ]]; then
      return 0
    fi
  done
  return 1
}

print_usage() {
  cat <<'EOF'
  Usage: /workspace/scripts/train.sh [CLI train args]

Runs inside the container and forwards args to:
  python cli.py train

  Working directory is set to /workspace before execution.

Example (inside docker):
    /workspace/scripts/train.sh \
      --config experiments/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yaml \
    --output-dir /workspace/out

Defaults come from MODEL_BACKEND and NN_* env vars when flags are omitted.
EOF
}

if has_arg "-h" "$@" || has_arg "--help" "$@"; then
  print_usage
  exit 0
fi

cd "$REPO_ROOT"

CMD=("$PYTHON_BIN" cli.py train)

if ! has_arg "--config" "$@"; then
  CMD+=(--config "$CONFIG_PATH")
fi
if ! has_arg "--output-dir" "$@"; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$MODEL_SOURCE_ROOT" ]] && ! has_source_root_override "$@"; then
  CMD+=(--overrides "model.source_root=$MODEL_SOURCE_ROOT")
fi
CMD+=("$@")

exec "${CMD[@]}"
