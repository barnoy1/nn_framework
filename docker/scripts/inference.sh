#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace"
PYTHON_BIN="${NN_PYTHON_BIN:-python}"

CONFIG_PATH="${NN_EXPERIMENT_CONFIG:-}"
OUTPUT_DIR="${NN_OUTPUT_DIR:-}"
INPUT_DIR="${NN_INPUT_DIR:-}"
DEVICE="${NN_DEVICE:-}"
CHECKPOINT_PATH="${NN_CHECKPOINT:-}"
MODEL_SOURCE_ROOT="${NN_MODEL_SOURCE_ROOT:-}"

resolve_config_path() {
  local raw_path="$1"
  if [[ -z "$raw_path" ]]; then
    echo ""
    return
  fi
  if [[ "$raw_path" == /* ]]; then
    echo "$raw_path"
    return
  fi
  if [[ -f "$REPO_ROOT/$raw_path" ]]; then
    echo "$raw_path"
    return
  fi
  echo "experiments/$raw_path"
}

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
  Usage: /workspace/scripts/inference.sh [CLI inference args]

Runs inside the container and forwards args to:
  python cli.py inference

  Working directory is set to /workspace before execution.

  Default settings:
    /workspace/scripts/inference.sh

Example (inside docker):
    /workspace/scripts/inference.sh \
      --config experiments/rtdetrv2_r18vd_120e_coco_instance_seg_rle_1ch.yaml \
    --checkpoint /workspace/weights/rtdetrv2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
    --input-dir /workspace/input_data \
    --output-dir /workspace/out \
    --device cuda

Defaults come from NN_* env vars exposed inside the container.
EOF
}

if has_arg "-h" "$@" || has_arg "--help" "$@"; then
  print_usage
  exit 0
fi

cd "$REPO_ROOT"
CONFIG_PATH="$(resolve_config_path "$CONFIG_PATH")"

CMD=("$PYTHON_BIN" cli.py inference)

if ! has_arg "--config" "$@"; then
  if [[ -n "$CONFIG_PATH" ]]; then
    CMD+=(--config "$CONFIG_PATH")
  fi
fi
if ! has_arg "--output-dir" "$@"; then
  if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
  fi
fi
if ! has_arg "--input-dir" "$@"; then
  if [[ -n "$INPUT_DIR" ]]; then
    CMD+=(--input-dir "$INPUT_DIR")
  fi
fi
if ! has_arg "--device" "$@"; then
  if [[ -n "$DEVICE" ]]; then
    CMD+=(--device "$DEVICE")
  fi
fi
if ! has_arg "--checkpoint" "$@"; then
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: --checkpoint (or NN_CHECKPOINT) is required for inference." >&2
    exit 1
  fi
  CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi
if [[ -n "$MODEL_SOURCE_ROOT" ]] && ! has_source_root_override "$@"; then
  CMD+=(--overrides "model.source_root=$MODEL_SOURCE_ROOT")
fi
CMD+=("$@")

exec "${CMD[@]}"
