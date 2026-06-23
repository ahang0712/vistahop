#!/usr/bin/env bash
set -euo pipefail

SUMMARIZER_MODEL_PATH="${SUMMARIZER_MODEL_PATH:?Set SUMMARIZER_MODEL_PATH to the summarizer model checkpoint}"
SUMMARIZER_MODEL_NAME="${SUMMARIZER_MODEL_NAME:-$(basename "$SUMMARIZER_MODEL_PATH")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8123}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP="${TP:-2}"
DP="${DP:-1}"

export CUDA_VISIBLE_DEVICES
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"

python3 -m sglang.launch_server \
  --model-path "$SUMMARIZER_MODEL_PATH" \
  --served-model-name "$SUMMARIZER_MODEL_NAME" \
  --tp "$TP" \
  --dp "$DP" \
  --dtype bfloat16 \
  --host "$HOST" \
  --port "$PORT" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.6}"
