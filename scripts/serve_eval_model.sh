#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the evaluation model checkpoint}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8899}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP="${TP:-2}"
DP="${DP:-2}"

export CUDA_VISIBLE_DEVICES
export FLASHINFER_WORKSPACE_SIZE_MB="${FLASHINFER_WORKSPACE_SIZE_MB:-4096}"
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tp "$TP" \
  --dp "$DP" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.6}" \
  --context-length "${CONTEXT_LENGTH:-40960}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-1024}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}"
