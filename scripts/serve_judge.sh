#!/usr/bin/env bash
set -euo pipefail

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH to the judge model checkpoint}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-$(basename "$JUDGE_MODEL_PATH")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8181}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP="${TP:-2}"
DP="${DP:-2}"

export CUDA_VISIBLE_DEVICES
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-8192}"

python3 -m sglang.launch_server \
  --model-path "$JUDGE_MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --served-model-name "$JUDGE_MODEL_NAME" \
  --tp "$TP" \
  --dp "$DP" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.6}" \
  --context-length "${CONTEXT_LENGTH:-40960}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-512}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}"
