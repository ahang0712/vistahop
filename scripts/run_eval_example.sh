#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the served evaluation model name}"
DATASETS_CONFIG="${DATASETS_CONFIG:?Set DATASETS_CONFIG to the dataset config JSON path}"
DATA_ROOT="${DATA_ROOT:-.}"
EXP_NAME="${EXP_NAME:-vistahop_eval}"

export MODEL_BASE_URL="${MODEL_BASE_URL:-http://localhost:8899}"
export SUMMARIZER_BASE_URL="${SUMMARIZER_BASE_URL:-http://localhost:8123}"
export SUMMARIZER_MODEL="${SUMMARIZER_MODEL:-qwen3-32b}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8181/v1}"
export USE_GOOGLE="${USE_GOOGLE:-true}"
export USE_OPENAI_JUDGE="${USE_OPENAI_JUDGE:-false}"
export no_fetch="${no_fetch:-false}"
export no_summary="${no_summary:-false}"

python3 evaluation/run_vistahop_eval.py \
  --model-client openai \
  --judge-client openai \
  --model "$MODEL_NAME" \
  --mode tool \
  --datasets "$DATASETS_CONFIG" \
  --data-root "$DATA_ROOT" \
  --tool-config evaluation/tools.yaml \
  --max-concurrent "${MAX_CONCURRENT:-64}" \
  --max-turns "${MAX_TURNS:-20}" \
  --serper-concurrency "${SERPER_CONCURRENCY:-16}" \
  --exp-name "$EXP_NAME" \
  --query-cache-dir "${QUERY_CACHE_DIR:-./cache/query_cache}" \
  --temperature "${TEMPERATURE:-0.7}"
