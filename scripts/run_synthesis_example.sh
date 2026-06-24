#!/usr/bin/env bash
set -euo pipefail

python -m synthesis \
  --entities "Eiffel Tower" "Paris" \
  --chains-per-entity 1 \
  --questions-per-chain 1 \
  --max-depth 3 \
  --max-stage 6 \
  --generate-vqa \
  --enable-stage6-fusion \
  --output-dir "${SYNTHESIS_OUTPUT_DIR:-./outputs/synthesis}"
