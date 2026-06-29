# VistaHop Evaluation

This directory contains the evaluation runner and helper scripts used for the VistaHop benchmark.

> A subset of the benchmark has been uploaded.

## Contents

- `evaluation/run_vistahop_eval.py`: main evaluation entry point.
- `evaluation/tools.yaml`: tool schema exposed to tool-use models.
- `evaluation/query_cache.py`: SQLite cache for raw search results.
- `evaluation/mt_text_search.py`: optional internal search adapter, configured only through environment variables.
- `evaluation/url_fetcher.py`: standalone URL fetch fallback utility.
- `synthesis/`: VistaHop data synthesis pipeline, stage1 through stage6.
- `scripts/`: launch templates for the evaluated model, judge model, summarizer model, and an example evaluation run.

## Setup

```bash
bash scripts/setup_eval_env.sh
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`, then load it before running evaluation:

```bash
set -a
source .env
set +a
```

## Required Services

The evaluator expects OpenAI-compatible endpoints:

- `MODEL_BASE_URL`: model under evaluation, for example `http://localhost:8899`.
- `SUMMARIZER_BASE_URL`: summarizer endpoint, for example `http://localhost:8123`.
- `JUDGE_BASE_URL`: judge endpoint, for example `http://localhost:8181/v1`.

For web search, set `SERPER_API_KEY` or `SERPER_API_KEYS`. `SERPER_API_KEYS` accepts a comma-separated list and rotates keys when quota errors are returned.

## Run

Start model services with the templates in `scripts/`, for example:

```bash
MODEL_PATH=/path/to/model SERVED_MODEL_NAME=my-model bash scripts/serve_eval_model.sh
JUDGE_MODEL_PATH=/path/to/judge JUDGE_MODEL_NAME=Qwen3-VL-32B-Instruct bash scripts/serve_judge.sh
SUMMARIZER_MODEL_PATH=/path/to/summarizer SUMMARIZER_MODEL_NAME=qwen3-32b bash scripts/serve_summarizer.sh
```

Then run evaluation:

```bash
MODEL_NAME=my-model \
DATASETS_CONFIG=/path/to/datasets.json \
DATA_ROOT=/path/to/data/root \
bash scripts/run_eval_example.sh
```

Outputs are written to timestamped `eval_*` directories unless `--output-dir` is provided.

## Data Synthesis

The synthesis pipeline is available as a top-level Python package:

- Stage 1: extract named visual entities from an image, or accept seed entities directly.
- Stage 2: build node descriptions and attributes.
- Stage 3: construct evidence chains.
- Stage 4: generate multi-hop questions.
- Stage 5: convert questions into VQA items.
- Stage 6: fuse multiple VQA chains into a composite question.

Configure an OpenAI-compatible model in `.env`:

```bash
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1
SERPAPI_KEY=...
```

Run a small example:

```bash
set -a
source .env
set +a
bash scripts/run_synthesis_example.sh
```

Or invoke the pipeline directly:

```bash
python -m synthesis \
  --image-url /path/to/image.jpg \
  --max-stage 6 \
  --generate-vqa \
  --enable-stage6-fusion \
  --output-dir ./outputs/synthesis
```

For multi-key local runs, create an untracked `model_rpm.yaml` from `model_rpm.example.yaml`.

## Notes

- Do not commit real API keys, local `.env` files, generated caches, or evaluation outputs.
- Do not commit `model_rpm.yaml`; use `model_rpm.example.yaml` as the template.
- Set `USE_GOOGLE=false` only if `MT_SEARCH_BASE_URL` and `MT_SEARCH_AUTHORIZATION` are configured for the optional internal search adapter.
