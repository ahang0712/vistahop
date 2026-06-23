#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -U pip
python3 -m pip install openai aiohttp pyyaml pillow playwright beautifulsoup4 lxml requests
python3 -m playwright install chromium
