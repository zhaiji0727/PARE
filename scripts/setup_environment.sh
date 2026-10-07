#!/usr/bin/env bash
set -euo pipefail
PARE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PARE_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python python3.10 .venv
fi
if [[ -s requirements.lock.txt ]]; then
  uv pip sync --python .venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match requirements.lock.txt
  uv pip check --python .venv/bin/python
  exit 0
fi
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision==0.24.1
uv pip install --python .venv/bin/python -r requirements-runtime.txt
uv pip check --python .venv/bin/python
