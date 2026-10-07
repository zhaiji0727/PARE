#!/usr/bin/env bash
# Shared paths; source this file from an experiment launcher.
PARE_CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RS_RAG_ROOT=${RS_RAG_ROOT:-$(dirname -- "$PARE_CODE_DIR")}
export RS_RAG_DATA_ROOT=${RS_RAG_DATA_ROOT:-${DATA_ROOT:-$RS_RAG_ROOT/data}}
export RS_RAG_MODEL_ROOT=${RS_RAG_MODEL_ROOT:-${MODEL_ROOT:-$RS_RAG_ROOT/models}}
export RS_RAG_RESULTS_ROOT=${RS_RAG_RESULTS_ROOT:-$RS_RAG_ROOT/results}
export RS_RAG_PYTHON=${RS_RAG_PYTHON:-${PYTHON:-$RS_RAG_ROOT/.venv/bin/python}}
export HF_HOME=${HF_HOME:-$RS_RAG_MODEL_ROOT/huggingface}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export PYTHONUNBUFFERED=1
PY=$RS_RAG_PYTHON
DATA=$RS_RAG_DATA_ROOT
RESULTS=$RS_RAG_RESULTS_ROOT
if [[ ! -x "$PY" ]]; then
  printf 'Python environment not found: %s\n' "$PY" >&2
  return 1
fi
mkdir -p "$RESULTS"
