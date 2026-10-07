#!/usr/bin/env bash
PARE_WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RS_RAG_ROOT="$PARE_WORKSPACE"
export RS_RAG_DATA_ROOT="$PARE_WORKSPACE/data"
export RS_RAG_MODEL_ROOT="$PARE_WORKSPACE/models"
export RS_RAG_PYTHON="$PARE_WORKSPACE/.venv/bin/python"
export HF_HOME="$PARE_WORKSPACE/models/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PATH="$PARE_WORKSPACE/.venv/bin:$PATH"
