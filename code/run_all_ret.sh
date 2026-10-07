#!/usr/bin/env bash
# Five-backbone launcher. HF downloads resume and reuse the local cache.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
CHECKPOINT_DIR="$RS_RAG_MODEL_ROOT/external_checkpoints"
mkdir -p "$CHECKPOINT_DIR/remoteclip" "$CHECKPOINT_DIR/georsclip"
"$PY" -c 'from huggingface_hub import hf_hub_download; import sys; hf_hub_download(repo_id="chendelong/RemoteCLIP", filename="RemoteCLIP-ViT-B-32.pt", local_dir=sys.argv[1])' "$CHECKPOINT_DIR/remoteclip"
"$PY" -c 'from huggingface_hub import hf_hub_download; import sys; hf_hub_download(repo_id="Zilun/GeoRSCLIP", filename="ckpt/RS5M_ViT-B-32.pt", local_dir=sys.argv[1])' "$CHECKPOINT_DIR/georsclip"
bash "$PARE_CODE_DIR/run_ret.sh" vitb32 ViT-B-32 laion2b_s34b_b79k
bash "$PARE_CODE_DIR/run_ret.sh" vitb16 ViT-B-16 laion2b_s34b_b88k
bash "$PARE_CODE_DIR/run_ret.sh" remoteclip ViT-B-32 laion2b_s34b_b79k "$CHECKPOINT_DIR/remoteclip/RemoteCLIP-ViT-B-32.pt"
bash "$PARE_CODE_DIR/run_ret.sh" siglip ViT-B-16-SigLIP webli
bash "$PARE_CODE_DIR/run_ret.sh" georsclip ViT-B-32 openai "$CHECKPOINT_DIR/georsclip/ckpt/RS5M_ViT-B-32.pt"
printf 'All five backbones done\n'
