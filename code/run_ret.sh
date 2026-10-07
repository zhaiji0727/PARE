#!/usr/bin/env bash
# Usage: run_ret.sh <tag> <model> <pretrained-or-local-path> [checkpoint]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
if [[ $# -lt 3 ]]; then
  printf 'Usage: %s <tag> <model> <pretrained> [checkpoint]\n' "$0" >&2
  exit 2
fi
TAG=$1
MODEL=$2
PRETRAINED=$3
CHECKPOINT=${4:-}
MODEL_ARGS=(--model "$MODEL" --pretrained "$PRETRAINED")
if [[ -n "$CHECKPOINT" ]]; then
  MODEL_ARGS+=(--remoteclip "$CHECKPOINT")
fi
CACHE_ARGS=(--img_root "$DATA/SARimages_pre" --max_images "${PARE_MAX_IMAGES:-0}"
  --encode_bs "${PARE_ENCODE_BS:-128}")
TRAIN_ARGS=(--epochs "${PARE_EPOCHS:-10}" --bs "${PARE_BS:-64}"
  --lr "${PARE_LR:-2e-3}" --kappa "${PARE_KAPPA:-0.03}"
  --weight_decay "${PARE_WEIGHT_DECAY:-1e-4}" --seed "${PARE_SEED:-0}"
  --text_bs "${PARE_TEXT_BS:-256}" --max_pairs "${PARE_MAX_PAIRS:-0}")
EVAL_ARGS=(--candidate_K "${PARE_CANDIDATE_K:-100}" --pools "${PARE_POOLS:-5}"
  --hard_M "${PARE_HARD_M:-50}" --max_q "${PARE_MAX_Q:-0}"
  --seed "${PARE_SEED:-0}" --text_bs "${PARE_TEXT_BS:-256}")
TRAIN_CACHE="$RESULTS/cache_${TAG}_train.pt"
TEST_CACHE="$RESULTS/cache_${TAG}_test.pt"
FULL_HEAD="$RESULTS/head_${TAG}_pare_soft.pt"
PLAIN_HEAD="$RESULTS/head_${TAG}_plain.pt"
TRAIN_CAPTIONS="${PARE_TRAIN_CAPTIONS:-$DATA/ret_clean_20261007/Caption_train.csv}"
TEST_CAPTIONS="${PARE_TEST_CAPTIONS:-$DATA/ret_clean_20261007/Caption_test.csv}"

printf '[%s] model=%s pretrained=%s results=%s\n' "$TAG" "$MODEL" "$PRETRAINED" "$RESULTS"
"$PY" "$PARE_CODE_DIR/cache_primitives.py" --ids "${PARE_TRAIN_IDS:-$DATA/ret_clean_20261007/train_ids.json}" \
  "${MODEL_ARGS[@]}" "${CACHE_ARGS[@]}" --out "$TRAIN_CACHE"
"$PY" "$PARE_CODE_DIR/cache_primitives.py" --ids "${PARE_TEST_IDS:-$DATA/ret_clean_20261007/test_ids.json}" \
  "${MODEL_ARGS[@]}" "${CACHE_ARGS[@]}" --out "$TEST_CACHE"
"$PY" "$PARE_CODE_DIR/train_saqre.py" --cache "$TRAIN_CACHE" --captions "$TRAIN_CAPTIONS" \
  "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" --out "$FULL_HEAD"
"$PY" "$PARE_CODE_DIR/eval_retrieval_phys.py" --cache "$TEST_CACHE" --captions "$TEST_CAPTIONS" \
  --head "$FULL_HEAD" "${MODEL_ARGS[@]}" "${EVAL_ARGS[@]}" --out "$RESULTS/retrieval_${TAG}_pare_soft.json"
"$PY" "$PARE_CODE_DIR/train_saqre.py" --cache "$TRAIN_CACHE" --captions "$TRAIN_CAPTIONS" \
  "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" --ablate sr,ty,cfar --out "$PLAIN_HEAD"
"$PY" "$PARE_CODE_DIR/eval_retrieval_phys.py" --cache "$TEST_CACHE" --captions "$TEST_CAPTIONS" \
  --head "$PLAIN_HEAD" "${MODEL_ARGS[@]}" "${EVAL_ARGS[@]}" --ablate sr,ty,cfar \
  --out "$RESULTS/retrieval_${TAG}_plain.json"
"$PY" "$PARE_CODE_DIR/eval_retrieval_baselines.py" --cache "$TEST_CACHE" --captions "$TEST_CAPTIONS" \
  "${MODEL_ARGS[@]}" "${EVAL_ARGS[@]}" --out "$RESULTS/retrieval_${TAG}_baselines.json"
printf '[%s] done\n' "$TAG"
