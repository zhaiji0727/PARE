#!/usr/bin/env bash
# Retrain leave-one-component-out heads using existing ViT-B/32 caches.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
for COMPONENT in sr ty cfar; do
  HEAD="$RESULTS/head_vitb32_without_${COMPONENT}.pt"
  "$PY" "$PARE_CODE_DIR/train_saqre.py" \
    --cache "$RESULTS/cache_vitb32_train.pt" --captions "${PARE_TRAIN_CAPTIONS:-$DATA/ret_clean_20261007/Caption_train.csv}" \
    --model ViT-B-32 --pretrained "${PARE_VITB32_PRETRAINED:-laion2b_s34b_b79k}" \
    --epochs "${PARE_EPOCHS:-10}" --bs "${PARE_BS:-64}" --lr "${PARE_LR:-2e-3}" \
    --kappa "${PARE_KAPPA:-0.03}" --weight_decay "${PARE_WEIGHT_DECAY:-1e-4}" --seed "${PARE_SEED:-0}" \
    --text_bs "${PARE_TEXT_BS:-256}" --max_pairs "${PARE_MAX_PAIRS:-0}" \
    --ablate "$COMPONENT" --out "$HEAD"
  "$PY" "$PARE_CODE_DIR/eval_retrieval_phys.py" \
    --cache "$RESULTS/cache_vitb32_test.pt" --captions "${PARE_TEST_CAPTIONS:-$DATA/ret_clean_20261007/Caption_test.csv}" \
    --head "$HEAD" --model ViT-B-32 --pretrained "${PARE_VITB32_PRETRAINED:-laion2b_s34b_b79k}" \
    --candidate_K "${PARE_CANDIDATE_K:-100}" --pools "${PARE_POOLS:-5}" --hard_M "${PARE_HARD_M:-50}" \
    --max_q "${PARE_MAX_Q:-0}" --text_bs "${PARE_TEXT_BS:-256}" --seed "${PARE_SEED:-0}" \
    --ablate "$COMPONENT" --out "$RESULTS/retrieval_vitb32_without_${COMPONENT}.json"
done
printf 'ViT-B/32 ablations done\n'
