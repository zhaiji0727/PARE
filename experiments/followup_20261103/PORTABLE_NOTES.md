# Reproduce the fixed exploratory follow-up

The `original/` directory preserves the actual executed source bytes. Its
protocol binds the original serialized cache and local split metadata. A newly
generated cache contains new timestamps and paths, so its file hash normally
differs even when the extraction contract is unchanged. The `portable/` entry
points allow only those two file bindings to change. They do not alter the
training or evaluation algorithms, the factor definitions, the 14 configurations,
the three seeds, the fit/validation populations, or any contrast coefficient.
`PORTABILITY.diff` records the complete source adaptation; the original and
portable source hashes remain distinct.

This remains post-hoc exploratory analysis on the existing 918-image validation
split, previously used for hyperparameter selection. It is not a new blinded
test and does not replace the original full-method test results or definition.
Keep and report all 42 new results, including negative differences. The original
published results remain tied to the original protocol and source hashes.

## Inputs and environment

Use the recorded environment from the repository root README. Prepare the
licensed SAR images, the clean `Caption_train.csv`, the published fixed
`fit_ids.json`, `val_ids.json`, `train_ids.json`, and a local split `manifest.json`.
Captions, images, and pretrained backbone weights are not distributed here.
Prepare the original ViT-B/32 `laion2b_s34b_b79k` weights offline. Set
`RS_RAG_MODEL_ROOT` to their cache directory; the scripts do not download them.

Run from the repository root. The example uses one available physical GPU;
choose its index for your machine. The image-root path is an example and should
point to your licensed dataset copy. Cache only the fixed training IDs.

```sh
PARE_STUDY="$PWD/experiments/followup_20261103"
PARE_REPRO="$PWD/reproductions/followup_b32"
export RS_RAG_MODEL_ROOT="$PWD/models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 PARE_DEVICE=cuda:0
python "$PARE_STUDY/original/base_code/cache_primitives.py" \
  --ids data/ret_clean_20261007/train_ids.json \
  --img_root "$PWD/data/SARimages_pre" \
  --model ViT-B-32 --pretrained laion2b_s34b_b79k \
  --primitive_mode physics --max_regions 6 --max_sc 8 \
  --out "$PARE_REPRO/cache_physics_train.pt"
python "$PARE_STUDY/portable/scripts/prepare_reproduction.py" \
  --cache "$PARE_REPRO/cache_physics_train.pt" \
  --captions data/ret_clean_20261007/Caption_train.csv \
  --split-dir data/ret_clean_20261007 \
  --model-root "$RS_RAG_MODEL_ROOT" --out "$PARE_REPRO/plan"
```

Preparation verifies the original cache signature, package versions, extractor
sources, backbone weight hash, dimension and region capacities (512/6/8),
physics mode, all 9,181 cache IDs, the exact caption and split-file hashes, and
the 8,263/918 disjoint split with 28,759 fit caption pairs. It generates a new
protocol and a preparation manifest; it starts no training. The new protocol
keeps the original design timestamp and adds separately dated reproduction
metadata. Only `input_bindings.training_cache_sha256` and
`input_bindings.original_split_manifest_sha256` may be rebound. Every other
field must deeply match the original protocol. New input and source hashes are
recorded in the resulting heads and manifests.

## Preflight and all three seeds

Preparation prints three complete run commands, also stored as argument arrays
in `plan/preparation_manifest.json`. Each command includes the required
`--expected-protocol-sha256` and executes all 14 configurations for its seed.
Keep `RS_RAG_MODEL_ROOT` and the offline environment variables set when running
them. Run a command first with `--preflight-only` and a separate output path;
then run each seed command without that flag in its fresh output directory.
The preflight reads and validates inputs without text encoding or training.

The original model parameters and computation are unchanged in the portable
runner. Ten epochs and final-epoch validation are fixed; there is no early
stopping or winner-selection step. Each run retains the actual fit-only head,
all 918 query ranks, a complete 918×918 validation score matrix, per-epoch
losses and shuffle hashes, and execution provenance. `--resume` validates prior
completed artifacts. Retrying an incomplete run additionally requires
`--retry-incomplete`, which preserves the previous attempt.

## Summarize the complete new reproduction

After all three seed commands finish, the portable aggregator checks all 42
run artifacts against their recorded file hashes, the complete fixed run matrix,
initialization/shuffle pairing, query identities, and every metric against its
918 exported ranks. Set `PARE_PROTOCOL_SHA` to the digest printed by preparation.
If using the printed commands unchanged, their results are under `plan/results`:

```sh
python "$PARE_STUDY/portable/scripts/aggregate_reproduction.py" \
  --protocol "$PARE_REPRO/plan/protocol.json" \
  --expected-protocol-sha256 "$PARE_PROTOCOL_SHA" \
  --runs "$PARE_REPRO/plan/results" \
  --split-dir data/ret_clean_20261007 --out "$PARE_REPRO/summary"
```

This exports all 42 run metrics, all 14 configuration summaries, all 108 paired
seed contrasts, all 36 contrast summaries, and the nine predeclared primary
descriptive contrasts. It imports the original unchanged aggregation and table
arithmetic. It **does not** load torch checkpoints, recompute scores, or replay
rankings from score matrices; the output explicitly records that narrower
verification scope. The score-matrix file bytes are checked against the run
manifest. The original experiment's independent matrix-rank replay remains in
its separate historical audit.

The archived `original` verifier and aggregation tools document the exact
published execution and its fixed hash bindings. Portable runs record their new
protocol and source hashes and must not be relabelled as the original execution.
Regenerating score matrices requires the licensed data and pretrained weights.

## Adaptation tests

```sh
python -m unittest discover \
  -s experiments/followup_20261103/portable/tests -v
```

The tests reject changes to scientific fields, seeds, populations, evidence
labels, and contrast coefficients; reject invalid cache contracts; and verify
that the factor adapter bytes and training, validation, and state-hashing
function syntax trees are unchanged. They are not additional scientific runs.
