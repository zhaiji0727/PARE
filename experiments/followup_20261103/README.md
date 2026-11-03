# Factorial follow-up: original execution and portable reproduction

This is the complete **post-hoc exploratory** follow-up to the November study.
Its 918-image validation set had already supported hyperparameter selection.
The hypothesis followed previously observed test results. There is no new blinded
test, winner promotion or replacement of the original full method. All 14
configurations × seeds 0/1/2 and all 36 predefined contrasts are retained.
See [results and limitations](../../benchmarks/followup_20261103/README.md).

## Original executed bytes

`original/` preserves the scientific source, runner, fixed protocol, tests and
execution evidence. `original/FOLLOWUP_SOURCE_FROZEN.json` is the actual 24-file
freeze from before training; its SHA-256 is
`3ebc187c1f9b67256747bc46e5383d43a166fdb11cb5441aea2660a937aa011a`.
The original protocol SHA-256 is
`42789ee8793b7c5bd59bf950dd8deefe8bf1c3caa2e9be2d54143630f63bf288`.
`SOURCE_MANIFEST.json` records original and portable sources separately.

The original runner deliberately pins the exact cache file and original split
metadata bytes. A new cache has new serialization metadata and local paths, so
its file hash can differ even when scientific inputs and extraction are the same.
Run the portable entry point below for a new machine. The original launcher is
an archived record of the three-GPU arrangement used for the experiment.

The CPU and CUDA checks each passed 240 forward/gradient comparisons and 36
small two-epoch optimizer comparisons, all with maximum error zero. They cover
full, strict and legacy-CFAR anchors, all reliability/token settings, empty and
populated regions and two embedding dimensions. Path tests check the retained
second gate and the exact B/C interventions. The real full/seed-0 endpoint also
reproduced the previous fit-only validation trial's parameters and all 918 ranks
exactly. These checks do not imply that every component improves retrieval.

## Portable run on the same fixed scientific protocol

First prepare the recorded environment, data and offline backbone weights using
the repository [README](../../README.md). Commands below run from the repository
root. Use a new output directory and select an available physical GPU.

```bash
source env.sh
export CUDA_VISIBLE_DEVICES=0
export PARE_DEVICE=cuda:0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python experiments/followup_20261103/original/base_code/cache_primitives.py \
  --ids data/ret_clean_20261007/train_ids.json \
  --img_root data/SARimages_pre --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k --primitive_mode physics \
  --out results/followup_reproduction/cache_physics_train.pt

python experiments/followup_20261103/portable/scripts/prepare_reproduction.py \
  --cache results/followup_reproduction/cache_physics_train.pt \
  --captions data/ret_clean_20261007/Caption_train.csv \
  --split-dir data/ret_clean_20261007 --model-root models \
  --out results/followup_reproduction/plan
```

An already generated compatible B32 physics **training** cache can be used in the
second command; do not generate or inspect test retrieval results for this study.
The cache contains 9,181 images, but only the fixed 8,263 fit images contribute to
training loss. Each head uses all 28,759 fit caption pairs. The other 918 images
are the complete validation gallery and one first-caption query per image.

Preparation validates the original extraction signature, package versions,
backbone weights, all fixed IDs and caption hashes, primitive capacities and
fit/validation separation before writing anything. It creates `protocol.json`,
its exact original markdown companion and `preparation_manifest.json`, and prints
three complete commands (one for each seed). Run **all three** printed commands.
For a preflight, append `--preflight-only` and give it a separate `--out` directory;
use the original printed output directory for actual training. Jobs can run
sequentially on one GPU or independently on three idle GPUs. They do not need
the archived launcher.

Each command explicitly binds `--expected-protocol-sha256` to the locally prepared
protocol. Only the cache file SHA and local split-metadata SHA are rebound;
the original time, factor matrix, seeds, learning rate, temperatures, epoch count,
caption/ID/model hashes and every contrast coefficient remain identical. The
adapter rejects other protocol changes. New runs record their actual source and
input hashes. They are new reproductions, not the original archived execution.
The adapter diff is [PORTABILITY.diff](PORTABILITY.diff).

The runner saves every final head, complete validation score matrix, per-query
rank, caption hash, per-epoch loss and shuffle hash. No early stopping, new sweep
or favorable-cell selection is performed. Retain all outputs, including negative
results. The portable summary entry point is documented in
[PORTABLE_NOTES.md](PORTABLE_NOTES.md).

## Checks without training

```bash
FOLLOWUP_BASE_CODE="$PWD/experiments/followup_20261103/original/base_code" \
  python -m unittest discover \
  -s experiments/followup_20261103/original/tests -v
python -m unittest discover \
  -s experiments/followup_20261103/portable/tests -v
python tools/followup/verify_public_followup.py
```

The last command needs only Python's standard library and the files in this
repository. It checks all 42 distributed heads by hash, decompresses the original
rank records, recomputes metrics and all 36 contrasts, and checks JSON/CSV tables.
It does not regenerate embeddings or model scores. The original full audit did
replay all 42 saved score matrices; those matrices remain in the complete archive
with published SHA-256 hashes, outside Git. To obtain new score matrices, run the
portable reproduction.
