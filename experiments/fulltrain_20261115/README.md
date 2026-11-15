# PARE: full-training experiment records and portable reproduction

This package preserves one complete, independently audited experiment: 145
training trajectories, checkpoints at epochs 20 and 50 (290 heads), and 315
retrieval evaluations. `MANIFEST.json` and `SHA256SUMS` identify the exact files.
This is an export of recorded measurements; exporting does not retrain or
reevaluate a model. No manuscript or editorial correspondence is included.

## Inspect the evidence

- `results/SUMMARY.json`: every audited measurement, including primary seeds
  1–4, secondary seeds 0–4, paired differences of either sign, and macro results.
  Frozen Scene is one deterministic baseline, not five synthetic seed samples.
- `results/ranks/`: one independent per-query rank file for each of the 315
  evaluations; exact tie sizes and first-caption ranks are retained.
- `results/QUERY_IDENTITIES.json`: the 3,688 gallery IDs, 6,190 query identities,
  caption hashes and the 3,688 first-caption row indices. Full captions are not
  distributed. First captions are not assumed to be an array prefix.
- `results/training/RUNS.json`: all 145 complete 50-epoch loss histories and
  selected original run metadata. This is not a convergence assertion.
- `weights/INDEX.json`: all 290 original-run checkpoint identities and hashes,
  including every control and both saved budgets.
- `provenance/`: original audit receipts, source bindings and per-tensor export
  verification. Historical author paths are evidence only; execution does not
  depend on their existence.

Metrics are percentages. The main cohort is five backbones × five models × five
seeds, evaluated at epochs 20 and 50. The B/32 diagnostic cohort contains four
additional controls × five seeds × both epochs. The remaining evaluations are
five frozen baselines and B/32 adapter/PARE path-100 at epoch 50 for scale
factors 1.5 and 2. These are fixed experiment identities, not a selected subset.
The public summary preserves all signed comparisons without filtering outcomes.

Three evidence scopes must remain separate:

1. **Formal measurements:** the audits and numerical records above.
2. **Portable implementation checks:** two complete training replays for B/32
   and SigLIP, path-100, seed 1, compared at epochs 20 and 50. Their exact-match
   claim is limited to these two cases and the recorded environment.
3. **Fresh reader reproduction:** the stages below write new provenance and new
   measurements. Small raw-input smoke tests are separately labeled in
   `provenance/validation`; they do not establish exact reconstruction of every
   input, model, device, or complete pipeline. Earlier development experiments
   are not substituted into the formal results.

## Install and supply assets

Use Python 3.10+ with PyTorch and NumPy. Raw encoding also needs Pillow, SciPy,
OpenCLIP and its tokenizer dependencies; see
`software/raw_inputs/requirements-runtime.reference.txt` for the reference
environment. The original experiment used the versions recorded in the audits.
Matching package versions alone does not guarantee bitwise results on a
different device.

Provide existing public SARLANG assets and pretrained encoders locally:

```text
data/SARimages_pre/<original image basenames, optionally nested>
data/ret_clean_20261007/train_ids.json
data/ret_clean_20261007/test_ids.json
data/ret_clean_20261007/Caption_train.csv
data/ret_clean_20261007/Caption_test.csv
models/<pretrained files and huggingface tokenizer cache>
reports/models_manifest.json
```

Input preparation checks exact split, caption, pretrained-weight and encoder
identities. The training population is 9,181 images, 31,945 caption–image pairs,
and 18,461 unique training captions. The test gallery has 3,688 images and
6,190 total captions. Raw data, caption text and third-party weights are not
redistributed here. Existing assets must not be downloaded or unpacked again.
Byte-identical historical download/split helpers are retained under
`software/asset_preparation/scripts/` for readers who still need those assets;
they are optional, explicit commands and are never dispatched automatically.
The original public split lists are included, so no author-only directory is
needed. For a fresh asset installation only, run:

```sh
python software/asset_preparation/scripts/prepare_data.py --workspace ./asset_work --split-source ./software/asset_preparation/data/splits_original
python software/asset_preparation/scripts/prepare_content_disjoint_split.py --workspace ./asset_work
python software/asset_preparation/scripts/prepare_models.py
```

The first command downloads the original SARLANG-1M files from pinned revision
`0271408f71bcd9afff69e4e7d324cae912461ebd`, verifies archive hashes and extracts
the required images/captions. The second preserves the official test CSV,
removes exact RGB overlap/duplicates from training and writes
`asset_work/data/ret_clean_20261007/`. The third writes
`software/asset_preparation/models/` and its sibling `reports/models_manifest.json`.
Use `--data ./asset_work/data --models ./software/asset_preparation/models`
for the input stage in this installation. Existing users can point to their
already prepared data/model roots instead; these commands are not prerequisites
when the assets already exist.

The historical clean-split receipt is preserved under
`provenance/software/content_disjoint_data.manifest.json`. Its generated
`Caption_train.csv` SHA is
`1c8c524eabb0e7f5e246cc0bb0f0ec774692c233428674acb57af28d1e10589d`,
and `Caption_test.csv` SHA is
`7a1878610e0a5089380b2fb7fd825d43edf8682cba513a76ca36661d3094724d`.
Both match the new raw-input entry's required files. That entry also verifies
the ordered 9,181/3,688 image-ID digests, not merely the row counts.

## New canonical entry

Run from this package root. The default prints a command only; it does not
construct models, write a cache, train, score, download, or unpack anything.

```sh
python scripts/reproduce_fulltrain.py inputs -- --data ./data --models ./models --out ./reader_runs/train_b32 --backbone vitb32 --population train --device cuda
```

To execute exactly one selected stage, add `--execute` before its stage name.
CUDA execution also requires one explicitly selected GPU, for example
`CUDA_VISIBLE_DEVICES=0`. Use new output directories; stages do not resume or
silently overwrite. `--device cpu` is available. The backbone choices are
`vitb32`, `vitb16`, `remoteclip`, `siglip`, and `georsclip`.

1. Build training inputs with the command above and `--execute`; repeat with
   `--population test` and a distinct output directory. The initial two-image
   smoke option `--max-images 2` produces `SMOKE_INPUTS.json` and cannot be used
   as the complete training set. Full input construction omits this option.
2. Train a fresh head using the fixed recipe. Replace `INPUT_SHA` with the
   actual SHA-256 of the new input manifest. The dispatcher binds the bundled
   training protocol automatically:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_fulltrain.py --execute train -- --inputs ./reader_runs/train_b32/INPUTS.json --inputs-sha256 INPUT_SHA --backbone vitb32 --variant path_100__full --seed 1 --device cuda --output ./reader_runs/head_b32_s1
```

3. Evaluate either a new training checkpoint or one of the published weights.
   The following published-weight example takes filenames and hashes from
   `weights/INDEX.json`; replace the uppercase hash placeholders with those
   recorded values, and `TEST_SHA` with the new test-input manifest hash:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_fulltrain.py --execute evaluate -- --checkpoint ./weights/main__vitb32__path_100__full__s1__e50.pt --checkpoint-sha256 WEIGHT_SHA --checkpoint-binding ./weights/main__vitb32__path_100__full__s1__e50.binding.json --checkpoint-binding-sha256 BINDING_SHA --inputs ./reader_runs/test_b32/INPUTS.json --inputs-sha256 TEST_SHA --model path:100:full --backbone vitb32 --device cuda --out ./reader_runs/eval_b32_s1
```

For a newly trained checkpoint replace the sidecar arguments with
`--training-inputs ./reader_runs/train_b32/INPUTS.json --training-inputs-sha256 INPUT_SHA`.
The reader evaluator checks the full encoder signature and fixed ordered
query/content hashes before evaluation. Its results are labeled fresh reader
reproduction, with `eligible_for_formal_paper_tables=false`; they do not rewrite
the historical formal table. This minimal reader entry does not yet rebuild
the separate scale-transformed input caches. The earlier public repository
contains `tools/scale/stress_cache.py` and `scale_contract.py` for its historical
scale protocol; those files are not a drop-in reconstruction of the present
v9 input signature. The 20 formal scale evaluations and their ranks remain in
this release, but a portable raw-to-scale adapter for this new reader is not
included. Consequently this package does not claim an end-to-end rebuild of
all 315 evaluations through one command.

Model names map explicitly: training `adapter` → evaluation `joint:adapter`,
`original_full` → `joint:original_full`, `joint_full` → `joint:joint_full`, and
`path_000__full`/`path_100__full` → `path:000:full`/`path:100:full`.
The B/32 controls use `path_100__constant`, `path_100__no_token`,
`path_100__uniform`, or `path_100__stats_off`; evaluation uses the same last
component in `path:100:CONTROL`. The bundled protocol fixes all hyperparameters,
epochs and valid combinations. No outcome-based tuning is performed by the CLI.

## What published weights contain

Each `.pt` file contains the original trained head `state_dict` and selected
original metadata. Every tensor's dtype, shape and bytes were verified against
the source checkpoint and the reloaded exported file. The public file SHA is
normally different because optimizer/RNG/history objects are omitted; it is
not described as the original file. The adjacent sidecar records both file
SHAs, the exact state digest, original identity, complete training encoder
signature, and a relative link to the public training-audit projection. No
fake input-manifest field is inserted into historical metadata.

Keep code, JSON summaries, ranks and SHA indexes in version control. Distribute
the 290 binary weights as a separate release asset. `scripts/package_assets.py`
plans two archives by default and only creates them with explicit `--execute`:

```sh
python scripts/package_assets.py --release . --output ../new_release_assets
```

Extract the code/results archive and weights archive into the same new
directory, then run `shasum -a 256 -c SHA256SUMS`. Packaging performs no upload,
git commit, or release creation. Raw score matrices are deliberately omitted;
their source hashes and the independently derived per-query ranks are retained.
