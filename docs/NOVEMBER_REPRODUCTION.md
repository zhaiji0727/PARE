# November revision reproduction

The five-backbone formal run is complete. Its measured results and explicit
source/projection/storage bindings are in `benchmarks/revision_20261103/`.
Install the recorded environment and prepare data and offline weights as
described in the root README.
Do not reuse caches or heads from before the repaired extractor.

## Training and independent evaluation

`code/batch_scoring.py` vectorizes the unchanged `SAQREHead.fuse_batch` operation
across text queries. It does not replace the model formula. The training and
selection implementations record this helper's SHA-256 in each checkpoint.
The reviewed CUDA comparison covered dimensions 512 and 768, empty and populated
regions, and full/stats-off/strict/legacy-CFAR configurations. The largest score
and gradient differences were below 1e-7. The timing check was an implementation
microbenchmark, not a paper retrieval latency result.

For a new reproduction, one workspace can generate its own caches and train.
For example, from the repository root (choose an available physical GPU):

```sh
export CUDA_VISIBLE_DEVICES=0
source env.sh
python scripts/run_repaired_experiment.py --tag vitb32_formal_20261103 --out results/vitb32 --device cuda:0 --stage cache
python scripts/run_repaired_experiment.py --tag vitb32_formal_20261103 --out results/vitb32 --device cuda:0 --stage sweep
python scripts/run_repaired_experiment.py --tag vitb32_formal_20261103 --out results/vitb32 --device cuda:0 --stage train
```

Use the backbone model/pretrained/checkpoint arguments listed in the root README
for other tags, and preserve those arguments for all stages. The nine selection
trials use the clean 8,263/918 fit/validation split and ten epochs per trial. The
four final heads use all 9,181 clean train images and all 31,945 caption pairs,
for ten epochs. The full-grid selection JSON retains unrounded metrics; its CSV
is a display artifact and must not be used to recalculate the selected settings.

The November provenance validators distinguish the backbone alias (`vitb32`,
`vitb16`, `remoteclip`, `siglip`, `georsclip`) from its fixed recorded run label
(`ALIAS_formal_20261103`). Use that exact run label with `--tag` and the short
alias as the result directory name when reproducing this experiment.

The following alternative runs all five backbones sequentially on the selected
GPU. Use a fresh results directory; do not rerun it over the example above.

```bash
run_stages() {
  local backbone_alias="$1"
  shift
  local experiment_stage
  for experiment_stage in cache sweep train; do
    python scripts/run_repaired_experiment.py \
      --tag "${backbone_alias}_formal_20261103" \
      --out "results/${backbone_alias}" --device cuda:0 \
      --stage "$experiment_stage" "$@" || return
  done
}
run_stages vitb32 --model ViT-B-32 --pretrained laion2b_s34b_b79k
run_stages vitb16 --model ViT-B-16 --pretrained laion2b_s34b_b88k
run_stages remoteclip --model ViT-B-32 --pretrained laion2b_s34b_b79k \
  --remoteclip "$PWD/models/external_checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt"
run_stages siglip --model ViT-B-16-SigLIP --pretrained webli
run_stages georsclip --model ViT-B-32 --pretrained openai \
  --remoteclip "$PWD/models/external_checkpoints/georsclip/ckpt/RS5M_ViT-B-32.pt"
```

Run `tools/evaluation/evaluate_multik.py` on the four final heads and the two test
caches, following its [CLI guide](../tools/evaluation/README.md). It constructs
each complete float32 score matrix once, then evaluates K=100, 200, 500, 1000,
2000 and 3688 on the same ranks. It uses every test image's first caption in CSV
encounter order (3,688 queries). It preserves original candidate generation and
tie handling. Natural shortlist misses count as failures. Positive-inserted
controlled and hard pools are reported separately and are not end-to-end recall.

The independent evaluator writes its own `independent_manifest.json`; it does
not mark the original runner's unused evaluation stages complete. Consequently,
a runner manifest can remain `partial` while all four caches, selection and four
training stages are complete and separately bound evaluation is complete. Check
the stage records and independent manifest, not only the runner's top-level word.

`tools/evaluation/export_paper_results.py` requires five explicitly supplied
completed runs and corresponding nine-trial selections. It verifies the original
executed evaluator hash. `layout_paper_tables.py` similarly verifies the reviewed
exporter and all numeric exports. These tools reject modified executors rather
than silently treating a different implementation as the reviewed experiment.
If changing the method, create a new provenance chain and do not claim identity
with the original November experiment.

## Explicit cache import used for the November experiment

The experiment began extracting frozen caches before the batching optimization
was ready. `scripts/import_verified_caches.py` supports this specific transparent
migration: four completed original caches must have compatible extraction code,
model weights, data split and configuration. It checks artifact and log hashes,
requires a passed equivalence report, creates symlinks, and records
`imported_verified_cache` with the original manifest path/SHA, executed commands,
log hashes and verification. It never rewrites cache bytes or claims that new
training source generated old caches. The original manifest must remain frozen.

`scripts/continue_formal_training.py` is optional orchestration for that split
workspace arrangement. It requires a `TRAINING_SOURCE_FROZEN.json` containing
the actual workspace's `file_sha256` map and an equivalence report under
`validation/equivalence_cuda.json`. It is not needed for the single-workspace
commands above. The optional evaluation controller also targets the original
five-GPU arrangement and requires the matching frozen manifest and training
state. For another machine, use the evaluator directly with explicit inputs and
`--device`, rather than assuming the original controller's GPU assignments.

## Synthetic scale sensitivity

See [scale tools](../tools/scale/README.md). The 1.5x and 2x transformations are
synthetic resampling conditions with fixed trained heads. They are not new SAR
acquisitions and cannot establish robustness to acquisition GSD, polarization,
incidence angle, or sensor transfer. Raster header inspection distinguishes
projected metre pixel spacing from verified acquisition GSD.

## Tests and public contents

```sh
python -m pytest tests -q
python -m unittest discover -s tools/evaluation -p 'test_*controller.py' -v
python -m unittest discover -s tools/evaluation -p 'test_export_paper_results.py' -v
python -m unittest discover -s tools/evaluation -p 'test_layout_paper_tables.py' -v
TEST_RELEASE_CODE_ROOT="$PWD/code" python -m unittest discover -s tools/scale -p 'test_*.py' -v
```

The evaluator's tensor/rank regression is a separate CLI, documented in its
README. No dataset captions, images, pretrained weights, feature caches, private
submission correspondence or authentication information are distributed. The
October release remains available in Git history; its original release manifest
is retained under `provenance/20261007/`.
