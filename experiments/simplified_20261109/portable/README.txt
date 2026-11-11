Portable reproduction of the fixed matched study

Recommended from the current repository: use scripts/reproduce_matched.py.
It creates a detached source worktree at b174e52fa987085f09b2c75ca173fa0ca042c484
and runs this unmodified entry there. Current code comments differ from the
historical byte hashes; direct use with --repo pointing to current main is
intentionally rejected by SOURCE_REQUIREMENTS. Do not change or bypass those
checks. Alternatively, check out b174e52 explicitly in a separate clone and use
the historical direct commands below. That commit includes this experiment
folder and all required compatible source; no manual additive copy is needed.

The pinned launcher and a clean b174e52 checkout have passed real dataset,
caption-byte and offline-tokenizer preflight. That preflight is not a claim of
retraining all 75 heads or end-to-end score equivalence for newly trained heads.

The retained data layout is DATA/ret_clean_20261007/{train_ids.json,
test_ids.json,Caption_train.csv,Caption_test.csv} and DATA/SARimages_pre/.
There must be 9181/31945 train images/pairs and 3688/6190 test images/pairs.
Backbone files use the documented MODELS directory layout. RemoteCLIP and
GeoRSCLIP are external weights; they are not redistributed here.

Hugging Face cache consistency matters. Run preparation and execution with the
SAME HF_HOME. The repository env.sh sets this to MODELS/huggingface; the
launcher also defaults to that location and permits explicit --hf-home.
In particular, a locally available SigLIP weight does not replace its separate
cached tokenizer. The launcher tests every tokenizer with HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1 before an expensive stage. Missing offline assets cause
a clear failure, not an automatic download or a changed experiment.

From the repository root, after preparation:

  source env.sh
  export CUDA_VISIBLE_DEVICES=0
  python experiments/simplified_20261109/portable/run.py \
    --repo "$PWD" --data "$PWD/data" --models "$PWD/models" \
    --hf-home "$HF_HOME" --out "$PWD/results/simplified_reproduction" \
    --stage plan

Use the same arguments with these stages, in order:
  preflight
  cache             # 10 original-scale physics caches
  train             # all 75 original-trainer heads, no hyperparameter search
  evaluate          # all 75 original-scale full-gallery evaluations
  scale-cache       # two synthetic scales, ViT-B/32 only
  evaluate-scales   # all 30 fixed-head scale evaluations

An available single GPU is selected explicitly by CUDA_VISIBLE_DEVICES; the
process uses cuda:0 within that selection. --device cpu is also accepted, but
CPU rankings near exact ties need not be bit-identical to recorded CUDA runs.
Each head and output is fresh. Existing outputs or prior attempt records are
never overwritten or silently skipped. Preserve a failed attempt for diagnosis
before choosing a fresh output directory; this launcher performs no automatic
retry, model selection or seed selection.

Training remains the original one-direction contrastive loss and duplicate-
image behavior. The new training_entry.py only sets deterministic/thread/TF32
flags then runs repository code/train_saqre.py unchanged. The evaluator uses
query chunk 32 and document chunk 512 and the original scene-order then 1D
head-argsort tie rule. All 3688 first-caption queries and all 3688 gallery images
are retained, without injecting positives into a reduced shortlist.

Portable outputs have their own schema and provenance. They are not inputs to
the archived-result verifier, which intentionally requires its original freeze
and original checkpoint hashes. Recompute summaries from every portable rank
vector; do not compare only favorable cells. Original logs/checkpoints generated
on a reader's machine may contain that reader's local paths. Per-query JSON from
this portable evaluator excludes caption bodies and local paths.

Validation status: static parsing, exact source-hash checks, command-grid
coverage, output-protection and projection tests. Six original measured seed0
heads (ViT-B/32 and SigLIP, three variants each) also passed complete GPU
score-matrix, query-identity and rank equality, with maximum score difference
zero. No portable retraining was performed or claimed. Portable code is never
relabeled as the measured execution source.

Two checkpoint schemas are supported explicitly:

1. Results newly trained by original code/train_saqre.py retain their original
   config/provenance format and pass its unchanged artifacts.validate_head,
   plus the fixed experiment recipe checks.
2. Actual measured checkpoints use schema pare.simplified-head.v1. They are
   validated through checkpoint_contract.py against the exact frozen study,
   backbone, seed, variant, cache signature, all input hashes, ten-epoch
   settings/history, 500 batches per epoch and nine-pair final batch. They must
   ALSO match every float32 parameter's dtype, shape and raw-byte digest in
   the complete 75-head HEAD_PROJECTION_MANIFEST.json. Strict parameter loading
   follows validation. Unknown schemas are rejected; no legacy config is
   invented for measured checkpoints.

For measured heads, evaluate.py defaults to the manifest at
REPO/checkpoints/simplified_20261109/HEAD_PROJECTION_MANIFEST.json, or accepts an
explicit --measured-head-manifest. Both the original measured checkpoint and
its path-redacted public copy share the same parameter identities. The output
labels which checkpoint schema/training origin was evaluated; running a new
scoring adapter does not imply that the head was newly trained.

Release preparation only: project_heads.py accepts --study, --independent-audit
and a NEW --out directory, and refuses to proceed without all 75 independently
audited heads and 105 evaluations. It replaces only private path/host strings
in metadata. No tensor or numeric metadata is edited. It saves each public
copy, reloads it with weights_only=True, and compares the dtype, shape and
bytes of every parameter directly with the original. The manifest retains
original/public file hashes, per-parameter identities and field-level path
redaction mappings (original path values are hashed, not disclosed).
--verify-only independently rechecks an existing projection against its
original inputs and complete manifest. This is not retraining.

The six-head scoring-only check is recorded in
../verification/portable_gpu_parity/PARITY_VERIFICATION.json. It establishes
exact complete-matrix/rank equality for those six original measured heads only.
All75 public projections separately passed per-parameter byte identity and
reload checks; they were not subjected to six redundant GPU evaluations.
There is no claim of end-to-end portable retraining equivalence, nor a claim
that other backbones/seeds received this portable GPU check.
