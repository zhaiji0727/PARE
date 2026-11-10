# November 11 review diagnostics

This addendum does not replace or modify the frozen November 9 study. It contains post-hoc checks on the same previously inspected benchmark, not a new blind test.

## What was run

1. **25 frozen-head probes.** For every original PARE-S backbone/seed, replace each regional mixture posterior by that fitted head's mean posterior over all valid retained-training coarse regions. Region embeddings, reliability, gating and learned parameters are unchanged. Report the difference **constant posterior minus original**. The unmodified scorer reproduces every first-caption rank for seed 0 of each backbone before the probe; this compatibility check covers five heads, not all 25.
2. **25 matched constant-descriptor trainings.** Use the same five backbones/seeds, initial parameter bytes, minibatch order, ten epochs, and validation-selected settings as PARE-S. Replace each regional input descriptor by the mean over all valid coarse regions in the 9,181-image retained training cache, with the detector coordinate set to zero. Keep the original trainable descriptor head and prototypes. Report **original conditional PARE-S minus constant descriptor**. This controls region-varying conditioning in this parameterization, not every possible constant-offset model.
3. **75 full-caption evaluations plus five frozen Scene references.** Keep all 6,190 caption rows, including repeated text, and the complete 3,688-image gallery. Reuse the original 3,688 first-caption ranks verbatim and evaluate the remaining 2,502 caption rows with the same fitted models; combine by explicit query IDs with equal weight per caption row. No images or rows are selected according to outcome.
4. **Numerical input checks.** Verify source caption/ID files, caches and pretrained/external checkpoint hashes. Re-encode all 3,688 original and 2,502 extra caption rows. Original vectors differ by at most 5.37e-7 (four CLIP-family backbones); SigLIP is exact. All extra vectors reproduce exactly. Float64 cosine similarity is at least 0.999999999998616. Re-scoring seed 0 of all three models and Scene gives identical R@10 membership for all 20 checks; RemoteCLIP full has one non-top-10 rank change. This is numerical agreement, not universal bitwise equality.
5. **Text overlap and descriptive groups.** Check exact stripped text against every retained training caption. The 412 overlapping and 3,276 nonoverlapping first-caption queries are summarized from existing ranks, without changing the gallery or retraining.

All completed outcomes are retained. There is no outcome-triggered extension, parameter search, or selected seed. The constant-descriptor contrasts use a separate five-comparison family if family-adjusted intervals are displayed; they are not added to or relabeled as the original ten-comparison protocol. Five-seed Student-t intervals remain small-sample parametric summaries on fixed data. Do not interpret an interval spanning zero as equivalence.

## Files

- `probe/<backbone>/seed_*.json`: full ranks, intervention, train-posterior summaries, token-norm bounds and original/probe metrics.
- `constant_training/<backbone>/protocol.json`: configuration and input hashes written before that backbone's first new training.
- `constant_training/<backbone>/seed_*/`: final head, per-epoch losses, initialization/batch-order hashes and full ranks.
- `allcaptions/<backbone>/`: full 6,190 ranks per model/seed and Scene, plus measured preprocessing. The large extra text embedding tensors stay in the server research directory and are omitted from the distribution; they can be regenerated from the original caption files. No caption text is redistributed here.
- `input_verification/`: measured errors, source hashes, software versions, preprocessing and rank comparison counts.
- `caption_overlap.json`: exact-match counts, query-group membership and five-backbone group summaries.
- `new_diagnostics_summary.json`: generated summaries with explicit numerical acceptance gate.
- `failed_attempts/` and `verify_*.log`: retained tokenizer-cache and overstrict bitwise-verification failures. `verify_numerical_*.log` and corresponding JSON supersede the failed bitwise-only check; the failure is not hidden.
- `independent_audit/`: separately written verification, including the portable constant-head loader check when present.

## Reproduce on the original research layout

The archived executed scripts intentionally retain their input paths. They depend on the unchanged `simplified_confirm_20261109_v1` study (including `base_code`, `reference`, `code`, 75 original heads and shared text tensors), `formal_experiments_20261103/results/<backbone>/cache_physics_{train,test}.pt`, retained caption/ID files, and original provider weights. They are not standalone scripts with synthetic replacement data. The study's input manifest specifies each file hash.

With `W=/home/coder/my_workspace/rs_paper/pare_workspace`, the environment is `W/.venv/bin/python`; set `HF_HOME=W/models/huggingface` before any SigLIP tokenizer invocation. For each backbone (`vitb32`, `vitb16`, `remoteclip`, `siglip`, `georsclip`):

```sh
python run_posthoc.py --mode probe --backbone vitb32 --gpu 0
python train_constant.py --backbone vitb32 --gpu 0
python run_posthoc.py --mode allcaptions --backbone vitb32 --gpu 0
python verify_text_numerical.py --backbone vitb32 --gpu 0
python caption_overlap_audit.py
python summarize_new.py
```

Use an empty addendum output directory for a fresh run. Completed per-head outputs are skipped, while partial constant-training directories cause an error and remain for inspection. The original study is read-only throughout. `verify_text_inputs.py` and `measure_text_inputs.py` are retained diagnostic history; the final numerical validation is `verify_text_numerical.py`.

## Loading a constant-descriptor checkpoint

**Do not load these 25 checkpoints as an ordinary SAQREHead.** The weights alone omit the constant-input intervention. Add the frozen `base_code` directory and the directory containing `constant_descriptor_head.py` to `PYTHONPATH`, then use:

```python
from constant_descriptor_head import load_constant_descriptor_head
from train_saqre import move_docs
from batch_scoring import score_matrix

head = load_constant_descriptor_head("constant_training/vitb32/seed_0/head.pt", "cuda")
scores = score_matrix(head, query_vectors, move_docs(test_cache, image_indices, "cuda"))
```

The loader restores `constant_descriptor`, sets `cfar_strict`, and overrides both scalar and batched augmentation. The source `train_constant.py` used its own equivalent batched subclass for reported training; the portable loader is an additional checked adapter, not a claim that a different entry point produced those runs. A second implementation checked all 25 heads on five queries and four synthetic documents, including a document with no coarse region: batched outputs match exactly, and scalar/batched outputs differ by at most 5.22e-8.

## Scope of the new result

The full-caption analysis retains higher mean R@10 for S than F on each backbone. Constant-descriptor comparisons do not establish a stable additional gain from region-varying descriptors on any backbone under the five-seed interval criterion. This narrows the mechanism claim without changing the strong measured contribution of the overall learned-token component. Neither constant-descriptor training nor frozen-head replacement proves that descriptors have no effect, that two models are equivalent, or that the gate alone explains all improvement.
