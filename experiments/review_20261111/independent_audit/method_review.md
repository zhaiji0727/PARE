# Independent review of the added diagnostics

Read-only review of `run_posthoc.py`, `train_constant.py`, `verify_text_inputs.py`
and the frozen study's scoring, ranking, query-selection and training kernels.
No original result or experiment source was modified, and no model run was repeated.

## Design findings

1. Combining 3,688 preserved first-caption ranks with 2,502 additional-caption
   ranks is legitimate for this inference function: queries are scored independently,
   with no query-batch normalization or cross-query retrieval operation. Query IDs
   encode image ID plus caption position, so repeated caption text is retained without
   dictionary collision. All 6,190 rows get equal weight; this is not an image-balanced
   metric or 6,190 statistically independent images.
2. This combination relies on identical image-gallery order, original caption text,
   tokenization, weights and evaluation transforms. The runner asserts gallery/query
   identity and the companion input verifier hashes inputs and weights. The initial
   strict bitwise check failed for four CLIP-family backbones. The completed numerical
   check used the same batch size (256) and measured maximum first-caption component
   differences of 0 to 5.364418e-7, with double-precision cosine similarity above
   0.9999999999986. SigLIP was bitwise equal; the other four were not. All 2,502 added
   vectors reproduced bitwise for each backbone. For Scene and the three seed-0
   model variants, all 20 R@10-hit comparisons matched; one non-top-ten RemoteCLIP-F
   rank differed. The actual combined evaluation preserves the original ranks.
3. The constant-descriptor control uses the mean over valid training-region feature
   vectors, excludes padding, and zeros the fourth descriptor coordinate. This is
   explicitly region-weighted (not one vote per training image). It does not use test
   descriptors to form the constant. Initial parameter tensors, ten-epoch budget,
   training pair order, optimizer and all architecture shapes remain matched to PARE-S.
4. The replacement occurs in `_augment_b`, which is called by `fuse_batch` through
   `score_matrix` in both `train_one` and full-gallery evaluation. It therefore changes
   the intended path in the actual experiment. The scalar `forward` path is not patched:
   public reproduction must restore the custom class and use the batched path, or supply
   a semantically equivalent scalar override. Loading the state_dict into an ordinary
   SAQREHead is not enough; restore constant_descriptor and the cfar_strict setting.
5. The frozen-head probe replaces each posterior with that same fitted head's mean
   posterior over valid training regions. It tests dependence of those fitted weights
   on spatially varying mixing coefficients; it is not the same intervention as retraining
   on a mean input descriptor and cannot isolate all other spatial/radiometric effects.
6. Sign conventions are consistent in the runners: probe reports constant minus original;
   the retrained comparison reports conditional PARE-S minus constant-descriptor control.
   Metrics use percentage units and MRR truncated after ten. Ranking retains the original
   head tie convention (scene ordering, then one-dimensional head argsort) and the
   original direct two-dimensional Scene ordering.
7. No stable/causal claim follows merely from a small point difference. The planned
   five-backbone paired interval family for the new retraining control should be labeled
   post-hoc and use n=5 (df=4), sample SD and SE=SD/sqrt(5). It must not turn the 25
   backbone/seed pairs into 25 independent seeds or assert equivalence from an interval
   crossing zero.

## First completed-result snapshot

`new_results_audit.json` records file hashes and exact checks for 25 probes,
7 constant-descriptor training results and 67 all-caption evaluations. No metric,
rank count, query identity, preserved first-caption rank, initial-state hash or pair-order
mismatch was found among those completed files. Input-verifier outputs were not yet
present at this snapshot; remaining runs and aggregate intervals require a final pass.

## Completed audit

The final `new_results_audit.json` covers all 25 probes, 25 constant-descriptor
heads, 75 all-caption evaluations and five input-verification records (230 JSON
files checked). It verifies metric arithmetic, delta directions, query identities,
preserved first-caption ranks, checkpoint initialization identities and pair orders.
No inconsistency was found. The initial partial snapshot is retained separately.

`new_summary_audit.json` binds the complete summary and its raw source records.
All five probe contrasts, five training contrasts and twenty all-caption model
summaries reproduce. Sample SD is based on n-1; ordinary 95% and five-comparison
Bonferroni intervals use df=4 critical values 2.7764451052 and 4.6040948713.
Every retrained conditional-minus-constant interval includes zero. For the fixed
ViT-B/16 posterior probe, the ordinary interval excludes zero but the adjusted
interval includes zero; this distinction is retained in the response letter.
These post-hoc results do not prove equality or absence of a conditioning effect.

`constant_loader_check.json` tests all 25 resulting heads on small CPU tensors,
including an image with zero valid coarse regions. The portable custom class
matches the original training intervention's batched score bitwise for every
head. Its scalar scores differ from batched scores by at most 5.22e-8. This is a
portability check, not a new retrieval experiment. No frozen measured-source file
or original experiment result was modified by these audits.
