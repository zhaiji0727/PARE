# Historical repair exploration (November 2026)

This archive supplies the complete numerical outcomes of the earlier repair
exploration: **54 training histories, 108 development/internal-check rank
evaluations, and 42 fixed-head probes**. They are separate from the historical
24-page technical record and from the later five-backbone matched study.

The data were already used during method development. The internal-check split
is not an external blind test, and these outcomes did not replace the main
matched-study table. All favorable, unfavorable and inconclusive outcomes are
included, not only the ultimately preferred configuration.

## Contents and accounting

- `gate_reliability/`: 45 heads, each evaluated on 918 development images and
  1,653 internal-check images; training uses 6,610 images and 23,016 caption pairs.
  The complete selection lock and per-seed records are retained. Its
  `constant_probe/` subdirectory contains 42 inference-only mean-posterior probes.
- `context48/`: nine further matched trainings and 18 evaluations. The only
  crop intervention changes 24-pixel support to 48-pixel support with detected
  centers fixed, under the already locked reliability recipe.
- `MANIFEST.json`: original input-file hashes, projected content hashes and
  compressed-file hashes for every payload. JSON paths have been replaced by
  symbolic workspace prefixes. Numeric values, seeds, query/gallery identities,
  ranks and selection outcomes are unchanged. Large JSON files use gzip.
- `measured_source/` directories: the original experiment code. No numerical
  source changes are introduced by this archival release. The historical source
  uses its recorded cache/model layout; it is not a new one-command trainer.

Run `python experiments/repair_20261103/verify_outcomes.py` from the repository
root to verify every projected payload hash and recompute all 108 evaluation
and 42 probe metric records. This check uses only the Python standard library.

All numerical results can be independently recomputed from the rank vectors.
For ranks `r`, R@K is `100 * mean(r <= K)` and MRR@10 is
`100 * mean(where(r <= 10, 1/r, 0))`. Contrast summaries pair the same seed and
report sample standard deviations. The three-seed engineering decision rule
in the protocol is not a significance test. Query count differs between the
two evaluation splits, so their absolute scores are not directly interchangeable.

The archive excludes raw image/caption data, frozen-backbone weights, embedding
caches, trained head tensors and full score matrices. Their hashes remain in
the records, but their presence must not be inferred from the manifest. The
published rank vectors and training histories cover all 54/108/42 outcomes.
This is a complete *numerical outcome archive*, not every byte of the original
execution workspace or a new end-to-end reproduction claim.

## Interpretation

The first exploration found that excluding fine embeddings from the gate mean
could improve the tested full-geometry configuration. This same scene/coarse
gate summary also appears in PARE-S; it is incorrect to say that every explored
gate change was discarded. The later matched study did not adopt the repair
candidate's other reliability rules, token-magnitude cap or 48-pixel crop change.
The 48-pixel comparison did not establish a stable isolated fine-branch gain.
Mean-posterior probes preserved much of the trained token benefit and do not
establish a separate contribution from per-region conditioning.

See the [historical technical record and reading corrections](../../docs/review_20261111/README.md)
and [matched five-backbone study](../simplified_20261109/README.txt) for the
distinct protocols; do not merge their models, denominators or uncertainty.
