# Historical technical record and experiment index

[Historical Technical Record](Historical_Technical_Record_20261110.pdf) contains
24 pages of experiments through 10 November 2026. It is not the submitted
supplement. Its title area no longer includes an author list or manuscript
identifier. All scientific text, tables and numbers are retained from the
previous snapshot; [provenance](HISTORICAL_PDF_PROVENANCE.json) identifies the
original commit and both PDF hashes. The original file remains in git history.
The current author list is determined by the journal record, not this archive.

Read the record together with [ERRATA.md](ERRATA.md), including the numeric
[scope addendum](historical_scope_addendum.json). The errata correct omitted
comparisons, interpretation, source/spacing distinctions and outdated statements.

## Separate experiments

- [November 3 numerical repair archive](../../experiments/repair_20261103/README.md):
  all 54 training histories, 108 development/internal-check evaluations and 42
  frozen-model probes. These results are **not contained in the 24-page PDF**.
- [Matched study](../../experiments/simplified_20261109/README.txt): the 75 heads
  and 105 evaluations underlying the five-backbone original-scale/scale study.
  Ten F/S seed-0 results were known before selecting PARE-S; this is reused-data
  evidence, not 75 wholly new observations or a blind test.
- [November 11 descriptor/full-caption addendum](../../experiments/review_20261111/README_new_diagnostics.md):
  the 25 PARE-S constant-descriptor trainings, 25 fixed-head probes and 6,190-caption
  evaluations. These are separate from historical Section S9.
- [Earlier single-seed full-gallery baselines](../../benchmarks/revision_20261103/full_gallery_summary.csv):
  Scene, Mean-pool and MaxSim plus the trained configurations. Mean-pool/MaxSim
  full-gallery results were already stored; earlier prose describing them as
  absent was incorrect.

## Exact reproduction source

Use the repository-root `scripts/reproduce_matched.py` entry. It creates a
separate, detached source worktree at experimental commit
`b174e52fa987085f09b2c75ca173fa0ca042c484`. The original source requirements and
artifact hashes remain enforced. Current documentation corrections are not
silently treated as the source that generated historical caches or heads.

The legacy directory name `independent_audit` denotes separately written
**internal verification** of the authors' artifacts. It does not mean an
external independent laboratory replicated the study. Its frozen file hashes
are retained for provenance; this note clarifies the wording of those records.
