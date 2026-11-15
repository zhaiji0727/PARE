# Recompute public PARE tables from saved ranks

This standalone companion reconstructs every numerical summary from the 315
published per-query rank files. It needs Python 3.10+ and the standard library
only. It does not load Torch, trained weights, pretrained encoders, raw images,
caption text, or score tensors. No GPU or network access is used.

The current package is prepared and tested on a synthetic 315-evaluation
fixture. No unfinished formal experiment has been exported or analyzed here.
Use this command only after the complete audited public export exists:

```sh
python recompute_tables.py --release /path/to/complete-public-export --manifest-sha256 EXACT_MANIFEST_SHA --out /path/to/new/table-verification
```

When integrated under `software/result_tables/`, the same command is:

```sh
python software/result_tables/recompute_tables.py --release . --manifest-sha256 EXACT_MANIFEST_SHA --out ./reader_tables
```

The explicit manifest SHA establishes the chosen public snapshot. Every input
is then read through that manifest. Runtime paths are relative to the export
or companion directory; old author paths in original audit receipts are used
only as provenance labels. The tool refuses incomplete/mixed manifests,
missing or extra experiment identities, changed rank bytes, wrong query order,
GSD group mismatches, and any numeric disagreement with the published summary.
An existing output directory is never overwritten.

## Reconstructed quantities

- First-caption and all-caption R@1, R@5, R@10, and MRR@10 for each of the 315
  evaluations. The 3,688 first-caption rows are recovered by query identity;
  they are not treated as the first 3,688 rows of the 6,190-row array.
- First-caption raster-spacing subgroup metrics with the full 3,688-image
  gallery retained. The fixed groups contain 760, 238, 188, and 2,502 queries;
  the last group has no reliable stored spacing. Stored raster spacing is not
  automatically native SAR resolution, and these are observational subgroups.
- Primary seeds 1–4 and secondary seeds 0–4, per-backbone mean/sample SD, and
  five-backbone macros formed by averaging backbones within each seed before
  calculating across-seed mean/SD. Frozen Scene remains one deterministic
  observation with no invented seed SD.
- Every recorded signed core/control, 50-versus-20-epoch budget, and
  scale-versus-1 difference, including per-seed values. Negative differences
  are retained. The tool compares the complete measurement inventory, not just
  the 327 quantities used by the current manuscript slot registry.

The four modules under `runtime/` are byte-identical copies of the frozen
independent formal-score audit. `independent_core.metrics`,
`summary_tables.build`, and `bind_results.render` keep their original numeric
logic. The original `audit_scores.py` is included because the unchanged
renderer imports its lightweight SHA/Reader helpers; its score-audit entry is
never invoked. Source copies and hashes are recorded in `SOURCE_PROVENANCE.json`.

`metadata/RESULT_SLOTS.json`, `GSD_GROUPS.json`, and `FIXED_QUERY_PROTOCOL.json`
are included unchanged and contain no full captions. Their hashes are checked
against the original formal audit and/or the companion source lock. The
original formal audit and original summary are read from the public export's
`provenance/` directory. Only after the saved-rank reconstruction matches all
published values does the unchanged renderer use that genuine audit/summary
binding. No stand-in success audit is created by the production code.

## Outputs

- `RECOMPUTED_SUMMARY.json`: the full rebuilt observations and measurements,
  explicitly labeled as saved-rank verification rather than new evaluation.
- `PER_SEED_METRICS.csv` and `MEASUREMENTS.csv`: readable complete tables,
  including subgroup metrics, seed values and signed differences.
- `RESULT_VALUES.tex`: all 327 numeric LaTeX definitions.
- `RESULT_VALUES.binding.json`: exact numeric slot mapping; the six manual
  text slots remain unfilled for author review.
- `VERIFICATION.json` and, only after all checks pass, `complete.json`.

Recomputing recall from ranks does not reconstruct the score matrix or
independently determine tie order. The original formal audit checked the
scores and exact-tie ranking rule; this tool verifies its published rank-file
bytes and recomputes all downstream quantities. It never calls these checks
fresh model experiments or evidence of cross-device bitwise reproduction.

Run the complete small synthetic fixture with:

```sh
PYTHONDONTWRITEBYTECODE=1 python test_public_tables.py
```

`INTEGRATION.json` lists the companion assets for a future release export.
Preparing this directory does not modify the already frozen exporter,
training, scoring, or submission files, and does not publish anything.
