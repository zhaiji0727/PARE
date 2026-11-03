# Complete exploratory factorial follow-up

This release retains all 42 fit-only runs (14 configurations × seeds 0/1/2),
all 36 predefined contrasts and all per-query ranks. The follow-up was prompted
by previously observed test results. Its 918-image validation set had already
been used for hyperparameter selection; these are not new blinded test results.
The original full method and all original five-backbone test results are retained.
No follow-up cell is promoted to replace them.

## Fixed comparison

Each head uses the same frozen ViT-B/32 backbone and physical feature cache,
8,263 fit images / 28,759 caption-image pairs, ten epochs, batch 64, learning rate
0.002, contrastive temperature 0.03 and AdamW weight decay 0.0001. Within each seed,
every configuration receives the same initial parameters and independent replay
of the same pair-order shuffle. The final epoch is evaluated on all 918 first-
caption validation queries against all 918 validation images.

A changes only the fine attended output versus scene fallback while retaining
the second gate. B changes whether fine embeddings enter the gate aggregate.
C changes only the coarse descriptor's detector fraction. R toggles soft versus
constant reliability; T toggles learned descriptor tokens. All eight ABC cells
use R=T=1; full (111) and strict (000) additionally use RT=00,01,10.
The [exact fixed protocol](../../experiments/followup_20261103/original/plans/protocol.json)
contains every contrast coefficient and all scope limitations.

## Complete findings

With reliability held on, the learned token contribution is positive at both
endpoints: R@10 +12.200 ± 1.284 pp at full and +17.393 ± 3.587 pp at strict.
Soft reliability is not independently beneficial in a stable way: +0.036 ± 1.309
pp at full, −2.578 ± 2.903 pp at strict (tokens on). At A=C=1, adding fine embeddings
to the gate aggregate reduces R@10 in each seed (−3.268 ± 1.365 pp). The fine
attended-vector effect at B=C=1 is small and variable (+0.182 ± 0.926 pp).
All values are paired-seed means ± sample SD; these are not confidence intervals
or significance tests. Retrained contrasts include optimization effects and do
not identify physical scattering classes.

- [All 42 run metrics](summary/run_metrics.csv)
- [All 14 configuration means and sample SDs](summary/configuration_summary.csv)
- [All 108 within-seed contrast values](summary/paired_contrasts_by_seed.csv)
- [All 36 contrast summaries](summary/contrast_summary.csv)
- [All nine predefined primary contrasts](summary/primary_contrast_summary.csv)
- [Unrounded structured summary and contrast definitions](summary/summary.json)

CSV and JSON retain numeric precision. TeX files are display-only rounding. No
negative values or unfavorable cells are removed.

## Evidence and public storage

The [publication manifest](PUBLICATION_MANIFEST.json) binds each original head,
rank record and archived full-matrix hash. Per-query JSON and per-epoch history
are stored as deterministic gzip streams under `raw/`; the manifest records both
compressed and original uncompressed SHA-256. The three seed manifests are the
original bytes. Their artifact paths describe the historical execution layout;
use the publication manifest for the paths in this repository. The 42 original
[head files](../../checkpoints/followup_20261103/README.md) are included.

The original [complete audit](verification_complete.json) read all heads and
replayed ranks from all 42 full 918×918 score matrices on CUDA. It also checked
training populations, ten complete epochs, shared within-seed initialization,
shuffle records and unchanged input/source hashes. Full score matrices are
retained in the authors' complete archive, with hashes published here; they are
not distributed in Git. Images and caption bodies are not included.

The [real endpoint comparison](prior_sweep_anchor_comparison.json) confirms that
the full/seed-0 follow-up head is tensor-for-tensor identical to the prior fit-only
validation trial at the same settings; all 918 query ranks also match. Different
whole-checkpoint hashes reflect the additional factor/provenance packaging.

`summary/summary_manifest.public.json` is explicitly a projection of the original
summary manifest: only machine paths and its command are omitted, with the source
manifest SHA preserved. Other original files retain their exact bytes.

From the repository root, verify all distributed head/rank hashes and recompute
all metric and contrast tables without data, backbone downloads or torch:

```sh
python tools/followup/verify_public_followup.py
python -m unittest discover -s tools/followup -p 'test_*.py' -v
```

This lightweight check verifies the published ranking evidence; it does not
regenerate model scores. To regenerate scores, follow the
[portable reproduction guide](../../experiments/followup_20261103/README.md).
