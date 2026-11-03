# Independent PARE multi-K evaluation

This is an external evaluation program. It imports the explicitly selected
training workspace but does not edit its source, caches, checkpoints, or runner
manifest. Use a new, nonexistent output directory for every run.

## Full experiment

Use the original prepared Python environment and offline model environment
variables. Select the GPU with `CUDA_VISIBLE_DEVICES` and use `--device cuda:0`
inside that visible-device namespace.

```sh
python evaluate_multik.py \
  --code-dir /absolute/path/to/training_experiments_20261103/code \
  --scoring-helper /absolute/path/to/batch_scoring.py \
  --physics-cache /absolute/path/to/cache_physics_test.pt \
  --uniform-grid-cache /absolute/path/to/cache_uniform_grid_test.pt \
  --head-full /absolute/path/to/head_full.pt \
  --head-stats-off /absolute/path/to/head_stats_off.pt \
  --head-cfar-strict /absolute/path/to/head_cfar_strict.pt \
  --head-uniform-grid /absolute/path/to/head_uniform_grid.pt \
  --captions /absolute/path/to/ret_clean_20261007/Caption_test.csv \
  --model ViT-B-32 --pretrained laion2b_s34b_b79k \
  --device cuda:0 --query-chunk 32 --doc-chunk 512 \
  --out /absolute/path/to/new_independent_evaluation
```

Pass the relevant `--model`, `--pretrained`, and, for external weights,
`--remoteclip` for other backbones. Cache validation checks the actual weight
hashes, extraction code, package versions, primitive mode, and cache signature.
Head validation checks matching signatures, implementation, flags, dimension,
training IDs, and the declared final-training size/history. If the checkpoint
records a batching helper, its hash must match the supplied helper.

Formal mode requires 3,688 gallery images and every image's first caption in
CSV encounter order. It rejects truncation and requires the declared final
9,181-image / 31,945-pair / 10-epoch head. It evaluates K=100,200,500,1000,2000,3688.
Only the K=100 report includes controlled random pools and the hard pool, as in
the original runner. Random pools use one seed-0 CPU generator advanced through
five pools per query; each contains the positive plus 99 distinct negatives.
The hard pool contains the positive plus 49 strongest scene negatives. Natural
shortlists contain no inserted positive, and misses count as failures.

`--smoke` explicitly switches to 16 images, K=8/16, one random pool, and a hard
pool of eight. It is labeled as non-paper evidence in the manifest.

## Outputs and provenance

- `eval_{full,stats_off,cfar_strict,uniform_grid}_kK.json`: compatible metrics,
  query records, and per-query ranks from the original evaluator.
- `eval_baselines.json`: exact original Scene, mean-pool, and MaxSim formulas.
- `scores_VARIANT.npy`: complete float32 query-by-gallery model scores.
- `scene_scores_MODE.npy`, `baseline_scores_NAME.npy`, and
  `query_embeddings.npy`: float32 matrices used for evaluation.
- `queries.json`: ordered query/caption and gallery IDs.
- `candidates_MODE.npz`: random and hard candidate gallery rows plus complete
  scene ordering, with gallery/query IDs stored in the same file. Natural K
  takes the first K columns of the scene order. Load with `allow_pickle=False`.
- `independent_manifest.json`: explicit external-evaluator attribution, all
  input/source/helper hashes, cache/head provenance, timing scope, output
  hashes, and a final unchanged-input/source check. A failed or interrupted
  manifest is not a completed evaluation.

Both cache modes use their own scene scores/order. Their image IDs and original
image hashes must agree. This preserves the original evaluator's behavior even
if independent image-encoding batches introduce small scene-score differences.
Ranking intentionally uses the original one-dimensional `torch.argsort`
behavior for local pools, including tied scores. No mixed precision or TF32 is
used. Shared scoring time is not advertised as per-K end-to-end latency.

## Regression tests

`test_evaluate_multik.py` compares candidate sampling and ranks with the original
evaluator, including ties, candidate misses, and zero inclusion. It also checks
blocked vmap scores against a direct query loop and baseline scores against
explicit valid-crop formulas. `compare_real_smoke.py` reruns all eight original
head evaluations plus the original baseline evaluator on unchanged real smoke
artifacts and compares every metric and per-query rank.

```sh
python test_evaluate_multik.py --code-dir /path/to/code \
  --scoring-helper /path/to/batch_scoring.py --device cuda:0 \
  --out /path/to/synthetic_regression.json

python compare_real_smoke.py --code-dir /path/to/code \
  --smoke-artifacts /path/to/vitb32_smoke_cpu_bound_weights \
  --captions /path/to/Caption_test.csv --new-results /path/to/new_smoke_output \
  --out /path/to/new_original_comparison_directory
```

Reusable functions `score_head_matrix(...)` and `score_baselines(...)` return
CPU float32 tensors and deliberately perform no provenance validation. A
derived-cache adapter must explicitly validate its own contract before calling
them; it must not disguise a derived cache as a normal extraction cache.

## Paper numeric export

`export_paper_results.py` is a standard-library, read-only result exporter. It
imports the adjacent `evaluation_controller.py` for rank/metric verification;
it does not import PyTorch or run models. Supply exactly one explicit completed
evaluation directory and one completed sweep JSON for each of the five tags.
It never searches for the best test run or selects among attempt directories.
If a controller used an attempt directory, explicitly use that verified path.
Directory/CLI tags are `vitb32`, `vitb16`, `remoteclip`, `siglip`, and `georsclip`.
The exact corresponding training-config tags are respectively
`vitb32_formal_20261103`, `vitb16_formal_20261103`,
`remoteclip_formal_20261103`, `siglip_formal_20261103`, and
`georsclip_formal_20261103`. Validators use this explicit mapping and reject
other dates, suffixes, or cross-backbone assignments; they do not strip tags.

```sh
pare_base=/absolute/path/to/experiment_workspace
python "$pare_base/evaluation_tools_20261103/export_paper_results.py" \
  --evaluation "vitb32=$pare_base/evaluation_results_20261103/vitb32" \
  --evaluation "vitb16=$pare_base/evaluation_results_20261103/vitb16" \
  --evaluation "remoteclip=$pare_base/evaluation_results_20261103/remoteclip" \
  --evaluation "siglip=$pare_base/evaluation_results_20261103/siglip" \
  --evaluation "georsclip=$pare_base/evaluation_results_20261103/georsclip" \
  --sweep "vitb32=$pare_base/training_experiments_20261103/results/vitb32/hparam_best.json" \
  --sweep "vitb16=$pare_base/training_experiments_20261103/results/vitb16/hparam_best.json" \
  --sweep "remoteclip=$pare_base/training_experiments_20261103/results/remoteclip/hparam_best.json" \
  --sweep "siglip=$pare_base/training_experiments_20261103/results/siglip/hparam_best.json" \
  --sweep "georsclip=$pare_base/training_experiments_20261103/results/georsclip/hparam_best.json" \
  --out "$pare_base/paper_numeric_exports_20261103"
```

Optionally add `--stress vitb32=/absolute/path/to/completed_scale_evaluation`.
Scale input must include all three prespecified methods at factors 1, 1.5, 2.
Its original-scale per-query ranks must exactly match the corresponding formal
evaluation. Absent scale input produces no scale table, rather than placeholders.

The exporter rejects running/partial/failed evaluator manifests, smoke runs,
non-3688 test populations, mismatched hashes, incomplete nine-point sweeps,
non-prespecified parameter selections, and final heads that do not use the
selected settings. The *original training runner's* overall `partial` status is
expected because it did not execute the independent evaluator: only its explicitly
completed, hash-bound sweep and four training steps are accepted. Sweep input
must remain beside `run_manifest.json`, and all referenced trial files must be
available for rank and checkpoint-hash checks. The rounded intermediate sweep
CSV is not used for parameter selection or paper numbers.

All inputs are validated before a fresh output directory is created. Output
contains per-table CSV files, unrounded `paper_results.json`, LaTeX tabular
fragments rounded to two decimals, and an export manifest recording input,
source, and output hashes. Table I has all 25 backbone/method rows; Table II has
four ViT-B/32 controls; appendices include all 120 natural-K rows, 35 hard-pool
rows, 35 controlled-pool rows, all 45 validation trials and the five selected
settings. PARE differences retain their sign. Controlled/hard pools are labeled
as positive-inserted diagnostics, never as end-to-end retrieval. The optional
scale table reports synthetic scale sensitivity, not acquisition-GSD validation.

CPU-only export regression (synthetic fixtures are temporary and not paper data):

```sh
python -m unittest discover -s tools/evaluation \
  -p test_export_paper_results.py -v
```

## Paper table layout

`layout_paper_tables.py` is a separate standard-library formatter. It requires
a completed `export_manifest.json` from the reviewed exporter and verifies all
exported file hashes, JSON/CSV agreement, expected table membership, and repeated
values across tables before creating files. It accepts the export directory,
`paper_results.json`, or a CSV inside that same verified export directory; an
unbound standalone CSV is not sufficient evidence.

```sh
python layout_paper_tables.py \
  --input /absolute/path/to/completed_numeric_export/paper_results.json \
  --out /absolute/path/to/new_latex_tables
```

Main Table I places all five backbones horizontally with four metrics and six
fixed method rows. Its added CFAR-strict row uses the verified
`appendix_natural_k` records at `K=3688` and `protocol=full_gallery`; the original
25-row numeric Table I export is unchanged. The layout manifest records this
row's source explicitly. PARE remains the prespecified full configuration.
Table II has four ViT-B/32 control rows with Controlled100
R@10, natural100 candidate recall and R@10, and full-gallery R@10/MRR@10. Table III
has three method rows with original/1.5x/2x R@10 and signed 2x-minus-original
change; it is omitted if no completed verified ViT-B/32 scale results exist.
No unmeasured values or placeholders are inserted.

Each main table is emitted both as `*_tabular.tex` for insertion into an
existing float and as `*_float.tex` with suggested caption/label. Use one or the
other, not both. Complete supplementary floats retain every K and all four
end-to-end and conditional metrics, all seven methods' hard/controlled scores,
all 45 validation trials, five selected settings, signed differences, and every
completed scale condition's four metrics. `main_tables_all.tex` and
`supplementary_tables_all.tex` concatenate their respective floats.

The TeX requires `booktabs` and `graphicx`. Main Table I uses `table*`/`textwidth`;
other tables use `linewidth`. Each table starts at `footnotesize` and is saved
in a local TeX box: only a table wider than its target width is reduced. Small
tables retain their natural size and are never enlarged. Values are rounded only for display, to two
decimal places. No best-result bolding or significance claims are generated.
An undefined conditional metric (zero candidate inclusions) is shown as a dash.
Check actual font size and the five-page manuscript limit after
inserting real results; no final PDF is generated by this formatter.

```sh
python -m unittest discover -s tools/evaluation \
  -p test_layout_paper_tables.py -v
```

## Public result projections

`public_projection.py` creates explicitly labeled projections after validating
one complete formal backbone run. It calls the original result/selection
validators, checks the corresponding split files and all actual evaluated
inputs/final-head hashes, then removes caption bodies, private free-text fields,
commands and absolute machine paths. Only allowlisted result fields are copied;
an unexpected absolute path in a retained structure causes rejection. No
weights, caches, score matrices, or caption CSVs are copied by the projection tool. Exact evaluated PARE head checkpoints are released separately under `checkpoints/revision_20261103/`.

```sh
python public_projection.py \
  --evaluation /path/to/complete/backbone/evaluation \
  --training-run /path/to/corresponding/training/results/backbone \
  --split-dir /path/to/corresponding/clean_split \
  --out /path/to/new_public_projection
```

The output contains the 25 result JSON projections and `queries.json`, plus a
`projection_manifest.json` recording original-source and new-projection SHA-256
hashes, original manifest hashes, model/cache signatures, four final heads,
all nine validation trial bindings, selected parameters and exact split IDs.
Every projected JSON is marked `not_original_json`. The original machine paths
are replaced by documented logical aliases or basename/hash pairs. Internal
numeric exports must still read the original complete evaluation directories;
these public projections are deliberately not replacement original artifacts.

Optional `--stress /path/to/corresponding/complete_scale_evaluation` validates
the completed scale results and exact original-scale rank equivalence with the
primary experiment. It adds five `stress_*.json` projections: three scale
results, the comparison and query records. The original scale manifest hash,
input/source bindings and synthetic-scale limitation are retained without
machine paths. Supplying incomplete scale results rejects the entire operation.

The tool requires the adjacent reviewed `export_paper_results.py` and
`evaluation_controller.py` bytes. It accepts the original runner's overall
`partial` state only when the independent evaluation is complete and the sweep
and all four head steps are individually complete; it rejects active locks,
changed artifacts, non-3688 tests and incomplete input splits. It never pushes
or uploads files. Keep projected directories separate from original results.

```sh
python -m unittest discover -s tools/evaluation \
  -p test_public_projection.py -v
```
