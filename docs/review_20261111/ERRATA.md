# Reading corrections for the pre-review supplement

These notes accompany the immutable `Historical_Supplement_20261110.pdf`.
They correct interpretation of that snapshot, not its bytes, and do not replace
new experiment results or the current submission's short supplement.

| Historical section | Scope and current reading |
| --- | --- |
| S1, extraction audit | Measures extraction sensitivity only. The statement that retrieval-level threshold sensitivity still requires new caches and matched training remains a limitation; the matched five-seed study did not sweep those thresholds. |
| S2, paired implementation check | Compares implementations on the historical image list. Future-tense instructions to use the content-disjoint protocol have since been fulfilled by the November retrieval runs; the check itself is not a retrieval experiment. |
| S3, training-pair simulation | Historical simulation of the pair objective. It is not a substitute for trained full-gallery results and does not show test generalization. |
| S4, tensor storage and evidence | Describes historical representation/storage checks. The reference to the main paper's fit/validation grid belongs to that version; the nine-trial grid was used for PARE-F, with its settings reused for the controls. |
| S5, spatial metadata | The 0.15–0.60 m/pixel range applies only to files with supporting projected-metre tags. It is stored-grid spacing, not a verified range of every image's sensor GSD. Historical `PARE` results in this section refer to PARE-F, not the later PARE-S mean. |
| S6, preceding single-seed protocol | The cited release `6e4ed9a` is a historical release, not the latest repository commit. Use this section's hashes for its own runs; do not substitute the five-seed release's source hashes. |
| S7, preceding single-seed retrieval | `PARE` denotes PARE-F; `CFAR-strict` denotes PARE-S. Controlled positive-inserted pools, natural shortlists and full-gallery ranks answer different questions. They are single-seed diagnostics, not new five-seed estimates. |
| S8, exploratory factorial follow-up | Uses a previously used validation set and is exploratory. Its statement that no configuration was promoted describes that experiment's original scope; later selection of PARE-S occurred and must be disclosed. It does not establish that the architecture was fixed before any test results were inspected. |
| S9, conditional-token diagnostic | Compares conditional and constant descriptors with another full-branch recipe. It does not isolate descriptor conditioning in the final PARE-S configuration. Its negative and inconclusive differences remain part of the evidence. |
| S10, matched simplified study | 75 executed trainings include ten F/S seed-0 reproductions of outcomes known before choosing PARE-S. They are not 75 wholly new independent observations. Shared seeds across backbones and reused test data limit interpretation of the intervals. |

## Historical method labels

These definitions follow `code/eval_retrieval_baselines.py`,
`code/primitive_controls.py` and the explicit variant map in
`tools/evaluation/evaluate_multik.py`.

- **Scene-CLIP / Scene:** dot product of the unit-normalized text vector with
  the unit-normalized whole-image embedding; no trained retrieval head.
- **Mean-pool:** unit-normalize each valid scene/coarse/fine embedding, average
  the scene plus all valid coarse and fine embeddings with equal weight per
  primitive, normalize that average, then take its dot product with the unit
  query vector. Padded embeddings are excluded. This is query-agnostic pooling.
- **MaxSim:** maximum query cosine similarity across the whole image and every
  valid coarse and fine embedding. This is a single-vector query maximum, not a
  sum over text-token similarities. Padded primitives are excluded.
- **Stats-off:** retrained full-geometry PARE with `sr,ty` disabled. Reliability
  is set to one and token augmentation is bypassed; the CV/CFAR-derived crops,
  scene/coarse/fine fusion and trainable gate remain. It is different from the
  simplified model's token-only ablation, which retains reliability.
- **Uniform grid:** replace intensity-derived proposals with a 2-row by 3-column
  coarse partition (six tiles) and a 2-row by 4-column fine partition (eight
  tiles), retaining the scene. The control requires both `sr` and `ty` disabled;
  its placeholder statistics are unused. It is compared with Stats-off to
  examine proposal geometry under the same disabled reliability/token settings.
  It does not isolate proposal geometry with token augmentation active.

These historical Mean-pool/MaxSim rows belong to the candidate-pool diagnostics.
The current revision does not supply a separate full-gallery Mean-pool/MaxSim
experiment or claim that their candidate-pool values are full-gallery scores.
The evaluator's ability to produce such scores is not evidence that a reported
run was performed.

## Development work not fully represented in that PDF

The 3 November repair exploration considered gate inputs, several reliability
rules, token-magnitude limits and a 24-to-48-pixel crop expansion with centers
fixed. Its record reports 54 trainings, 108 development/internal-check evaluations
and 42 fixed-model probes. It found no stable isolated fine-branch benefit; the
48-pixel replacement was not adopted. Its internal-check data came from previously
used material, not a new external blind set. Those trials are distinct from the
42-run factorial follow-up and the later 15-head token mechanism study.

Statements such as “all outcomes are retained” apply only to the explicitly
identified study and its archived files. The unchanged PDF is not a complete
record of every development experiment. The current revision index should link
any additional released records separately and preserve unsuccessful outcomes.

## Source descriptions and metadata

The nominal source specifications are heterogeneous: SpaceNet6's airborne SAR
was orthorectified and resampled to 0.5 m pixels; OpenEarthMap-SAR describes Umbra
Spotlight data at 0.15–0.5 m; DFC2023 describes Gaofen-3 SAR at nominal 1 m.
The DFC2023 0.5 m and 0.8 m specifications refer to optical sensors and its 2 m
specification to nDSM. These literature values must not be imputed as measured
spacing for individual files lacking metadata.

The source literature is [SpaceNet6](https://doi.org/10.1109/CVPRW50498.2020.00106),
[OpenEarthMap-SAR](https://doi.org/10.1109/MGRS.2025.3599512), and the
[official DFC2023 data description](https://www.grss-ieee.org/community/technical-committees/2023-ieee-grss-data-fusion-contest/).
