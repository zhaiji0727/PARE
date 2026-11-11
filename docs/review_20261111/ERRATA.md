# Corrections for the historical technical record

These notes apply to [the technical record](Historical_Technical_Record_20261110.pdf).
Scientific contents retain their historical state; the title-area change is
recorded in [provenance](HISTORICAL_PDF_PROVENANCE.json). This document corrects
interpretation and omitted context, not historical measured numbers.

| Section | Correct scope |
|---|---|
| S1–S2 | Extraction and implementation audits, not retrieval experiments. The old 919-image audit is not the new 918-image validation split. |
| S3 | Simulation of duplicate-image batch behavior, not a trained retrieval result. |
| S4 | Storage arithmetic and evidence inventory; not inference latency. |
| S5 | Stored raster spacing where tags exist, not measured acquisition GSD for every image; PARE denotes the earlier full model. |
| S6–S7 | Earlier single-seed trained models. Positive-inserted pools, natural shortlists and full-gallery results have different denominators. |
| S8 | Post-hoc three-seed validation factorial experiment on previously used data. Its original non-promotion statement does not erase the subsequent choice of PARE-S. |
| S9 | Five-seed A1B0C1 experiment: fine attention on, fine embeddings excluded from gate mean, original detector-fraction descriptor on. It is neither full A1B1C1 nor PARE-S A0B0C0. |
| S10 | Matched 75-head study; ten F/S seed-0 runs reproduce known results. Shared seeds and benchmark reuse limit inference. |

## Method names

`full` is PARE-F; `strict` and `cfar_strict` are PARE-S; `strict_token_off`
is PARE-S without token augmentation; `strict_constant_descriptor` is the
PARE-S constant-descriptor control in the later addendum. In the factorial
notation, A enables the fine attended vector, B includes fine embeddings in
the gate mean, and C includes detector fraction in the coarse descriptor.
Thus A0B0C0 names geometry-path settings only; reliability/token settings must
also be stated.

Scene uses whole-image cosine similarity. Mean-pool normalizes the average of
the normalized scene and valid coarse/fine vectors. MaxSim takes the largest
query cosine across these valid vectors. Stats-off means `--ablate sr,ty`:
those flags **disable reliability and token augmentation**; the full geometry
and trainable gate remain. The uniform-grid control uses six coarse (2×3) and
eight fine (2×4) tiles with the same `sr,ty` ablations. It is not a token-on
PARE-S region replacement.

Full-gallery Mean-pool and MaxSim results already exist in
`benchmarks/revision_20261103/full_gallery_summary.csv`, with original per-query
records in each backbone's `eval_baselines.json.gz`. The earlier erratum that
no full-gallery experiment was supplied was incorrect. These are frozen,
training-free baselines; they are not a trained generic-adapter comparison.

## Missing numeric context

All values below were recomputed from original per-query ranks, and are
available without rounding in [historical_scope_addendum.json](historical_scope_addendum.json).

### S9: all seed differences and ordinary intervals

The historical table already reports the four-comparison adjusted intervals.
The ordinary intervals below do not replace them. Development conditioning
minus constant is negative at every seed and its ordinary interval excludes
zero; the adjusted interval crosses zero. Neither interval establishes
statistical equivalence.

| Split / contrast | Seed 0–4 R@10 differences (pp) | Ordinary 95% t interval |
|---|---|---|
| development / conditional_minus_constant_descriptor | -0.1089, -0.4357, -0.2179, -0.6536, -0.1089 | [-0.5982, -0.0118] |
| development / conditional_minus_token_off | +12.5272, +13.0719, +10.6754, +11.3290, +14.2702 | [+10.6096, +14.1398] |
| internal_check / conditional_minus_constant_descriptor | +1.0889, -0.3025, -0.1210, -0.1210, +0.5445 | [-0.5078, +0.9433] |
| internal_check / conditional_minus_token_off | +9.9214, +10.6473, +8.9534, +11.1918, +12.2807 | [+9.0346, +12.1632] |

### Table S27: omitted PARE-F minus CFAR-strict differences

| Backbone | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|
| vitb32 | -0.3254 | -0.7050 | -0.3796 | -0.4072 |
| vitb16 | -0.1085 | -0.8406 | -1.0846 | -0.4892 |
| remoteclip | -0.2983 | -0.2983 | -1.0304 | -0.4434 |
| siglip | -0.2983 | -0.5423 | -1.2202 | -0.4972 |
| georsclip | -0.4881 | -1.0575 | -2.3048 | -0.7548 |

### Table S4: omitted unavailable-spacing row

The unavailable group contains 2,502 queries: France 0, Japan 0, USA 0;
other literal prefixes are GF2 287, P 467, SN6 966 and SV 782. No GSD is
inferred from these prefixes. The following Table S5 already gives this
group's retrieval metrics.

### S7: GeoRSCLIP Stats-off and Scene

For the historical single-seed full-gallery study the two rows have exactly
matching aggregate R@1/5/10 and MRR@10: 0.32538, 1.35575, 2.90130 and 0.87256%.
Their top-ten memberships agree, but 14 of the 3,688 full ranks differ. Thus
identical aggregate metrics are a real recorded result, not evidence of
identical complete rankings or a proven implementation fault.

## Repair exploration and timing of selection

The [separate repair archive](../../experiments/repair_20261103/README.md) now
supplies the numerical outcomes of all 54 trainings, 108 evaluations and 42
probes. It includes unsuccessful reliability/token-limit/crop changes.
The scene/coarse gate mean examined in that exploration is also used by
PARE-S; saying none of the alternative gate ideas was adopted is inaccurate.
PARE-S does not adopt the locked repair candidate's reliability variant,
token cap or 48-pixel crop expansion. These exploratory data were already
used material, not a blind test, and are not inside the 24-page PDF.

## Source metadata and terminology

SpaceNet6 reports a 0.5 m resampled raster grid; OpenEarthMap-SAR describes
0.15–0.5 m Umbra Spotlight imagery; DFC2023 describes nominal 1 m Gaofen-3 SAR.
The DFC2023 0.5/0.8 m values are optical and 2 m is nDSM. Do not impute these
heterogeneous source values as verified per-image GSD. References:
[SpaceNet6](https://doi.org/10.1109/CVPRW50498.2020.00106),
[OpenEarthMap-SAR](https://doi.org/10.1109/MGRS.2025.3599512),
[official DFC2023 description](https://www.grss-ieee.org/community/technical-committees/2023-ieee-grss-data-fusion-contest/).

All uses of “independent verification” in the historical PDF and legacy
`independent_audit` file titles mean separately written internal checks,
not an external third-party replication. Historical code/JSON retain their
original byte-hash contracts. Use the pinned launcher rather than changing
source hashes to accept a newer, differently documented checkout.
