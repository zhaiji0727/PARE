# Measured heads from the 9 November matched study

This directory contains all 75 heads: five backbones (`vitb32`, `vitb16`,
`remoteclip`, `siglip`, `georsclip`), seeds 0–4, and the `full`, `strict`,
`strict_token_off` variants. Files use `<backbone>/seed_<seed>/<variant>/head.pt`.
`strict` is PARE-S; `full` is prior PARE-F.

Only personal machine paths and hostnames in checkpoint metadata were replaced.
Every tensor retains its original dtype, shape and exact bytes, verified after
saving and reloading the public copy. `HEAD_PROJECTION_MANIFEST.json` binds every
public file to its independently audited original hash, parameter identities,
and field-level path-redaction mapping. Original path values are represented by
hashes, not published text. There was no retraining during this projection.

The schema is `pare.simplified-head.v1`, distinct from the original trainer's
checkpoint schema. The [portable evaluator](../../experiments/simplified_20261109/portable/README.txt)
validates both explicitly, and requires this complete parameter manifest for
measured heads before loading their state. Backbone weights, images, caption
bodies and feature caches must be prepared separately; they are not included.

See the [complete experiment](../../experiments/simplified_20261109/README.txt)
for all positive, negative and inconclusive results and their limitations.
Six original measured heads passed exact full-matrix and rank GPU parity;
the 75 metadata projections passed parameter-byte/reload checks, not 75 GPU
re-evaluations. No end-to-end portable retraining equivalence is claimed.
