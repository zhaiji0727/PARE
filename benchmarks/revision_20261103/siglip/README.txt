These files are sanitized projections, NOT the original evaluator JSON files.
Caption bodies and absolute machine paths are omitted. Caption SHA-256 digests, image/query IDs, ranks and metrics are retained.
projection_manifest.json maps every original JSON SHA-256 to its distinct projection SHA-256 and binds model, final heads, selection and split IDs.
No model weights, cache/score matrices, source caption CSVs or original machine-path manifests are copied.
Internal paper-result exporters must read the original full result directories. These projections are for public inspection only.
Optional stress_* files are fixed-head synthetic-scale projections, not new acquisition-GSD measurements.
A partial original training-run manifest is permissible only because evaluation ran independently; the sweep and all final heads were verified complete.
