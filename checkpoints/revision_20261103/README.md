# Evaluated PARE heads

This directory contains the exact 20 PARE head checkpoints used for the
November formal results: `full`, `stats_off`, `cfar_strict`, and `uniform_grid`
for each of `vitb32`, `vitb16`, `remoteclip`, `siglip`, and `georsclip`.
`HEADS_MANIFEST.json` records every original file SHA-256, byte count, selected
learning rate/temperature, and training scope. The same hashes appear in the
public evaluation projections. These are small PARE heads, not third-party
pretrained image/text backbone weights.

Each head was fitted for 10 epochs on 9,181 images and 31,945 caption-image pairs
after the complete nine-point fit/validation selection for its own backbone.
The state dictionary contains only `SAQREHead` parameters. Checkpoints also keep
their original configuration, loss history, training IDs and provenance hashes;
they contain no original image pixels or caption bodies.

The files are published **without rewriting their bytes**, to retain the exact
hashes of the evaluated artifacts. Some configuration strings therefore contain
historical server paths for the training cache, caption CSV, output file and
external checkpoint. These strings are execution records. Do not recreate those
paths or blindly replay the saved configuration: pass your own paths to the
evaluation CLI.

For example, in the independent evaluation command documented at
[`tools/evaluation/README.md`](../../tools/evaluation/README.md), use:

```sh
--head-full "$PWD/checkpoints/revision_20261103/vitb32/head_full.pt" \
--head-stats-off "$PWD/checkpoints/revision_20261103/vitb32/head_stats_off.pt" \
--head-cfar-strict "$PWD/checkpoints/revision_20261103/vitb32/head_cfar_strict.pt" \
--head-uniform-grid "$PWD/checkpoints/revision_20261103/vitb32/head_uniform_grid.pt"
```

Prepare the original dataset and frozen backbone weights, regenerate the two
matching test caches with the recorded environment/source, and supply their
current paths alongside these heads. Artifact validation checks the matching
backbone, extraction signature, source and head flags. For another backbone,
change the directory and model/pretrained/checkpoint arguments together.
