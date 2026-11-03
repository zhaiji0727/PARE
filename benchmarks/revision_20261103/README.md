# November 2026 measured retrieval results

These are public, caption-redacted projections of the completed formal runs.
Each backbone has 3,688 first-caption queries and 3,688 gallery images. Five
backbones each have four separately fitted 10-epoch heads and one complete
nine-trial validation selection. No test metric selects hyperparameters.

`full_gallery_summary.csv` contains unrounded percentage metrics for all seven
methods. The 24 per-head/per-K reports plus baseline report retain all per-query
ranks, candidate inclusions and metrics. `queries.json.gz` retains query/image
IDs and caption hashes, but no caption bodies. ViT-B/32 additionally includes the
verified three-condition synthetic-scale results; these are not new acquisitions
or evidence of acquisition-GSD invariance.

Large projection JSON files are gzip-compressed to keep the repository compact.
Each `projection_manifest.json` describes the **uncompressed public projection**
and its distinct original-source JSON hash; it is not the original evaluator
manifest. `STORAGE_MANIFEST.json` additionally binds every gzip file and its
decompressed bytes. To inspect a report without extracting files:

```python
import gzip, json
with gzip.open("vitb32/eval_full_k3688.json.gz", "rt") as stream:
    report = json.load(stream)
print(report["natural_scene_shortlist"])
```

The original complete evaluator files remain the inputs to the strict paper
exporter. Do not pass these public projections to it or relabel them as original
files. No images, caption CSVs, model weights, feature caches, or score matrices
are included here. The source/projection/storage hashes preserve that distinction.

The `tables/` directory provides the complete strict numeric exports: all natural
K conditions, controlled and hard pools, 45 validation trials, chosen parameters,
controls and synthetic-scale results. `tables/ORIGIN.json` records their source
export manifest and byte-identical file hashes; machine-path manifests are omitted.
