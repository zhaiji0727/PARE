# Synthetic scale and raster-spacing checks

The tools use the root repository's recorded environment. `scale_contract.py`
uses Pillow and the standard library; header inspection uses the standard
library. Cache generation/evaluation additionally use the unchanged PARE code,
PyTorch, OpenCLIP, NumPy and SciPy.

`stress_cache.py` resamples each source image with factors 1, 1.5, or 2 and binds
the exact transformation, source-image hashes, original reference-cache hash,
and helper hashes to a separately signed cache. A generic PARE evaluator rejects
this derived signature. `evaluate_scale.py` is the explicit validating adapter;
it checks the transformation, original cache, fixed heads and complete gallery
before calling the unchanged independent score functions. It cannot treat a
transformed cache as an original-resolution cache without that validation.

Example, from the repository root, after completing the ViT-B/32 experiment:

```sh
export CUDA_VISIBLE_DEVICES=0
source env.sh
python tools/scale/stress_cache.py --code-root "$PWD/code" \
  --reference-cache "$PWD/results/vitb32/cache_physics_test.pt" \
  --ids "$PWD/data/ret_clean_20261007/test_ids.json" \
  --img-root "$PWD/data/SARimages_pre" --factor 1.5 \
  --out "$PWD/results/scale_1p5.pt" --device cuda
```

Repeat with `--factor 2` and a fresh output path. Consult each command's `--help`
for explicit model/checkpoint arguments. `evaluate_scale.py` requires both scale
caches, the original physics cache, full and stats-off heads, captions, the
independent `tools/evaluation/evaluate_multik.py`, and `code/batch_scoring.py`.
Use one fixed pair of trained heads for all three scales. A factor-1 regenerated
cache can additionally verify identity with the original extraction.

`inspect_raster_spacing.py` reads existing headers without extracting archives or
decoding pixel arrays. It interprets only supported projected GeoTIFF metadata
with explicit metre units. `spacing_groups.py` reports groups without inferring
missing sensor/acquisition metadata. Such header pixel spacing is not, by itself,
verified acquisition GSD. Synthetic degradation is likewise not a new acquisition.

## Exact experiment source versus portable source

The executed November scale sources are preserved byte-for-byte in
`provenance/20261103/scale_executed/`, with hashes in `RELEASE_PROVENANCE.json`.
Their GPU guard reserved physical GPU5 because GPUs0--4 were running the five
backbones. This public `tools/scale` copy allows any **one explicitly selected**
`CUDA_VISIBLE_DEVICES` entry and updates the associated tests/code-root default.
The exact diff is recorded beside the original sources. Extraction, resampling,
model and metric formulas are unchanged.

The portable `stress_cache.py` consequently has a different helper hash. Rebuild
both derived caches with this portable copy before evaluating them with it.
To validate the original experiment's derived cache bytes, use the preserved
exact original tools and their original signatures. Do not relabel old caches
with the portable hash. Original evaluator/exporter scripts are unchanged and
retain their explicit provenance checks.
