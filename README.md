# PARE: Physics-Aware Region Encoding for SAR Text-to-Image Retrieval

Research implementation accompanying *Physics-Aware Region Encoding for SAR
Text-to-Image Retrieval*. The main head is named `SAQREHead` in the source for
historical compatibility; it implements the PARE architecture.

## November 2026 revision

The repaired protocol has now been run for CLIP ViT-B/32, CLIP ViT-B/16,
RemoteCLIP, SigLIP and GeoRSCLIP: 45 ten-epoch validation trials, 20 ten-epoch
final heads, and complete retrieval evaluation using 3,688 first-caption queries
and the full 3,688-image test gallery. The release also includes fixed-head
synthetic scale-sensitivity results for ViT-B/32.

See the [measured results and provenance](benchmarks/revision_20261103/README.md),
[all seven methods' full-gallery metrics](benchmarks/revision_20261103/full_gallery_summary.csv),
and [November reproduction instructions](docs/NOVEMBER_REPRODUCTION.md).
The public result files omit caption bodies and machine paths. Distinct hashes
bind original results, public projections and compressed storage. The
[20 trained PARE heads](checkpoints/revision_20261103/README.md) are included with
their original evaluated hashes. Dataset images, captions and third-party
pretrained backbone weights are not redistributed.

Full-gallery R@10 and MRR@10 are higher for PARE than for Scene and stats-off on
all five backbones. However, the separately trained **CFAR-strict control is
higher than full PARE on both metrics for all five backbones**. Both variants
retain soft reliability weighting. These results do not establish a benefit
from the fine detector-derived inputs, or show that every proposed component is
necessary; they also do not isolate the effect of the soft reliability weights.
The complete control results are included in the linked tables.

## Original release status — 7 October 2026

The initial October release contained the repaired extractor, content-disjoint
split generator, training/evaluation tools and a measured 919-image extractor
diagnostic, with full retraining still pending at that date. Its historical
manuscript retrieval tables were not reproduced results of the repaired code.
The completed November experiments above supersede that pending status.

The extractor changed to float64 local statistics and deterministic spatial
tie-breaking for equal-intensity scatterers. Old feature caches and heads cannot
establish the performance of this version; regenerate and retrain them. The
artifact contract records code, model, environment and control-mode signatures
and refuses incompatible or unversioned artifacts by default.

The source workspace completed a CPU pipeline smoke with 16 fit images,
32 caption-image pairs per head and one epoch: four caches, four trained heads,
and nine evaluations. This is a functional check, not a retrieval benchmark.
See [diagnostic scope](diagnostics/README.md) and [original release provenance](provenance/20261007/RELEASE_PROVENANCE.json).

## Installation

Use Python 3.10. The recorded Linux environment uses PyTorch 2.9.1+cu128 and
torchvision 0.24.1+cu128. For the exact recorded environment, install `uv`, then:

```bash
git clone https://github.com/zhaiji0727/PARE.git
cd PARE
bash scripts/setup_environment.sh
source env.sh
python -m pytest tests -q
```

`requirements.lock.txt` records the complete tested environment;
`requirements-runtime.txt` lists the main non-PyTorch requirements. A CPU-only
environment can instead install a suitable CPU build of torch 2.9.1 and
torchvision 0.24.1, followed by `pip install -r requirements-runtime.txt`.
That alternative environment is not the recorded CUDA lockfile environment.
Artifact signatures intentionally distinguish environments.

The tests use synthetic tensors and images. Three real-image regressions skip
until the dataset is prepared. No remote model downloads are needed for the
synthetic tests. Run the tests again after preparation to include real images.

## Prepare the data

The official source is [SARLANG-1M](https://huggingface.co/datasets/YiminJimmy/SARLANG-1M),
pinned to revision `0271408f71bcd9afff69e4e7d324cae912461ebd`.
Obtain and use these data under the upstream terms. The preparation script
downloads the two official ZIP archives, verifies pinned SHA-256 hashes, and
extracts only the 12,879 images and captions needed by the original caption track.
The image archive alone is about 6.79 GB; allow additional space for extraction,
models, feature caches, and experiment outputs.

```bash
python scripts/prepare_data.py
python scripts/prepare_content_disjoint_split.py
```

The resulting layout is:

```text
data/SARimages_pre/                     original image files, flattened by ID
data/Text_unz/Text/Caption/{train,test}/ original caption CSV files
data/ret/                              original split IDs
data/ret_clean_20261007/                regenerated clean IDs, captions, manifest
```

The split generator preserves the test set, removes training images with exact
decoded-RGB content present in test, keeps a canonical ID for within-train
duplicates, and then applies an isolated seed-0 shuffle for fit/validation.
It does not change the official images or captions. Expected counts:

| Split | Images | Caption rows |
| --- | ---: | ---: |
| Train | 9,181 | 31,945 |
| Fit | 8,263 | 28,759 |
| Validation | 918 | 3,186 |
| Test | 3,688 | 6,190 |

`data/splits_original/` contains the preserved original IDs and
`data/splits_content_disjoint_20261007/` contains the expected clean IDs and a
sanitized reference manifest. The latter is a reference, not a runnable data
directory: captions and image fingerprints must be regenerated locally.
Machine paths in the archived manifest were made relative; the recorded hashes
are unchanged. Image-fingerprint JSON includes local paths when regenerated,
so its byte hash varies by machine. Compare the four ID-list hashes to verify
the actual split. Do not copy the reference manifest over a newly generated one.
Exact RGB grouping does not exclude near-duplicates, geographic overlap, or
pretraining-data overlap.

## Prepare frozen backbones

```bash
python scripts/prepare_models.py
python scripts/verify_models.py
```

These download the five backbone configurations below and needed tokenizer
assets into `models/`, and write hashes into `reports/`. The experiment runner
then works offline. Downloads come from the model owners, not this repository.

| Backbone | `--model` | `--pretrained` | Additional `--remoteclip` checkpoint |
| --- | --- | --- | --- |
| CLIP ViT-B/32 | `ViT-B-32` | `laion2b_s34b_b79k` | — |
| CLIP ViT-B/16 | `ViT-B-16` | `laion2b_s34b_b88k` | — |
| RemoteCLIP | `ViT-B-32` | `laion2b_s34b_b79k` | `models/external_checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt` |
| SigLIP | `ViT-B-16-SigLIP` | `webli` | — |
| GeoRSCLIP | `ViT-B-32` | `openai` | `models/external_checkpoints/georsclip/ckpt/RS5M_ViT-B-32.pt` |

The generic `--remoteclip` flag accepts either external checkpoint. Loading
checks exact state-dictionary keys and binds the external file hash to artifacts.

## Small end-to-end smoke

After data and model preparation:

```bash
python scripts/run_repaired_experiment.py --smoke --device cpu --tag smoke_example
```

This uses 16 fit and 16 test images, 32 training pairs, one epoch, four
configurations, and small natural/full-gallery evaluations. It deliberately
skips hyperparameter selection. Use a fresh tag for every changed configuration.
The runner refuses overwrites or reuse with different source/configuration hashes.

## Formal retrieval protocol

Select an available GPU and inspect the exact plan before a full run:

```bash
python scripts/run_repaired_experiment.py --device cuda:0 --tag vitb32_formal_20261103 --out results/vitb32 --dry-run
```

Use the [November stage commands](docs/NOVEMBER_REPRODUCTION.md) for all five
backbones and the independent multi-K evaluator. Those commands preserve the
exact recorded run labels while using short backbone names for output folders.
The formal runner performs:

1. Physics and uniform-grid primitive embedding caches, using frozen encoders.
2. Nine clean fit/validation trials (learning rates 0.0005, 0.001, 0.002;
   contrastive temperatures 0.03, 0.05, 0.07), ten epochs each. Select maximum
   validation R@10, then MRR@10 to break ties; test labels do not select parameters.
3. Ten-epoch clean-train fitting for full PARE, stats-off (`sr,ty`), strict
   CFAR-free (`cfar_strict`), and uniform-grid physics-free (`sr,ty`) controls.
4. All 3,688 test images' first captions as queries, natural candidate sizes
   100, 200, 500, 1000, 2000 and the full 3,688-image gallery; five controlled
   pools and hard-negative evaluation at the initial candidate size; frozen
   training-free baselines. Per-query IDs, ranks and coverage are retained.

Training uses all retained caption-image pairs, keeps the final partial batch,
and can place multiple captions for one image in a contrastive batch. This is
part of the current implementation, not a one-caption-per-image protocol.
Controlled pools insert the ground-truth image, so their recall cannot represent
end-to-end retrieval; natural pools do not insert it. Full-gallery PARE scoring
is query-conditioned and more costly than frozen global-embedding search.

Results go under `results/repaired_20261007/<tag>/`, with exact commands,
logs, hashes, checkpoints, selection results and per-query evaluation records.
`--stage cache|sweep|train|eval` supports staged execution while checking
prerequisites. The shell launchers in `code/` are lower-level historical utilities;
use the Python runner above for the full revised validation protocol.

## Extractor diagnostics

The published [919-image diagnostic](diagnostics/extractor_919/summary.json)
compares the frozen pre-repair and current extractor on the original 919-ID
validation list. To recompute after preparing data:

```bash
python scripts/audit_extractor_fix.py --workers 4 --output results/extractor_diagnostic_new
```

It verifies local statistics against direct sums, detector monotonicity,
nonnegative backgrounds, crop coordinates and exact box-set differences.
It does not measure retrieval or object-detection accuracy. The original
validation list is intentionally retained for this paired implementation audit;
formal training/evaluation uses the new 918-image validation split instead.

## Code map

- `code/saqre_phys.py`: grayscale proxy, ENL, proposals, detector, descriptors and head.
- `code/primitive_controls.py`: strict and uniform-grid controls.
- `code/cache_primitives.py`: frozen primitive embedding extraction.
- `code/train_saqre.py`, `code/sweep_hparams.py`: head training and selection.
- `code/eval_retrieval_phys.py`, `code/eval_retrieval_baselines.py`: evaluation.
- `code/artifacts.py`: compatibility signatures and provenance checks.
- `scripts/`: preparation, the revised runner, and implementation audits.
- `tests/`: numerical, artifact, training-pair and evaluation regressions.

## Citation and terms

Please identify this repository, its exact commit, and the manuscript title when
describing work based on this implementation. A final bibliographic record is
not included because the manuscript is under revision. See
[third-party notices](THIRD_PARTY_NOTICES.md) for upstream resources.
No general-purpose source-code license has yet been selected for this repository;
publication here is not an additional license grant. Third-party packages,
datasets and model weights remain governed by their respective upstream terms.
