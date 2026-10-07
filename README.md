# PARE: Physics-Aware Region Encoding for SAR Text-to-Image Retrieval

Research implementation accompanying *Physics-Aware Region Encoding for SAR
Text-to-Image Retrieval*. The main head is named `SAQREHead` in the source for
historical compatibility; it implements the PARE architecture.

## Release status — 7 October 2026

This release contains the repaired extractor, content-disjoint split generator,
training/evaluation tools, and a measured 919-image extractor diagnostic.
**Full retraining and full-gallery retrieval benchmarks for this repaired version
are pending.** Historical manuscript retrieval tables are not reproduced results
of this code. No trained PARE weights, dataset images, dataset captions, or
pretrained model weights are redistributed.

The extractor changed to float64 local statistics and deterministic spatial
tie-breaking for equal-intensity scatterers. Old feature caches and heads cannot
establish the performance of this version; regenerate and retrain them. The
artifact contract records code, model, environment and control-mode signatures
and refuses incompatible or unversioned artifacts by default.

The source workspace completed a CPU pipeline smoke with 16 fit images,
32 caption-image pairs per head and one epoch: four caches, four trained heads,
and nine evaluations. This is a functional check, not a retrieval benchmark.
See [diagnostic scope](diagnostics/README.md) and [release provenance](RELEASE_PROVENANCE.json).

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
python scripts/run_repaired_experiment.py --device cuda:0 --tag vitb32_full --dry-run
python scripts/run_repaired_experiment.py --device cuda:0 --tag vitb32_full
```

For another backbone, set the arguments from the table and a distinct tag.
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
