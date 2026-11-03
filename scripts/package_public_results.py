"""Compress validated public projections while preserving uncompressed hashes."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

TAGS = ('vitb32', 'vitb16', 'remoteclip', 'siglip', 'georsclip')
METRICS = ('R@1', 'R@5', 'R@10', 'MRR@10')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    inputs = {}
    rows = []
    for tag in TAGS:
        root = args.source / tag
        manifest_bytes = (root / 'projection_manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        assert manifest['status'] == 'complete' and manifest['tag'] == tag
        assert manifest['not_original_result_files'] is True
        assert len(manifest['projection_files']) == (31 if tag == 'vitb32' else 26)
        for name, entry in manifest['projection_files'].items():
            assert Path(name).name == name and name.endswith('.json')
            assert sha((root / name).read_bytes()) == entry['projection_sha256']
        inputs[tag] = (manifest, manifest_bytes)
        base = json.loads((root / 'eval_baselines.json').read_text())['full_gallery']
        for key, label in [('scene', 'Scene'), ('mean_pool', 'Mean-pool'), ('MaxSim', 'MaxSim')]:
            rows.append({'backbone': tag, 'method': label, **base[key]})
        for variant, label in [('full', 'PARE'), ('stats_off', 'PARE-stats-off'),
                               ('cfar_strict', 'PARE-CFAR-strict'), ('uniform_grid', 'PARE-uniform-grid')]:
            metrics = json.loads((root / f'eval_{variant}_k3688.json').read_text())['natural_scene_shortlist']['PARE_end_to_end']
            rows.append({'backbone': tag, 'method': label, **metrics})
    args.out.mkdir(parents=True)
    storage = {'schema': 'pare.compressed-public-projections.v1', 'status': 'complete',
               'description': 'Gzip storage of already-redacted public projections. Original evaluator files are not copied.',
               'gzip_mtime': 0, 'backbones': {}}
    for tag, (manifest, manifest_bytes) in inputs.items():
        source = args.source / tag
        dest = args.out / tag
        dest.mkdir()
        (dest / 'projection_manifest.json').write_bytes(manifest_bytes)
        shutil.copy2(source / 'README.txt', dest / 'README.txt')
        entry = {'projection_manifest_sha256': sha(manifest_bytes), 'compressed_files': {}}
        for name, provenance in manifest['projection_files'].items():
            original = (source / name).read_bytes()
            compressed = gzip.compress(original, compresslevel=9, mtime=0)
            assert gzip.decompress(compressed) == original
            filename = name + '.gz'
            (dest / filename).write_bytes(compressed)
            assert sha(gzip.decompress((dest / filename).read_bytes())) == provenance['projection_sha256']
            entry['compressed_files'][filename] = {
                'gzip_sha256': sha(compressed), 'gzip_bytes': len(compressed),
                'uncompressed_name': name, 'uncompressed_sha256': sha(original),
                'uncompressed_bytes': len(original)}
        storage['backbones'][tag] = entry
    with (args.out / 'full_gallery_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['backbone', 'method', *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    storage['summary_sha256'] = sha((args.out / 'full_gallery_summary.csv').read_bytes())
    (args.out / 'STORAGE_MANIFEST.json').write_text(json.dumps(storage, indent=2) + '\n')
    (args.out / 'README.md').write_text('''# November 2026 measured retrieval results

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
''')
    print(json.dumps({'status': 'complete', 'files': sum(p.is_file() for p in args.out.rglob('*')),
                      'bytes': sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file()),
                      'storage_manifest_sha256': sha((args.out / 'STORAGE_MANIFEST.json').read_bytes())}))


if __name__ == '__main__':
    main()
