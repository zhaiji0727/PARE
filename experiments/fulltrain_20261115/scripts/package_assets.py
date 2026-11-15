"""Create separate code/results and weight release archives. No upload or git commands."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_files(root):
    root = root.resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    if manifest['status'] != 'exported_from_complete_independent_audits' or (root/'EXPORT_FAILED.json').exists():
        raise ValueError('Only a fully completed local export may be packaged')
    records = dict(manifest['files'])
    records['MANIFEST.json'] = {'sha256': sha(root/'MANIFEST.json')}
    paths = []
    for name, row in records.items():
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file() or (root/name).is_symlink() or sha(path) != row['sha256']:
            raise ValueError('Release file missing/changed/unsafe: '+name)
        paths.append((name, path))
    declared = ''.join(f'{row["sha256"]}  {name}\n' for name, row in sorted(records.items()))
    if (root/'SHA256SUMS').read_text() != declared:
        raise ValueError('SHA256SUMS does not bind the complete export')
    paths.append(('SHA256SUMS', root/'SHA256SUMS'))
    weights = [(n,p) for n,p in paths if n.startswith('weights/') and n.endswith('.pt')]
    if len(weights) != 290:
        raise ValueError('All 290 weights are required')
    lightweight = [(n,p) for n,p in paths if (n,p) not in weights]
    return lightweight, weights


def archive(destination, records):
    with destination.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for name, path in sorted(records):
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                    with path.open('rb') as source:
                        tar.addfile(info, source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    root = args.release.resolve()
    light, weights = checked_files(root)
    plan = {'schema': 'pare.public-release-assets.v1', 'status': 'planned_only', 'published': False,
            'code_results_files': len(light), 'weight_files': len(weights),
            'weight_bytes': sum(p.stat().st_size for _,p in weights),
            'extraction': 'Extract both archives into the same new directory; verify SHA256SUMS.'}
    if args.execute:
        if args.output.exists():
            raise ValueError('Output must be a new directory')
        args.output.mkdir(parents=True)
        try:
            assets = []
            for name, records in [('pare-fulltrain-code-results.tar.gz', light), ('pare-fulltrain-weights.tar.gz', weights)]:
                path = args.output/name; archive(path, records)
                assets.append({'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size})
            checked_files(root)
            plan.update(status='local_assets_created_not_published', assets=assets)
            (args.output/'ASSETS.json').write_text(json.dumps(plan, indent=2)+'\n')
            (args.output/'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {r["path"]}\n' for r in assets))
        except BaseException as error:
            (args.output/'PACKAGING_FAILED.json').write_text(json.dumps({'error': str(error)}))
            raise
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
