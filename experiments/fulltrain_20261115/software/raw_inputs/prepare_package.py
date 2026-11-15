"""Maintain the NEW portable package source lock; never touches frozen experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    paths = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'runtime').rglob('*.py'))
    paths += [ROOT / 'BACKBONES.json', ROOT / 'SOURCE_PROVENANCE.json']
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    lock = {'schema': 'pare.portable-raw-input-source-lock.v1', 'files': files,
            'scope': 'New portable raw reconstruction. Does not modify/reference-release the frozen experiment.'}
    (ROOT / 'SOURCE_LOCK.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(hashlib.sha256((ROOT / 'SOURCE_LOCK.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
