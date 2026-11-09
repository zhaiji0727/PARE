#!/usr/bin/env python3
"""Project all 75 measured heads by redacting metadata paths only.

No tensor, training setting, history value, input hash or scientific identifier
is changed. Original/public file hashes and every parameter's dtype, shape and
bytes are recorded. Requires the completed independent 75/105 audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from checkpoint_contract import validate_measured

TAGS = ('vitb32', 'vitb16', 'remoteclip', 'siglip', 'georsclip')
VARIANTS = ('full', 'strict', 'strict_token_off')
PRIVATE = re.compile(r'/(?:Users|home)/[^\s\"\'<>;,\)\]\}]+')
HOST = re.compile(r'coder\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*')
FORBIDDEN = {'caption', 'captions', 'caption_text', 'query_text', 'password', 'access_token', 'authorization'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def project_metadata(value, field='', changes=None):
    if changes is None:
        changes = []
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str) or PRIVATE.search(key) or HOST.search(key) or key.lower() in FORBIDDEN:
                raise ValueError('Unexpected private key or raw caption/credential metadata')
            output[key] = project_metadata(item, field+'/'+key.replace('~', '~0').replace('/', '~1'), changes)
        return output
    if isinstance(value, (list, tuple)):
        items = [project_metadata(item, field+'/'+str(i), changes) for i, item in enumerate(value)]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, str):
        public = PRIVATE.sub('<PRIVATE_PATH>', HOST.sub('<PRIVATE_HOST>', value))
        if public != value:
            changes.append({'json_pointer': field, 'original_value_sha256': hashlib.sha256(value.encode()).hexdigest(),
                            'public_value': public})
        return public
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError('Metadata contains an unsupported object; tensor changes are not permitted')


def tensor_record(tensor, torch):
    if not torch.is_tensor(tensor):
        raise ValueError('Every state_dict entry must be a tensor')
    value = tensor.detach().cpu().contiguous()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError('Nonfinite checkpoint parameter')
    data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'bytes_sha256': hashlib.sha256(data).hexdigest()}


def compare_states(original, public, torch):
    if set(original) != set(public):
        raise ValueError('Projection changed state_dict keys')
    records = {}
    for key in sorted(original):
        left, right = tensor_record(original[key], torch), tensor_record(public[key], torch)
        if left != right:
            raise ValueError('Projection changed parameter dtype/shape/bytes: ' + key)
        # Direct byte equality is checked as well as its per-parameter digest.
        lbytes = original[key].detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        rbytes = public[key].detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if lbytes != rbytes:
            raise ValueError('Projection changed raw parameter bytes: ' + key)
        records[key] = left
    return records


def measured_args(tag, seed, variant, requirements):
    b = requirements['backbones'][tag]
    return SimpleNamespace(backbone=tag, seed=seed, variant=variant, model=b['model'], pretrained=b['pretrained'],
                           remoteclip=bool(b['external_checkpoint_sha256']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--independent-audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--requirements', type=Path, default=Path(__file__).with_name('SOURCE_REQUIREMENTS.json'))
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    import torch
    requirements = json.loads(a.requirements.read_text())
    audit = json.loads(a.independent_audit.read_text())
    study = requirements['measured_study']
    if not (audit.get('status') == 'pass' and audit.get('heads') == 75 and audit.get('evaluations') == 105 and
            len(audit.get('runs', [])) == 75 and len(audit.get('legacy_anchors', {})) == 10 and len(audit.get('baselines', [])) == 7):
        raise ValueError('Complete independent 75-head/105-evaluation audit required')
    if any(audit.get(k) != study[k] for k in ('protocol_sha256', 'freeze_sha256')):
        raise ValueError('Wrong independently audited study')
    if not a.verify_only and a.out.exists():
        raise FileExistsError('Projection output must be a new directory')
    prior = json.loads((a.out/'HEAD_PROJECTION_MANIFEST.json').read_text()) if a.verify_only else None
    records = {}
    for tag in TAGS:
        for seed in range(5):
            for variant in VARIANTS:
                relative = Path(tag)/f'seed_{seed}'/variant/'head.pt'
                source_relative = Path('results')/relative
                source = a.study/source_relative
                source_hash = sha(source)
                if source_hash != audit['artifact_sha256'][str(source_relative)]:
                    raise ValueError('Original checkpoint differs from independent audit')
                original = torch.load(source, map_location='cpu', weights_only=True)
                args = measured_args(tag, seed, variant, requirements)
                signature = study['cache_signatures'][tag]
                validate_measured(args, original, signature, requirements, 768 if tag == 'siglip' else 512)
                metadata = {k: v for k, v in original.items() if k != 'state_dict'}
                changes = []
                projected_metadata = project_metadata(metadata, changes=changes)
                target = a.out/relative
                if not a.verify_only:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({**projected_metadata, 'state_dict': original['state_dict']}, target)
                public = torch.load(target, map_location='cpu', weights_only=True)
                public_metadata = {k: v for k, v in public.items() if k != 'state_dict'}
                if object_sha(projected_metadata) != object_sha(public_metadata):
                    raise ValueError('Saved projection changed non-path metadata')
                parameters = compare_states(original['state_dict'], public['state_dict'], torch)
                validate_measured(args, public, signature, requirements, original['D'])
                record = {'original_relative_file': str(source_relative), 'original_sha256': source_hash,
                          'public_sha256': sha(target), 'original_metadata_sha256': object_sha(metadata),
                          'public_metadata_sha256': object_sha(public_metadata), 'path_only_projection': changes,
                          'parameters': parameters, 'all_parameter_dtype_shape_bytes_identical': True,
                          'backbone': tag, 'seed': seed, 'variant': variant}
                if a.verify_only and prior['files'].get(str(relative)) != record:
                    raise ValueError('Projection manifest mismatch: ' + str(relative))
                records[str(relative)] = record
    if len(records) != 75:
        raise ValueError('Incomplete projection grid')
    report = {'schema': 'pare.path-only-head-projection.v1', 'status': 'all_75_heads_parameter_bytes_verified',
              'count': 75, 'original_checkpoint_schema': 'pare.simplified-head.v1',
              'protocol_sha256': study['protocol_sha256'], 'freeze_sha256': study['freeze_sha256'],
              'independent_audit_sha256': sha(a.independent_audit), 'projection_source_sha256': sha(__file__),
              'requirements_sha256': sha(a.requirements), 'files': records,
              'note': 'Only metadata path strings are redacted. Public file hashes differ; parameter dtype, shape and bytes are identical. No retraining occurred.'}
    if not a.verify_only:
        (a.out/'HEAD_PROJECTION_MANIFEST.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    elif prior != report:
        raise ValueError('Projection manifest header or complete file set differs')
    print(json.dumps({'status': report['status'], 'count': len(records), 'verify_only': a.verify_only}))


if __name__ == '__main__':
    main()
