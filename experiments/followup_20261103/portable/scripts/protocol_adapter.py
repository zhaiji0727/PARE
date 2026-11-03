"""Narrow portability boundary around the immutable exploratory protocol.

Only cache serialization and local split-metadata byte hashes can be rebound.
All scientific fields remain exactly equal to the original executed protocol.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

ORIGINAL_PROTOCOL_SHA256 = '42789ee8793b7c5bd59bf950dd8deefe8bf1c3caa2e9be2d54143630f63bf288'
ORIGINAL_FREEZE_SHA256 = '3ebc187c1f9b67256747bc46e5383d43a166fdb11cb5441aea2660a937aa011a'
REBOUND_FIELDS = ['input_bindings.training_cache_sha256',
                  'input_bindings.original_split_manifest_sha256']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_original(original_root):
    root = Path(original_root)
    path = root / 'plans/protocol.json'
    if sha(path) != ORIGINAL_PROTOCOL_SHA256:
        raise ValueError('Bundled original protocol is not the executed revision 2')
    plan = json.loads(path.read_text())
    if sha(root / 'plans/protocol.md') != plan['protocol_markdown_sha256']:
        raise ValueError('Bundled original protocol markdown changed')
    freeze_path = root / 'FOLLOWUP_SOURCE_FROZEN.json'
    if sha(freeze_path) != ORIGINAL_FREEZE_SHA256:
        raise ValueError('Original executed source manifest changed')
    for relative, digest in json.loads(freeze_path.read_text())['file_sha256'].items():
        if sha(root / relative) != digest:
            raise ValueError('Original executed source changed: ' + relative)
    return plan


def check_scientific_identity(plan, original):
    """Reject every amendment except the two bindings and declared metadata."""
    if plan == original:
        return
    candidate = deepcopy(plan)
    meta = candidate.pop('reproduction', None)
    keys = {'schema', 'parent_protocol_sha256', 'created_utc',
            'scientific_design_unchanged', 'rebound_fields'}
    if not isinstance(meta, dict) or set(meta) != keys:
        raise ValueError('Expected explicit, narrowly defined reproduction metadata')
    if (meta['schema'] != 'pare.portable-reproduction.v1'
            or meta['parent_protocol_sha256'] != ORIGINAL_PROTOCOL_SHA256
            or meta['scientific_design_unchanged'] is not True
            or meta['rebound_fields'] != REBOUND_FIELDS):
        raise ValueError('Invalid reproduction-parent declaration')
    try:
        date = datetime.fromisoformat(meta['created_utc'])
        if date.tzinfo is None:
            raise ValueError('missing timezone')
    except (ValueError, TypeError):
        raise ValueError('Reproduction timestamp must be a timezone-aware ISO datetime')
    for field in ('training_cache_sha256', 'original_split_manifest_sha256'):
        if not is_sha(candidate.get('input_bindings', {}).get(field)):
            raise ValueError('Invalid rebound file digest: ' + field)
        candidate['input_bindings'][field] = original['input_bindings'][field]
    if candidate != original:
        raise ValueError('Scientific protocol changed outside the two allowed byte bindings')


def make_reproduction_protocol(original, cache_sha256, split_manifest_sha256, created_utc):
    result = deepcopy(original)
    result['input_bindings']['training_cache_sha256'] = cache_sha256
    result['input_bindings']['original_split_manifest_sha256'] = split_manifest_sha256
    result['reproduction'] = {'schema': 'pare.portable-reproduction.v1',
                              'parent_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
                              'created_utc': created_utc,
                              'scientific_design_unchanged': True,
                              'rebound_fields': REBOUND_FIELDS[:]}
    check_scientific_identity(result, original)
    return result


def validate_reproduction_protocol(path, expected_sha256, original_root):
    if not is_sha(expected_sha256) or sha(path) != expected_sha256:
        raise ValueError('Protocol does not match the explicitly supplied SHA-256')
    original = load_original(original_root)
    plan = json.loads(Path(path).read_text())
    if 'reproduction' not in plan and expected_sha256 != ORIGINAL_PROTOCOL_SHA256:
        raise ValueError('Unmodified original protocol must retain its exact original bytes')
    check_scientific_identity(plan, original)
    if sha(Path(path).with_suffix('.md')) != original['protocol_markdown_sha256']:
        raise ValueError('Reproduction must use the unmodified original protocol markdown')
    return plan


def check_cache_contract(cache, signature, original, train_ids):
    """Additional fixed-study checks after the original validate_cache succeeds."""
    if (cache.get('D'), cache.get('Mc'), cache.get('Mf')) != (512, 6, 8):
        raise ValueError('Expected original B32 dimension and region capacities: 512/6/8')
    if (signature.get('primitive_mode') != 'physics'
            or signature.get('model') != original['backbone']['model']
            or signature.get('pretrained') != original['backbone']['pretrained']
            or signature.get('pretrained_weight_sha256') != original['backbone']['pretrained_weight_sha256']
            or signature.get('external_checkpoint_sha256') is not None):
        raise ValueError('Cache does not implement the fixed physics/B32 backbone contract')
    ids = cache.get('ids', [])
    if len(ids) != 9181 or len(set(ids)) != 9181 or set(ids) != set(train_ids) or len(train_ids) != 9181:
        raise ValueError('Cache must contain exactly the original 9181 training IDs')
