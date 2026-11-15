"""Validate only complete original-scale v9 train caches, never old-schema bypasses."""
import hashlib
from pathlib import Path
import re
from cache_common import DIMENSIONS, objsha, read, require

SCHEMA = 'pare.legacy-fine-inputs.v1'
PRESERVED = ['scene', 'nu', 'c_emb', 'c_mu', 'c_t', 'c_tf', 'c_n', 'L', 'c_star']
FINE = ['f_emb', 'f_mu', 'f_t', 'f_tf', 'f_n']


def normalized_transform(value):
    return re.sub(r' at 0x[0-9a-fA-F]+', '', str(value))


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def validate_audit_header(audit, row, tag, ids, source_lock_sha):
    contract, task, output = audit['contract'], audit['task'], audit['output']
    require(audit['status'] == 'input_cache_built' and audit['preserved_fields_exact'] is True and
            audit['serialized_output_roundtrip_exact'] is True, 'Incomplete/failed v9 build audit')
    require(contract['schema_version'] == SCHEMA and contract['population'] == 'train_9181' and
            contract['factor'] == '1', 'Reject SMOKE/test/scale/old-schema cache')
    require(task == row['v9_task'] and contract['input_task_sha256'] == objsha(task), 'v9 task audit differs')
    require((task['backbone'], task['D'], task['population'], task['factor'], task['id']) ==
            (tag, DIMENSIONS[tag], 'train', '1', tag+'_train'), 'v9 task identity/dimension')
    require(contract['source_lock_sha256'] == source_lock_sha, 'v9 build source lock differs')
    require(contract['parent_cache_sha256'] == row['parent_cache']['sha256'] and
            contract['parent_extraction_signature'] == contract['base_extraction_signature'] == row['parent_extraction_signature'],
            'Parent extraction signature is not the bound original-scale signature')
    require(contract['ids_sha256'] == objsha(ids), 'Audit ordered population checksum differs')
    require(contract['fine_crop_half_pixels'] == 12 and contract['fine_slots'] == 8 and
            set(contract['preserved_tensor_sha256']) == set(PRESERVED) and contract['changed_fields'] == FINE+['agg'], 'v9 field/crop contract')
    require(contract['encoding_policy'] == 'each image: one float32 eval forward of 8 fine crops; RGB-black 24x24 right padding; padding discarded' and
            contract['precision'] == 'float32, no autocast, no TF32, deterministic algorithms, CUBLAS :4096:8', 'v9 encoding contract changed')
    for key in ('path', 'sha256', 'bytes', 'signature_sha256'):
        require(output[key] == row['cache'][key], 'Audit output binding differs: ' + key)
    require([r['id'] for r in audit['images']] == ids, 'Audit missing/reordered/full image IDs')
    require(all(r['scale_record'] is None for r in audit['images']), 'Transformed images are not allowed')


def validate_layout(torch, cache, dimension, ids, label):
    require((cache['D'], cache['Mc'], cache['Mf']) == (dimension, 6, 8), label + ' D/slots')
    require(cache['ids'] == ids and len(ids) == len(set(ids)) == 9181, label + ' complete ordered train IDs')
    require(all(isinstance(x, str) and Path(x).name == x for x in ids), label + ' basename IDs')
    fields = set(PRESERVED) | set(FINE) | {'agg'}
    require({k for k, v in cache.items() if torch.is_tensor(v)} == fields, label + ' unexpected tensor layout')
    shapes = {'scene': (9181, dimension), 'agg': (9181, dimension),
              **{k: (9181,) for k in ('nu', 'L', 'c_star')}}
    for branch, slots in (('c', 6), ('f', 8)):
        shapes.update({branch+'_emb': (9181, slots, dimension), branch+'_tf': (9181, slots, 4),
                       branch+'_mu': (9181, slots), branch+'_t': (9181, slots), branch+'_n': (9181,)})
        n = cache[branch+'_n']
        require(n.dtype in (torch.int32, torch.int64) and ((n >= 0) & (n <= slots)).all().item(), label + ' counts')
    for key, shape in shapes.items():
        value = cache[key]
        require(tuple(value.shape) == shape and torch.isfinite(value).all().item(), label + ' shape/finiteness: ' + key)
        if key not in ('c_n', 'f_n'):
            require(value.dtype == torch.float32, label + ' float32: ' + key)
    require('scale_test' not in cache, label + ' must be original scale')


def load_checked_cache(catalog, tag):
    """Caller checks all declared file hashes first; this verifies tensor contents."""
    import torch
    row = catalog['backbones'][tag]
    ids = read(catalog['common']['train_ids']['path'])
    fit, val = [read(catalog['common'][key]['path']) for key in ('fit_ids', 'val_ids')]
    require(len(ids) == len(set(ids)) == 9181 and len(fit) == len(set(fit)) == 8263 and
            len(val) == len(set(val)) == 918 and set(fit).isdisjoint(val) and set(fit)|set(val) == set(ids), 'Complete fit/val partition')
    audit = read(row['build_audit']['path'])
    validate_audit_header(audit, row, tag, ids, catalog['v9_builder_source_lock_sha256'])
    parent = torch.load(row['parent_cache']['path'], map_location='cpu', weights_only=True)
    cache = torch.load(row['cache']['path'], map_location='cpu', weights_only=True)
    validate_layout(torch, parent, DIMENSIONS[tag], ids, 'parent')
    validate_layout(torch, cache, DIMENSIONS[tag], ids, 'v9 output')
    pp, cp = parent['provenance'], cache['provenance']
    require(pp['schema_version'] == 2 and pp['signature'] == row['parent_extraction_signature'] and
            pp['signature_sha256'] == objsha(pp['signature']), 'Parent schema/signature digest')
    require(pp['ids_sha256'] == objsha(ids) and pp['ids_file_sha256'] == catalog['common']['train_ids']['sha256'] and
            set(pp['image_file_sha256']) == set(ids), 'Parent complete ID/image provenance')
    contract = audit['contract']
    expected_signature = {**contract, 'fine_tie_selection': 'legacy_raster'}
    require(cp['schema_version'] == SCHEMA and cp['signature'] == expected_signature and
            cp['signature_sha256'] == objsha(expected_signature) == row['cache']['signature_sha256'], 'New schema/signature digest')
    require(cp['ids_sha256'] == objsha(ids) and cp['ids_file_sha256'] == pp['ids_file_sha256'] and
            cp['image_file_sha256'] == pp['image_file_sha256'], 'New cache full image provenance')
    require({r['id']: r['image_sha256'] for r in audit['images']} == pp['image_file_sha256'], 'Build audit image hashes')
    require(cp['parent_provenance'] == pp and cp['parent_non_tensor_metadata_sha256'] ==
            objsha({k: v for k, v in parent.items() if not torch.is_tensor(v)}), 'Preserved parent metadata')
    require(cache['config'] == {'schema_version': SCHEMA, 'parent_config': parent['config'],
            'task_id': tag+'_train', 'fine_tie_selection': 'legacy_raster'}, 'New cache config')
    require(normalized_transform(cp['eval_transform']) == normalized_transform(pp['eval_transform']) == contract['preprocess'], 'Image preprocessing differs')
    for key, value in parent.items():
        if not torch.is_tensor(value) and key not in ('ids', 'config', 'provenance'):
            require(cache[key] == value, 'Other parent metadata changed: ' + key)
    hashes = {k: tensor_sha(v) for k, v in cache.items() if torch.is_tensor(v)}
    require(hashes == audit['output']['tensor_sha256'], 'Output tensor checksums differ from build audit')
    for key in PRESERVED:
        require(hashes[key] == tensor_sha(parent[key]) == contract['preserved_tensor_sha256'][key], 'Preserved tensor changed: ' + key)
    for i, record in enumerate(audit['images']):
        count = int(cache['f_n'][i])
        require(count == len(record['boxes']) and 0 <= count <= 8, 'Fine count/boxes differ')
        require(all(len(b) == 4 and all(type(x) is int for x in b) and b[0] < b[2] and b[1] < b[3] for b in record['boxes']), 'Invalid fine boxes')
    receipt = {'schema': SCHEMA, 'task_id': tag+'_train', 'dimension': DIMENSIONS[tag],
               'cache_sha256': row['cache']['sha256'], 'cache_signature_sha256': cp['signature_sha256'],
               'parent_sha256': row['parent_cache']['sha256'], 'audit_sha256': row['build_audit']['sha256'],
               'full_train_ids_sha256': objsha(ids), 'train_images': 9181, 'fit_images': 8263, 'val_images': 918,
               'output_tensor_sha256': hashes, 'preserved_parent_tensors_exact': True}
    del parent
    return cache, receipt
