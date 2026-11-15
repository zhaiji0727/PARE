"""Portable reader bindings. Independent of the formal-score release gate."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from data_contract import objsha, query_sets


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Reader:
    def __init__(self):
        self.files = {}

    def bind(self, binding, parent=ROOT):
        require(isinstance(binding, dict) and isinstance(binding.get('path'), str)
                and isinstance(binding.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', binding['sha256']), 'Explicit path and SHA256 required')
        path = Path(binding['path']).expanduser()
        path = (path if path.is_absolute() else parent / path).resolve()
        require(sha(path) == binding['sha256'], 'File hash differs: ' + str(path))
        require(str(path) not in self.files or self.files[str(path)] == binding['sha256'], 'Conflicting input binding')
        self.files[str(path)] = binding['sha256']
        return path

    def read(self, binding, parent=ROOT):
        return json.loads(self.bind(binding, parent).read_text())

    def finish(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, 'Source/input changed during evaluation: ' + path)


def source_lock(reader):
    lock_path = ROOT / 'SOURCE_LOCK.json'
    lock = reader.read({'path': str(lock_path), 'sha256': sha(lock_path)})
    required = {'evaluate.py', 'evaluation_inputs.py', 'runtime/evaluate_core.py',
                'runtime/scoring_contract.py', 'runtime/pare_portable/api.py',
                'runtime/data_contract.py', 'REFERENCE_TRAINING_PROTOCOL.json', 'FIXED_QUERY_PROTOCOL.json'}
    require(required <= set(lock['files']), 'Incomplete evaluator source lock')
    for name, digest in lock['files'].items():
        path = (ROOT / name).resolve()
        require(not Path(name).is_absolute() and ROOT in path.parents, 'Unsafe source lock path')
        reader.bind({'path': str(path), 'sha256': digest})
    return sha(lock_path)


def validate_fixed_queries(queries, fixed):
    actual = {}
    for name in ('first', 'extra', 'all'):
        actual[name + '_text_sha256'] = objsha([r['caption_sha256'] for r in queries[name]])
        actual[name + '_query_ids_sha256'] = objsha([r['query_id'] for r in queries[name]])
    actual['all_caption_row_map_sha256'] = objsha(queries['all_caption_row_map'])
    require(actual == fixed['digests'], 'Caption content or query order differs from fixed full benchmark')


def load_inputs(reader, binding, backbone, smoke):
    """Load raw-builder test output and rebuild every query using data_contract."""
    from pare_portable.io import load_tensor_artifact, documents_from_cache
    import torch
    path = reader.bind(binding)
    manifest = reader.read(binding)
    expected_status = 'SMOKE_ONLY' if smoke else 'BOUND'
    require(manifest.get('schema') == 'pare.portable-retrieval-inputs.v1'
            and manifest.get('status') == expected_status, 'Input status/schema differs from selected scope')
    require(manifest['backbone'] == backbone, 'Test input backbone differs')
    dimension = 768 if backbone == 'siglip' else 512
    require(manifest['dimension'] == dimension, 'Backbone dimension differs')
    paths = {key: reader.bind(manifest[key], path.parent) for key in ('cache', 'query_text', 'query_identity')}
    cache = load_tensor_artifact(paths['cache'], manifest['cache']['sha256'])
    text = load_tensor_artifact(paths['query_text'], manifest['query_text']['sha256'])
    declared = reader.read(manifest['query_identity'], path.parent)
    require((cache['D'], cache['Mc'], cache['Mf']) == (dimension, 6, 8), 'Cache dimensions/capacities differ')
    ids = cache['ids']
    require(ids and len(ids) == len(set(ids)), 'Nonempty unique gallery required')
    expected_tensors = {'scene', 'agg', 'nu', 'L', 'c_star'} | {
        branch + '_' + field for branch in ('c', 'f') for field in ('emb', 'mu', 't', 'tf', 'n')}
    require({k for k, v in cache.items() if torch.is_tensor(v)} == expected_tensors, 'Cache tensor schema differs')
    provenance = cache['provenance']; signature = provenance['signature']
    require(provenance['schema_version'] == signature['schema'] == 'pare.portable-raw-cache.v1'
            and signature['origin'] == 'new_raw_image_reproduction'
            and signature['population'] == 'test' and signature['smoke'] is smoke,
            'Cache is not a new raw test reproduction of the selected scope')
    require(objsha(signature) == provenance['signature_sha256'] and objsha(ids) == provenance['ids_sha256']
            == signature['ids_sha256'], 'Cache identity/signature differs')
    require('scale_test' not in cache, 'This entry accepts original-scale test inputs only')
    require(manifest['encoder_signature'] == signature['encoder'], 'Test encoder binding differs')
    if not smoke:
        require(len(ids) == 3688 and objsha(ids) == 'e1982fc39e596abb1d63bf4bd78cf77a482696af42f2b1c3c2f5ead087fc26d7',
                'Official test gallery identity differs')
    captions = {}
    for row in declared['all']:
        require(row['image_id'] in ids and isinstance(row['caption'], str), 'Invalid declared caption')
        captions.setdefault(row['image_id'], []).append(row['caption'])
    queries = query_sets(ids, captions, validate_official=not smoke)
    require(declared == queries, 'Declared query identities/order differ from reconstructed data_contract')
    if not smoke:
        fixed = json.loads((ROOT / 'FIXED_QUERY_PROTOCOL.json').read_text())
        validate_fixed_queries(queries, fixed)
    require(len(queries['first']) == len(ids), 'Every gallery image needs a first-caption query')
    require(set(text) == {'first', 'extra'}, 'Unexpected query-text keys')
    for key in ('first', 'extra'):
        value = text[key]
        require(value.dtype == torch.float32 and tuple(value.shape) == (len(queries[key]), dimension)
                and bool(torch.isfinite(value).all()), 'Invalid query text tensor: ' + key)
    for key in ('L', 'c_star'):
        require(cache[key].shape == (len(ids),) and cache[key].dtype == torch.float32
                and bool(torch.isfinite(cache[key]).all()), 'Invalid numeric cache field: ' + key)
    documents_from_cache(cache, ids, 'cpu')  # Existing validator, before any CUDA move.
    return manifest, cache, text, queries


def validate_metadata(metadata, backbone, model_spec):
    from pare_portable.spec import ModelSpec
    protocol = json.loads((ROOT / 'REFERENCE_TRAINING_PROTOCOL.json').read_text())
    model = ModelSpec.parse(model_spec)
    variant = 'path_' + model.abc + '__' + model.control if model.family == 'path' else model.variant
    require(backbone in protocol['backbones'] and variant in protocol['models_by_backbone'][backbone],
            'Model/backbone absent from the fixed training matrix')
    require(metadata['backbone'] == backbone and metadata['variant'] == variant, 'Checkpoint model/backbone differs')
    require(metadata['seed'] in protocol['seeds'] and type(metadata['seed']) is int, 'Training seed differs')
    require(metadata['dimension'] == (768 if backbone == 'siglip' else 512), 'Checkpoint dimension differs')
    require(metadata['train_images'] == 9181 and metadata['train_pairs'] == 31945
            and metadata['training_texts'] == 18461
            and metadata['epochs'] == 50 and metadata['checkpoint_epochs'] == [20, 50]
            and metadata['initialization'] == 'fresh_from_seed'
            and metadata['descriptor_mean_population'] == 'all_valid_regions_of_all_9181_training_images',
            'Checkpoint is not the fixed complete training recipe')
    return variant


def checkpoint_identity(reader, checkpoint, checkpoint_binding, backbone, model_spec,
                        test_encoder, training_inputs=None, published_binding=None):
    """Return identity for either a new portable run or an audited public projection."""
    from text_helpers import state_sha
    metadata = checkpoint['metadata']
    variant = validate_metadata(metadata, backbone, model_spec)
    state = checkpoint['state_dict']
    for name, key in [('coarse_mean', 'full_training_crop_mean_c'), ('fine_mean', 'full_training_crop_mean_f')]:
        require(state[name].tolist() == metadata[key], 'Training descriptor mean metadata differs')
    if training_inputs is not None:
        require(metadata['schema'] == 'pare.portable-full-training-replay.v1', 'Expected a new portable training checkpoint')
        require(metadata['inputs_manifest_sha256'] == training_inputs['sha256'], 'Training INPUTS does not belong to checkpoint')
        train = reader.read(training_inputs)
        require(train['schema'] == 'pare.portable-full-training-inputs.v1' and train['status'] == 'BOUND'
                and train['backbone'] == backbone and train['dimension'] == metadata['dimension'], 'Training INPUTS identity differs')
        for key in ('cache', 'training_text', 'captions'):
            require(train[key]['sha256'] == metadata['input_files'][key]['sha256'], 'Checkpoint training artifact binding differs')
        require(train['encoder_signature'] == test_encoder, 'Train/test encoder signatures differ')
        require(train['training_ids_sha256'] == 'aebf2530ab350d94f3a325b514ba872ed1fa0e434ed9d7973c2c209a502b74f5'
                and train['pair_order_sha256'] == metadata['training_pair_order_sha256']
                and train['training_text_sha256'] == metadata['training_text_sha256'], 'Training population/text identity differs')
        history = checkpoint['history']; epoch = len(history)
        require(epoch in (20, 50) and [r['epoch'] for r in history] == list(range(1, epoch + 1))
                and all(r['batches'] == 500 for r in history), 'Checkpoint budget/history differs')
        identity = {'origin': 'new_portable_full_training', 'epoch': epoch, 'training_inputs': training_inputs}
    else:
        require(metadata['schema'] == 'pare.final-full-train.run.v1' and published_binding is not None,
                'Formal published weights require an explicit publication sidecar')
        sidecar_path = reader.bind(published_binding)
        sidecar = reader.read(published_binding)
        require(sidecar['schema'] == 'pare.published-checkpoint-binding.v1'
                and sidecar['status'] == 'projected_from_audited_checkpoint', 'Unrecognized publication binding')
        require(all(isinstance(sidecar.get(key), str) and re.fullmatch('[0-9a-f]{64}', sidecar[key])
                    for key in ('source_checkpoint_sha256', 'training_protocol_sha256', 'source_training_audit_sha256')),
                'Malformed source audit/checkpoint hash')
        require(sidecar['checkpoint_sha256'] == checkpoint_binding['sha256']
                and sidecar['state_dict_sha256'] == state_sha(state), 'Published checkpoint/state binding differs')
        require(all(sidecar[k] == metadata[k] for k in ('backbone', 'variant', 'seed'))
                and sidecar['epoch'] in (20, 50), 'Published checkpoint identity differs')
        require(sidecar['training_protocol_sha256'] == metadata['protocol_sha256']
                and sidecar['training_population'] == {'images': 9181, 'pairs': 31945, 'texts': 18461},
                'Published training recipe differs')
        require(sidecar['encoder_signature'] == test_encoder, 'Published train/test encoder signatures differ')
        audit = reader.read(sidecar['training_audit'], sidecar_path.parent)
        require(audit['schema'] == 'pare.published-training-audit.v1' and audit['status'] == 'audit_passed'
                and audit['source_training_audit_sha256'] == sidecar['source_training_audit_sha256']
                and audit['training_protocol_sha256'] == sidecar['training_protocol_sha256'], 'Public training audit binding differs')
        matching = [r for r in audit['checkpoints'] if all(r[k] == sidecar[k] for k in ('backbone', 'variant', 'seed', 'epoch'))]
        require(len(matching) == 1 and matching[0]['source_checkpoint_sha256'] == sidecar['source_checkpoint_sha256'],
                'Checkpoint absent/duplicated in the public audit inventory')
        identity = {'origin': 'published_projection_of_formal_checkpoint', 'epoch': sidecar['epoch'],
                    'publication_binding': published_binding, 'sidecar': sidecar}
    return {**identity, 'backbone': backbone, 'variant': variant, 'seed': metadata['seed']}
