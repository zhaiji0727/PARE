"""Strict, schema-specific validation for two documented checkpoint formats.

This portable adapter is not a modification of either experiment's frozen code.
Path values are deliberately not identities: path-only public projections retain
the original file hashes, model state, settings and scientific provenance.
"""
from functools import lru_cache
import hashlib
import json
import math
import random

MEASURED_SCHEMA = 'pare.simplified-head.v1'
FLAGS = {'full': [], 'strict': ['cfar_strict'], 'strict_token_off': ['cfar_strict', 'ty']}


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=5)
def pair_order_hashes(seed):
    order = list(range(31945))
    generator = random.Random(seed)
    hashes = []
    for _ in range(10):
        generator.shuffle(order)
        hashes.append(object_sha(order))
    return tuple(hashes)


def validate_backbone(args, signature, requirements):
    expected = requirements['backbones'].get(args.backbone)
    if expected is None or (args.model, args.pretrained) != (expected['model'], expected['pretrained']):
        raise ValueError('Backbone label/model/pretrained mismatch')
    for key in ('model', 'pretrained', 'pretrained_weight_sha256', 'external_checkpoint_sha256'):
        if signature.get(key) != expected[key]:
            raise ValueError('Wrong backbone weight identity: ' + key)
    if bool(args.remoteclip) != bool(expected['external_checkpoint_sha256']):
        raise ValueError('External checkpoint requirement differs from backbone label')
    if args.seed not in range(5) or args.variant not in FLAGS:
        raise ValueError('Unsupported fixed-study seed or variant')
    return expected


def validate_original_trainer(args, checkpoint, signature, requirements):
    """Additional fixed-recipe checks; original artifacts.validate_head also runs."""
    expected = validate_backbone(args, signature, requirements)
    if checkpoint.get('schema') is not None:
        raise ValueError('Unknown schema is not accepted as an original trainer checkpoint')
    config = checkpoint['config']
    required = {'model': expected['model'], 'pretrained': expected['pretrained'],
                'epochs': 10, 'bs': 64, 'text_bs': 256, 'lr': .002,
                'kappa': expected['kappa'], 'weight_decay': .0001, 'seed': args.seed,
                'max_pairs': 0, 'allow_legacy_artifacts': False}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Head training configuration does not match the fixed study')
    if (checkpoint['ablate'], checkpoint['n_images'], checkpoint['n_pairs']) != (FLAGS[args.variant], 9181, 31945):
        raise ValueError('Wrong original-trainer ablation or population')


def validate_measured(args, checkpoint, signature, requirements, dimension):
    """Validate actual measured heads without inventing legacy config/provenance."""
    expected = validate_backbone(args, signature, requirements)
    study = requirements['measured_study']
    if checkpoint.get('schema') != MEASURED_SCHEMA:
        raise ValueError('Expected actual simplified-study checkpoint schema')
    if (checkpoint.get('backbone'), checkpoint.get('seed'), checkpoint.get('variant'), checkpoint.get('ablate'),
        checkpoint.get('n_images'), checkpoint.get('n_pairs'), checkpoint.get('D')) != (
            args.backbone, args.seed, args.variant, FLAGS[args.variant], 9181, 31945, dimension):
        raise ValueError('Wrong measured checkpoint identity, ablation, population or dimension')
    if dimension != (768 if args.backbone == 'siglip' else 512):
        raise ValueError('Wrong embedding dimension for backbone')
    settings = {'epochs': 10, 'batch_size': 64, 'weight_decay': .0001, 'n_train_images': 9181,
                'n_train_pairs': 31945, 'batches_per_epoch': 500, 'tail_batch_pairs': 9,
                'optimizer': 'AdamW', 'precision': 'float32', 'lr': .002, 'kappa': expected['kappa']}
    if checkpoint.get('settings') != settings:
        raise ValueError('Measured head does not contain the exact fixed training settings')
    provenance = checkpoint['provenance']
    for key, value in {'backbone': args.backbone, 'seed': args.seed, 'variant': args.variant,
                       'protocol_sha256': study['protocol_sha256'], 'freeze_sha256': study['freeze_sha256']}.items():
        if provenance.get(key) != value:
            raise ValueError('Measured provenance mismatch: ' + key)
    if provenance.get('cache_signature') != signature or signature != study['cache_signatures'][args.backbone]:
        raise ValueError('Measured head/cache signature mismatch')
    for key, expected_hashes in [('input_files', study['input_sha256'][args.backbone]),
                                 ('common_input_files', study['common_input_sha256'])]:
        actual = provenance.get(key, {})
        if set(actual) != set(expected_hashes) or any(actual[name].get('sha256') != sha for name, sha in expected_hashes.items()):
            raise ValueError('Measured input-file hash binding mismatch: ' + key)
    if object_sha(provenance.get('train_ids')) != study['train_ids_object_sha256']:
        raise ValueError('Measured training IDs/order mismatch')
    if provenance.get('initial_pairs_sha256') != study['initial_pairs_sha256'][args.backbone]:
        raise ValueError('Measured initial training-pair binding mismatch')
    orders = list(pair_order_hashes(args.seed))
    if provenance.get('expected_pair_order_sha256') != orders:
        raise ValueError('Measured minibatch-order hashes mismatch')
    history = checkpoint.get('history', [])
    if len(history) != 10:
        raise ValueError('Ten completed training epochs are required')
    for index, row in enumerate(history):
        if (row.get('epoch'), row.get('batches'), row.get('pairs'), row.get('tail_batch_pairs'), row.get('pair_order_sha256')) != (
                index+1, 500, 31945, 9, orders[index]):
            raise ValueError('Measured epoch/batch/tail history mismatch')
        if not math.isfinite(row.get('mean_batch_loss', float('nan'))):
            raise ValueError('Nonfinite measured training loss')
    for name in ('initial_state_sha256', 'text_embeddings_sha256', 'shared_text_artifact_sha256'):
        value = provenance.get(name, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing measured provenance digest: ' + name)


def parameter_records(state, torch):
    if not state:
        raise ValueError('Empty measured parameter state')
    records = {}
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('Measured state must contain finite float32 tensors: ' + name)
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        records[name] = {'dtype': str(value.dtype), 'shape': list(value.shape), 'bytes_sha256': hashlib.sha256(raw).hexdigest()}
    return records


def validate_parameter_manifest(args, records, requirements, manifest):
    study = requirements['measured_study']
    if (manifest.get('schema'), manifest.get('status'), manifest.get('count')) != (
            'pare.path-only-head-projection.v1', 'all_75_heads_parameter_bytes_verified', 75):
        raise ValueError('Complete audited measured-parameter manifest required')
    if any(manifest.get(k) != study[k] for k in ('protocol_sha256', 'freeze_sha256')):
        raise ValueError('Measured parameter manifest belongs to another study')
    tags = ('vitb32', 'vitb16', 'remoteclip', 'siglip', 'georsclip')
    expected = {f'{t}/seed_{s}/{v}/head.pt' for t in tags for s in range(5) for v in FLAGS}
    if set(manifest.get('files', {})) != expected:
        raise ValueError('Measured parameter manifest must contain the exact 75-head grid')
    relative = f'{args.backbone}/seed_{args.seed}/{args.variant}/head.pt'
    row = manifest['files'][relative]
    if (row.get('backbone'), row.get('seed'), row.get('variant')) != (args.backbone, args.seed, args.variant):
        raise ValueError('Measured parameter identity label mismatch')
    if not row.get('all_parameter_dtype_shape_bytes_identical') or records != row.get('parameters'):
        raise ValueError('State_dict dtype/shape/bytes do not match the audited measured head')


def validate_checkpoint(args, checkpoint, cache_view, requirements, original_validator, measured_parameter_validator=None):
    signature = cache_view['provenance']['signature']
    if checkpoint.get('schema') == MEASURED_SCHEMA:
        validate_measured(args, checkpoint, signature, requirements, cache_view['D'])
        if measured_parameter_validator is None:
            raise ValueError('Measured checkpoint requires an audited parameter identity check')
        measured_parameter_validator(checkpoint['state_dict'])
        return 'actual_measured_simplified_study'
    if checkpoint.get('schema') is not None:
        raise ValueError('Unsupported checkpoint schema')
    original_validator(checkpoint, cache_view)  # Never weaken/bypass old validation.
    validate_original_trainer(args, checkpoint, signature, requirements)
    return 'original_train_saqre_reproduction'
