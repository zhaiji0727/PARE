"""Reproduce a fixed full-training trajectory from explicit, relocatable inputs.

This is a new reproduction executor, not the executor of the archived results.
It reuses the byte-identical frozen numerical training functions. Cache building
and retrieval evaluation are separate stages. No old checkpoint is accepted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checked_json(path, digest):
    require(sha(path) == digest, 'JSON hash differs: ' + str(path))
    return json.loads(Path(path).read_text())


def bind(row, relative_to):
    require(isinstance(row, dict) and isinstance(row.get('path'), str)
            and isinstance(row.get('sha256'), str), 'Explicit path/hash required')
    path = Path(row['path']).expanduser()
    path = (path if path.is_absolute() else relative_to / path).resolve()
    require(sha(path) == row['sha256'], 'Input hash differs: ' + str(path))
    return path


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_contracts(args):
    protocol = checked_json(args.protocol, args.protocol_sha256)
    inputs = checked_json(args.inputs, args.inputs_sha256)
    require(protocol.get('schema') == 'pare.portable-full-training-protocol.v1', 'Protocol schema')
    require(inputs.get('schema') == 'pare.portable-full-training-inputs.v1'
            and inputs.get('status') == 'BOUND', 'Input manifest is unbound')
    require(args.backbone in protocol['backbones'], 'Unknown backbone')
    require(args.variant in protocol['models_by_backbone'][args.backbone], 'Model absent from fixed matrix')
    require(args.seed in protocol['seeds'], 'Seed absent from fixed matrix')
    require(protocol['epochs'] == 50 and protocol['checkpoint_epochs'] == [20, 50]
            and protocol['batch_size'] == 64 and protocol['weight_decay'] == 0.0001,
            'Fixed training budget/recipe differs')
    require(inputs['backbone'] == args.backbone, 'Text/cache manifest backbone differs')
    lock = checked_json(ROOT / 'SOURCE_LOCK.json', protocol['source_lock_sha256'])
    for name, digest in lock['files'].items():
        path = (ROOT / name).resolve()
        require(not Path(name).is_absolute() and ROOT in path.parents, 'Unsafe source path')
        require(sha(path) == digest, 'Numerical/executor source differs: ' + name)
    files = {key: bind(inputs[key], args.inputs.resolve().parent)
             for key in ('cache', 'training_text', 'captions')}
    require(not args.output.exists(), 'Existing output cannot be overwritten or resumed')
    return protocol, inputs, files, lock


def validate_tensors(cache, text, inputs):
    import torch
    from cache_common import objsha
    from train_backend import state_sha
    from data_contract import load_captions, training_index
    d = inputs['dimension']
    require(d == (768 if inputs['backbone'] == 'siglip' else 512), 'Backbone dimension')
    require((cache['D'], cache['Mc'], cache['Mf']) == (d, 6, 8), 'Cache dimensions/slots')
    require(len(cache['ids']) == len(set(cache['ids'])) == 9181
            and objsha(cache['ids']) == inputs['training_ids_sha256'], 'Complete ordered training IDs')
    require('scale_test' not in cache, 'Scale/test cache cannot train the model')
    signature = cache['provenance']['signature']
    require(objsha(signature) == cache['provenance']['signature_sha256']
            == inputs['cache_signature_sha256'], 'Cache extraction identity')
    encoder = signature
    for key in inputs['encoder_signature_path']:
        encoder = encoder[key]
    require(encoder == inputs['encoder_signature'], 'Backbone/weights/preprocessing identity differs')
    shapes = {'scene': (9181, d), 'agg': (9181, d), 'nu': (9181,)}
    for branch, slots in [('c', 6), ('f', 8)]:
        n = cache[branch + '_n']
        require(n.shape == (9181,) and n.dtype in (torch.int32, torch.int64)
                and bool(((n >= 0) & (n <= slots)).all()), 'Invalid region count')
        shapes.update({branch + '_emb': (9181, slots, d), branch + '_mu': (9181, slots),
                       branch + '_t': (9181, slots), branch + '_tf': (9181, slots, 4)})
    for key, shape in shapes.items():
        value = cache[key]
        require(tuple(value.shape) == shape and value.dtype == torch.float32
                and bool(torch.isfinite(value).all()), 'Invalid cache tensor: ' + key)
    require(set(text) == {'embeddings', 'backbone', 'training_text_sha256'}, 'Training-only text schema')
    require(text['backbone'] == inputs['backbone'], 'Text encoder identity')
    q = text['embeddings']
    require(q.shape == (18461, d) and q.dtype == torch.float32
            and bool(torch.isfinite(q).all()), 'Training-only text matrix')
    require(state_sha({'train': q}) == inputs['training_tensor_sha256'], 'Text tensor identity')
    return q


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'inputs'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    p, inputs, files, lock = load_contracts(args)
    if args.device == 'cuda':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(visible and ',' not in visible, 'Select one physical GPU explicitly with CUDA_VISIBLE_DEVICES')
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='3',
                      MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3')
    sys.path.insert(0, str(ROOT / 'runtime'))
    import torch
    from data_contract import training_index, load_captions
    from matched_model_factory import build_matched_model
    from model_spec import parse_model_id
    from train_backend import documents_from_cache, full_region_means, train_epochs, state_sha
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        require(torch.cuda.device_count() == 1, 'Exactly one visible GPU required')
    cache = torch.load(files['cache'], map_location='cpu', weights_only=True)
    text = torch.load(files['training_text'], map_location='cpu', weights_only=True)
    q = validate_tensors(cache, text, inputs)
    data = training_index(cache['ids'], load_captions(files['captions']))
    require(data['pair_order_sha256'] == inputs['pair_order_sha256'], 'Training pair sequence')
    require(data['training_text_sha256'] == text['training_text_sha256']
            == inputs['training_text_sha256'], 'Caption/text row correspondence')
    q = q.to(args.device)
    documents = documents_from_cache(cache, args.device)
    means, counts = full_region_means(documents)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = build_matched_model(cache['D'], args.variant, *means).to(args.device)
    hp = p['hyperparameters'][args.backbone]
    optimizer = torch.optim.AdamW(model.active_parameters(), lr=hp['learning_rate'], weight_decay=p['weight_decay'])
    gpu_name = torch.cuda.get_device_name() if args.device == 'cuda' else None
    metadata = {'schema': 'pare.portable-full-training-replay.v1',
        'backbone': args.backbone, 'variant': args.variant, 'seed': args.seed,
        'dimension': cache['D'], 'epochs': p['epochs'], 'checkpoint_epochs': p['checkpoint_epochs'],
        'train_images': 9181, 'train_pairs': 31945, 'training_texts': 18461,
        'initialization': 'fresh_from_seed', 'descriptor_mean_population': 'all_valid_regions_of_all_9181_training_images',
        'descriptor_mean_counts': counts, 'full_training_crop_mean_c': means[0].tolist(),
        'full_training_crop_mean_f': means[1].tolist(), 'training_pair_order_sha256': data['pair_order_sha256'],
        'training_text_sha256': data['training_text_sha256'], 'initial_adapter_sha256': state_sha(model.adapter.state_dict()),
        'initial_physics_sha256': state_sha(model.physics.state_dict()),
        'model_spec': parse_model_id(args.variant), 'paths_enabled': getattr(model, 'paths_enabled', None),
        'ablate': sorted(model.physics.ablate), 'underlying_joint_variant': model.variant,
        'source_lock_sha256': p['source_lock_sha256'], 'protocol_sha256': args.protocol_sha256,
        'inputs_manifest_sha256': args.inputs_sha256,
        'input_files': {k: {'path': str(v), 'sha256': inputs[k]['sha256']} for k, v in files.items()},
        'reference_experiment_protocols': p['reference_experiment_protocols'],
        'torch': str(torch.__version__), 'gpu_name': gpu_name, 'device': args.device,
        'reference_runtime_matches': str(torch.__version__) == p['reference_runtime']['torch']
            and gpu_name == p['reference_runtime']['gpu_name'],
        'scope': 'New reproducibility run. No retrieval scoring or use of archived checkpoint parameters.',
        'started_utc': datetime.now(timezone.utc).isoformat()}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'run.json', metadata)
    try:
        def save_checkpoint(epoch, state):
            with (args.output / f'epoch_{epoch}.pt').open('xb') as stream:
                torch.save(state, stream)
        def progress(history):
            path = args.output / 'progress.json'
            temporary = args.output / 'progress.json.tmp'
            temporary.write_text(json.dumps({'history': history}, indent=2) + '\n')
            temporary.replace(path)
            print(json.dumps(history[-1]), flush=True)
        train_epochs(model, optimizer, documents, q, data['pairs'], seed=args.seed,
                     epochs=p['epochs'], kappa=hp['kappa'], metadata=metadata,
                     checkpoint_epochs=p['checkpoint_epochs'], checkpoint_callback=save_checkpoint,
                     progress_callback=progress)
        for key, path in files.items():
            require(sha(path) == inputs[key]['sha256'], 'Input changed during training')
        for name, digest in lock['files'].items():
            require(sha(ROOT / name) == digest, 'Source changed during training')
        write(args.output / 'complete.json', {'status': 'completed', 'metadata_schema': metadata['schema'],
            'artifacts': {path.name: sha(path) for path in args.output.iterdir() if path.is_file()},
            'ended_utc': datetime.now(timezone.utc).isoformat(), 'retrieval_scoring_performed': False})
    except BaseException as error:
        write(args.output / 'failed.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
