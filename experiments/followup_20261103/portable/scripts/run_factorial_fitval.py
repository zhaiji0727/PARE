#!/usr/bin/env python3
"""Fixed, post-hoc exploratory fit/validation experiment; no test-set inputs.

One invocation runs all 14 planned configurations for one planned seed. Existing
complete runs can only be reused after byte-hash validation; interrupted attempts
are retained and require an explicit retry. There is no model-selection operation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from protocol_adapter import validate_reproduction_protocol

PROTOCOL_SHA256 = '42789ee8793b7c5bd59bf950dd8deefe8bf1c3caa2e9be2d54143630f63bf288'
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT.parent / 'original'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_protocol(path):
    validate_reproduction_protocol(path, PROTOCOL_SHA256, ORIGINAL_ROOT)
    plan = json.loads(Path(path).read_text())
    if sha(Path(path).with_suffix('.md')) != plan['protocol_markdown_sha256']:
        raise ValueError('Protocol markdown differs from its recorded hash')
    expected = {(a, b, c, 1, 1) for a, b, c in itertools.product([0, 1], repeat=3)}
    expected |= {(v, v, v, r, t) for v in (0, 1) for r, t in ((0, 0), (0, 1), (1, 0))}
    tuples = [tuple(item['factors'][k] for k in 'ABCRT') for item in plan['configurations']]
    if len(tuples) != 14 or set(tuples) != expected or plan['seeds'] != [0, 1, 2]:
        raise ValueError('Expected exactly the fixed 14 configurations and three seeds')
    for item, factors in zip(plan['configurations'], tuples):
        a, b, c, r, t = factors
        if item['id'] != f'ABC{a}{b}{c}_R{r}T{t}':
            raise ValueError('Configuration ID/factor disagreement')
    logical = {(x['configuration_id'], x['seed']) for x in plan['logical_run_order']}
    expected_runs = {(x['id'], seed) for x in plan['configurations'] for seed in plan['seeds']}
    if len(plan['logical_run_order']) != 42 or logical != expected_runs:
        raise ValueError('Protocol does not account for exactly all 42 runs')
    return plan


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def train_one(head, cache, pairs, cap2vec, device, seed, settings, expected_orders, progress):
    import torch
    import torch.nn.functional as F
    from batch_scoring import score_matrix
    from train_saqre import move_docs
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    order = list(range(len(pairs)))
    rng = random.Random(seed)
    history = []
    start = time.monotonic()
    for epoch in range(settings['epochs']):
        rng.shuffle(order)
        digest = object_sha(order)
        if digest != expected_orders[epoch]:
            raise RuntimeError('Within-seed pair shuffle diverged')
        head.train()
        total_loss, batches = 0., 0
        for offset in range(0, len(order), settings['batch_size']):
            selected = [pairs[index] for index in order[offset:offset + settings['batch_size']]]
            rows = torch.tensor([row for row, _ in selected])
            q = torch.stack([cap2vec[caption] for _, caption in selected]).to(device)
            scores = score_matrix(head, q, move_docs(cache, rows, device))
            loss = F.cross_entropy(scores / settings['kappa'], torch.arange(len(selected), device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if batches != settings['batches_per_epoch'] or len(selected) != settings['tail_batch_pairs']:
            raise RuntimeError('Training batch/tail count mismatch')
        row = {'epoch': epoch + 1, 'batches': batches, 'pairs': len(pairs),
               'tail_batch_pairs': len(selected), 'pair_order_sha256': digest,
               'mean_batch_loss': total_loss / batches, 'elapsed_seconds': time.monotonic() - start}
        history.append(row)
        progress(history)
        print(json.dumps({'epoch_progress': row}), flush=True)
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise FloatingPointError('Non-finite final head')
    return history


def validate_one(head, cache, gallery, queries, cap2vec, device):
    import torch
    from batch_scoring import score_matrix
    from train_saqre import move_docs
    head.eval()
    gallery_index = {row: index for index, row in enumerate(gallery)}
    gold = [gallery_index[row] for row, _ in queries]
    query_vectors = torch.stack([cap2vec[caption] for _, caption in queries]).to(device)
    matrix = torch.empty((len(queries), len(gallery)), dtype=torch.float32)
    ranks = []
    with torch.no_grad():
        for start in range(0, len(queries), 16):
            end = min(start + 16, len(queries))
            scores = torch.empty((end - start, len(gallery)), device=device)
            for gs in range(0, len(gallery), 1024):
                ge = min(gs + 1024, len(gallery))
                docs = move_docs(cache, torch.tensor(gallery[gs:ge]), device)
                scores[:, gs:ge] = score_matrix(head, query_vectors[start:end], docs)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Non-finite validation score')
            ordered = torch.argsort(scores, dim=1, descending=True)
            targets = torch.tensor(gold[start:end], device=device).unsqueeze(1)
            ranks.extend((ordered == targets).nonzero(as_tuple=True)[1].add(1).tolist())
            matrix[start:end] = scores.cpu()
    if len(ranks) != len(queries) or not all(1 <= rank <= len(gallery) for rank in ranks):
        raise RuntimeError('Invalid validation ranks')
    count = len(ranks)
    metrics = {f'R@{k}': 100 * sum(rank <= k for rank in ranks) / count for k in (1, 5, 10)}
    metrics['MRR@10'] = 100 * sum(1 / rank if rank <= 10 else 0 for rank in ranks) / count
    return matrix, ranks, metrics, gold


def main():
    global PROTOCOL_SHA256
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--expected-protocol-sha256', default=PROTOCOL_SHA256,
                    help='explicit digest emitted by prepare_reproduction.py')
    ap.add_argument('--base-code', required=True, type=Path)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--captions', required=True, type=Path)
    ap.add_argument('--split-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', required=True, type=int, choices=[0, 1, 2])
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--retry-incomplete', action='store_true')
    args = ap.parse_args()
    PROTOCOL_SHA256 = args.expected_protocol_sha256
    plan = validate_protocol(args.protocol)
    sys.path[:0] = [str(args.base_code.resolve()), str(ROOT / 'code')]
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import open_clip
    import torch
    import torch.nn.functional as F
    from artifacts import validate_cache, resolve_pretrained_path
    from factorial_head import FactorialHead, Factors
    from runtime import create_backbone, get_device
    from train_saqre import load_captions
    bindings = plan['input_bindings']
    source_paths = {f'base_code/{p.name}': p for p in args.base_code.glob('*.py')}
    source_paths.update({f'code/{p.name}': p for p in (ROOT / 'code').glob('*.py')})
    source_paths['scripts/run_factorial_fitval.py'] = Path(__file__)
    source_paths['scripts/protocol_adapter.py'] = Path(__file__).with_name('protocol_adapter.py')
    source_sha = {name: sha(path) for name, path in source_paths.items()}
    for name, digest in bindings['frozen_source_sha256'].items():
        if source_sha.get(f'base_code/{name}') != digest:
            raise ValueError(f'Frozen base source mismatch: {name}')
    input_paths = {'training_cache': args.cache, 'training_captions': args.captions,
                   **{name: args.split_dir / name for name in bindings['split_file_sha256']},
                   'original_split_manifest': args.split_dir / 'manifest.json'}
    input_sha = {name: sha(path) for name, path in input_paths.items()}
    expected_inputs = {'training_cache': bindings['training_cache_sha256'],
                       'training_captions': bindings['clean_training_captions_sha256'],
                       **bindings['split_file_sha256'],
                       'original_split_manifest': bindings['original_split_manifest_sha256']}
    if input_sha != expected_inputs:
        raise ValueError('Input files do not match the fixed original training/split hashes')
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    model_name, pretrained = plan['backbone']['model'], plan['backbone']['pretrained']
    signature = validate_cache(cache, model_name, pretrained)
    if signature['primitive_mode'] != 'physics':
        raise ValueError('Only the original physical training cache is allowed')
    if signature['pretrained_weight_sha256'] != plan['backbone']['pretrained_weight_sha256']:
        raise ValueError('Backbone weights mismatch')
    input_paths['pretrained_weight'] = resolve_pretrained_path(model_name, pretrained)
    input_sha['pretrained_weight'] = signature['pretrained_weight_sha256']
    ids = cache['ids']
    fit = json.loads((args.split_dir / 'fit_ids.json').read_text())
    validation = json.loads((args.split_dir / 'val_ids.json').read_text())
    train = json.loads((args.split_dir / 'train_ids.json').read_text())
    if len(ids) != 9181 or len(set(ids)) != 9181 or set(ids) != set(train) or len(train) != 9181:
        raise ValueError('Expected exactly the fixed 9181-image training cache')
    if len(fit) != 8263 or len(set(fit)) != 8263 or len(validation) != 918 or len(set(validation)) != 918:
        raise ValueError('Incorrect fit/validation cardinalities')
    if set(fit) & set(validation) or set(fit) | set(validation) != set(ids):
        raise ValueError('Fit/validation do not disjointly partition the training cache')
    if cache['D'] != 512:
        raise ValueError('This protocol is restricted to ViT-B/32')
    index = {name: row for row, name in enumerate(ids)}
    fit_rows = {index[name] for name in fit}
    val_rows = {index[name] for name in validation}
    caps = load_captions(args.captions)
    pairs, queries = [], []
    for name, captions in caps.items():
        if name not in index:
            continue
        row = index[name]
        if row in fit_rows:
            pairs.extend((row, caption) for caption in captions)
        elif row in val_rows:
            queries.append((row, captions[0]))
    if len(pairs) != 28759 or {row for row, _ in pairs} != fit_rows:
        raise ValueError('Expected every caption pair from all 8263 fit images')
    if len(queries) != 918 or {row for row, _ in queries} != val_rows:
        raise ValueError('Expected one first-caption query from all 918 validation images')
    gallery = sorted(val_rows)
    caption_sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    pair_identity = [[ids[row], caption_sha(caption)] for row, caption in pairs]
    query_identity = [[ids[row], caption_sha(caption)] for row, caption in queries]
    order, expected_orders, rng = list(range(len(pairs))), [], random.Random(args.seed)
    for _ in range(plan['training']['epochs']):
        rng.shuffle(order)
        expected_orders.append(object_sha(order))
    identity = {'protocol_sha256': PROTOCOL_SHA256, 'seed': args.seed, 'source_sha256': source_sha,
                'input_sha256': input_sha, 'initial_pairs_sha256': object_sha(pair_identity),
                'validation_queries_sha256': object_sha(query_identity), 'epoch_order_sha256': expected_orders,
                'configuration_ids': [c['id'] for c in plan['configurations']]}

    def check_immutable():
        if sha(args.protocol) != PROTOCOL_SHA256:
            raise RuntimeError('Protocol changed during execution')
        if sha(args.protocol.with_suffix('.md')) != plan['protocol_markdown_sha256']:
            raise RuntimeError('Protocol markdown changed during execution')
        if any(sha(path) != input_sha[name] for name, path in input_paths.items()):
            raise RuntimeError('Original input changed during execution')
        if any(sha(path) != source_sha[name] for name, path in source_paths.items()):
            raise RuntimeError('Source changed during execution')

    if args.out.exists() and not args.resume:
        raise FileExistsError('Use a new output directory or explicitly --resume')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Resume identity/source/input mismatch')
    else:
        if args.resume:
            raise ValueError('No existing manifest to resume')
        manifest = {'schema': 'pare.factorial-fit-validation.v1', 'created_utc': now(),
                    'status': 'preflight', 'identity': identity, 'runs': {}, 'failure_retry_history': [],
                    'n_fit_images': 8263, 'n_fit_pairs': 28759, 'n_validation_queries': 918,
                    'n_validation_gallery': 918, 'cache_contains_images': 9181,
                    'posthoc_exploratory': True, 'uses_official_test': False,
                    'train_ids': [ids[row] for row in sorted(fit_rows)],
                    'validation_ids': [ids[row] for row in gallery], 'settings': plan['training'],
                    'environment': {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
                                    for name in ['torch', 'open_clip_torch', 'numpy', 'Pillow', 'scipy']}}}
        write_json(manifest_path, manifest)
    if args.preflight_only:
        check_immutable()
        write_json(args.out / f'preflight_{time.time_ns()}.json',
                   {'passed': True, 'identity': identity, 'checked_utc': now()})
        if manifest['status'] == 'preflight':
            manifest['status'] = 'preflight_passed_no_training'
            write_json(manifest_path, manifest)
        print(json.dumps({'preflight_passed': True, 'identity': identity}), flush=True)
        return

    device = get_device()
    if not str(device).startswith('cuda'):
        raise ValueError('Formal execution requires a checked CUDA device')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    attempt_environment = {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
                           for name in ['torch', 'open_clip_torch', 'numpy', 'Pillow', 'scipy']},
                           'cuda': torch.version.cuda, 'device': str(device),
                           'gpu': torch.cuda.get_device_name(device),
                           'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                           'tf32': False, 'deterministic_algorithms': True,
                           'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}
    manifest.setdefault('execution_attempts', []).append({'started_utc': now(), 'environment': attempt_environment})
    manifest['status'] = 'embedding_training_and_validation_captions'
    write_json(manifest_path, manifest)
    try:
        model, _, _ = create_backbone(model_name, pretrained)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(model_name)
        unique = sorted({caption for _, caption in pairs} | {caption for _, caption in queries})
        vectors = []
        with torch.no_grad():
            for start in range(0, len(unique), 256):
                tokens = tokenizer(unique[start:start + 256]).to(device)
                vectors.append(F.normalize(model.encode_text(tokens), dim=-1).cpu())
        vectors = torch.cat(vectors)
        if not torch.isfinite(vectors).all():
            raise FloatingPointError('Non-finite frozen text embeddings')
        cap2vec = dict(zip(unique, vectors))
        embedding_sha = state_sha({'text_embeddings': vectors})
        manifest['text_embeddings_sha256'] = embedding_sha
        del model
        torch.cuda.empty_cache()
        print(json.dumps({'captions_embedded': len(unique), 'sha256': embedding_sha}), flush=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        initial_head = FactorialHead(cache['D'])
        initial_state = {key: value.detach().clone() for key, value in initial_head.state_dict().items()}
        initial_sha = state_sha(initial_state)
        del initial_head
        if manifest.get('initial_state_sha256', initial_sha) != initial_sha:
            raise RuntimeError('Resumed initial parameters differ')
        manifest['initial_state_sha256'] = initial_sha
        for configuration in plan['configurations']:
            name = configuration['id']
            run_dir = args.out / name
            prior = manifest['runs'].get(name)
            if prior and prior['status'] == 'complete':
                if prior['initial_state_sha256'] != initial_sha:
                    raise ValueError('Completed run has different initialization')
                if any(sha(run_dir / file) != digest for file, digest in prior['artifacts_sha256'].items()):
                    raise ValueError('Completed run artifact changed')
                if prior['text_embeddings_sha256'] != embedding_sha:
                    raise ValueError('Completed run used different frozen text embeddings')
                continue
            if run_dir.exists():
                if not args.retry_incomplete:
                    raise FileExistsError('Incomplete run exists; explicit --retry-incomplete retains and retries it')
                archive = args.out / 'attempts' / f'{name}_{time.time_ns()}'
                archive.parent.mkdir(exist_ok=True)
                shutil.move(str(run_dir), archive)
                manifest['failure_retry_history'].append({'configuration': name, 'archived': str(archive.relative_to(args.out)),
                                                         'prior': prior, 'retried_utc': now()})
            run_dir.mkdir()
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            head = FactorialHead(cache['D'], Factors(**configuration['factors']))
            if state_sha(head.state_dict()) != initial_sha:
                raise RuntimeError('Configuration did not receive exactly the shared initialization')
            head = head.to(device)
            run = {'status': 'training', 'started_utc': now(), 'configuration': configuration,
                   'seed': args.seed, 'initial_state_sha256': initial_sha,
                   'execution_environment': attempt_environment,
                   'text_embeddings_sha256': embedding_sha, 'initial_pairs_sha256': identity['initial_pairs_sha256']}
            manifest['runs'][name] = run
            manifest['status'] = 'running'
            manifest['active_configuration'] = name
            write_json(manifest_path, manifest)

            def progress(history):
                write_json(run_dir / 'training_progress.json', {'history': history, 'updated_utc': now()})

            history = train_one(head, cache, pairs, cap2vec, device, args.seed, plan['training'], expected_orders, progress)
            matrix, ranks, metrics, gold = validate_one(head, cache, gallery, queries, cap2vec, device)
            check_immutable()
            checkpoint = {'schema': 'pare.factorial-head.v1', 'state_dict': {key: value.detach().cpu() for key, value in head.state_dict().items()},
                          'D': cache['D'], 'factors': configuration['factors'], 'seed': args.seed,
                          'n_images': 8263, 'n_pairs': 28759, 'history': history,
                          'provenance': {'protocol_sha256': PROTOCOL_SHA256, 'source_sha256': source_sha,
                                         'input_sha256': input_sha, 'cache_signature': signature,
                                         'train_ids': manifest['train_ids'], 'validation_ids': manifest['validation_ids'],
                                         'initial_state_sha256': initial_sha, 'text_embeddings_sha256': embedding_sha,
                                         'initial_pairs_sha256': identity['initial_pairs_sha256']},
                          'settings': plan['training']}
            torch.save(checkpoint, run_dir / 'head.pt')
            torch.save({'scores': matrix, 'query_ids': [item[0] for item in query_identity],
                        'gallery_ids': manifest['validation_ids'], 'query_caption_sha256': [item[1] for item in query_identity],
                        'gold_gallery_index': gold, 'protocol_sha256': PROTOCOL_SHA256}, run_dir / 'validation_scores.pt')
            result = {'configuration': configuration, 'seed': args.seed, 'metrics_percent': metrics,
                      'query_ids': [item[0] for item in query_identity], 'query_caption_sha256': [item[1] for item in query_identity],
                      'gallery_ids': manifest['validation_ids'], 'gold_gallery_index': gold, 'ranks': ranks,
                      'n_queries': len(queries), 'n_gallery': len(gallery),
                      'checkpoint_sha256': sha(run_dir / 'head.pt'), 'protocol_sha256': PROTOCOL_SHA256,
                      'ranking': 'torch.argsort(descending=True) on CUDA; all 918 validation images',
                      'posthoc_exploratory': True, 'official_test_accessed': False}
            write_json(run_dir / 'validation.json', result)
            run.update({'status': 'complete', 'completed_utc': now(), 'metrics_percent': metrics,
                        'epochs': len(history), 'batches': sum(item['batches'] for item in history),
                        'epoch_order_sha256': expected_orders,
                        'artifacts_sha256': {p.name: sha(p) for p in run_dir.iterdir() if p.is_file()}})
            write_json(manifest_path, manifest)
            print(json.dumps({'complete': name, 'seed': args.seed, 'metrics_percent': metrics}), flush=True)
            del head, matrix
        check_immutable()
        if set(manifest['runs']) != set(identity['configuration_ids']) or any(r['status'] != 'complete' for r in manifest['runs'].values()):
            raise RuntimeError('Not all 14 configurations are complete')
        manifest.update({'status': 'complete_all_14_configurations', 'completed_utc': now(), 'active_configuration': None})
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['failure_retry_history'].append({'failed_utc': now(), 'configuration': manifest.get('active_configuration'),
                                                 'type': type(error).__name__, 'message': str(error)})
        write_json(manifest_path, manifest)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
