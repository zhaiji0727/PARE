#!/usr/bin/env python3
"""Frozen two-stage repair experiment. Never consumes the official test split.

train: one seed's complete candidate grid, train6610 and development918 only.
lock-selection: all seeds must be complete; validate every artifact, then commit
one global development-based selection before any internal-check evaluation.
check: score every candidate on internal1653, bound to that immutable selection.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
P = Path('/home/coder/my_workspace/rs_paper/pare_workspace')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def source_files():
    paths = [p for d in ('base_code', 'code', 'scripts', 'tests', 'reference') for p in (ROOT / d).glob('*.py')]
    paths += [ROOT / 'REFERENCE_KERNEL_PROVENANCE.json']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def validate_plan(plan):
    configs = plan['configurations']
    names = [c['id'] for c in configs]
    if len(names) != len(set(names)) or not names or plan['seeds'] != [0, 1, 2]:
        raise ValueError('Unique candidates and exactly three shared seeds required')
    expected = {(a, 0, 1, 1, 'original', r, t) for a in (0, 1)
                for r in ('original', 'constant', 'bounded') for t in ('original', 'bounded')}
    expected |= {(a, a, a, 1, 'original', 'original', 'original') for a in (0, 1)}
    keys = ('A', 'B', 'C', 'T', 'fusion', 'reliability', 'token_mode')
    observed = [tuple(c['config'][k] for k in keys) for c in configs]
    if len(configs) != 14 or set(observed) != expected:
        raise ValueError('Protocol must contain the reviewed 14 configurations exactly')
    train = plan['training']
    expected_train = {'lr': .002, 'kappa': .03, 'epochs': 10, 'batch_size': 64,
                      'weight_decay': .0001, 'optimizer': 'AdamW', 'precision': 'float32', 'n_train_images': 6610}
    if any(train[k] != value for k, value in expected_train.items()):
        raise ValueError('Training deviates from the fixed repair protocol')
    n = train['n_train_pairs']
    if type(n) is not int or n <= 0 or train['batches_per_epoch'] != math.ceil(n / 64) or train['tail_batch_pairs'] != (n - (math.ceil(n / 64) - 1) * 64):
        raise ValueError('Pair/batch/tail counts must be finalized and internally consistent')
    selection = plan['selection']
    eligible = selection['eligible_configuration_ids']
    wanted = {c['id'] for c in configs if c['config']['A'] == 1 and c['config']['B'] == 0 and c['config']['C'] == 1}
    if set(eligible) != wanted or len(eligible) != len(wanted) or set(selection['priority_order']) != wanted or len(selection['priority_order']) != len(wanted):
        raise ValueError('Selection must contain all and only the six fine-preserving repair candidates')
    if selection['primary_metric'] != 'R@10' or selection['secondary_metric'] != 'MRR@10':
        raise ValueError('Unexpected selection metrics')
    controls = plan['selected_control']
    if controls['config_overrides'] != {'T': 0} or controls['expected_runs'] != 3 or controls['seeds'] != [0, 1, 2] or controls['expected_configurations'] != 1:
        raise ValueError('Exactly three selected T0 controls are allowed')
    return names


def check_freeze(args, plan):
    frozen = json.loads(args.freeze.read_text())
    if frozen['protocol_sha256'] != sha(args.protocol) or frozen['protocol_markdown_sha256'] != sha(args.protocol.with_suffix('.md')):
        raise ValueError('Protocol differs from frozen reviewed files')
    if frozen['source_sha256'] != source_files():
        raise ValueError('A source or test changed after freezing')
    if frozen['repair_split_sha256'] != {p.name: sha(p) for p in sorted(args.split_dir.glob('*_ids.json'))}:
        raise ValueError('Repair split files changed after freezing')
    validate_plan(plan)
    for name, digest in plan['input_bindings']['reference_source_sha256'].items():
        if sha(ROOT / 'base_code' / name) != digest:
            raise ValueError('Copied reference source differs from the original frozen baseline')
    return frozen


def validate_populations(original_fit, original_val, original_train, repair_train, dev, check):
    groups = [original_fit, original_val, original_train, repair_train, dev, check]
    if any(len(g) != len(set(g)) for g in groups):
        raise ValueError('Duplicate split IDs')
    if list(map(len, groups)) != [8263, 918, 9181, 6610, 918, 1653]:
        raise ValueError('Incorrect population counts')
    if set(original_fit) & set(original_val) or set(original_fit) | set(original_val) != set(original_train):
        raise ValueError('Original split mismatch')
    expected = sorted(original_fit, key=lambda name: (hashlib.sha256(('pare-repair-internal-check-v1|' + name).encode()).hexdigest(), name))[:1653]
    if set(check) != set(expected) or set(repair_train) != set(original_fit) - set(check) or set(dev) != set(original_val):
        raise ValueError('Repair split is not the prespecified deterministic partition')
    if any(set(a) & set(b) for a, b in ((repair_train, dev), (repair_train, check), (dev, check))):
        raise ValueError('Train/development/internal-check overlap')


def load_inputs(args, plan):
    import torch
    from artifacts import validate_cache, resolve_pretrained_path
    bindings = plan['input_bindings']
    paths = {'training_cache': args.cache, 'training_captions': args.captions}
    expected = {'training_cache': bindings['training_cache_sha256'], 'training_captions': bindings['clean_training_captions_sha256']}
    for prefix, directory, field in [('original', args.original_split_dir, 'original_split_file_sha256'), ('repair', args.split_dir, 'repair_split_file_sha256')]:
        for name, digest in bindings[field].items():
            paths[f'{prefix}/{name}'] = directory / name
            expected[f'{prefix}/{name}'] = digest
    actual = {name: sha(path) for name, path in paths.items()}
    if actual != expected:
        raise ValueError('Input hashes do not match finalized protocol')
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    backbone = plan['backbone']
    signature = validate_cache(cache, backbone['model'], backbone['pretrained'])
    if signature['primitive_mode'] != 'physics' or signature['pretrained_weight_sha256'] != bindings['pretrained_weight_sha256']:
        raise ValueError('Wrong physical cache or pretrained weights')
    paths['pretrained_weight'] = resolve_pretrained_path(backbone['model'], backbone['pretrained'])
    actual['pretrained_weight'] = sha(paths['pretrained_weight'])
    if actual['pretrained_weight'] != bindings['pretrained_weight_sha256']:
        raise ValueError('Pretrained weight file changed')
    read = lambda directory, name: json.loads((directory / name).read_text())
    original_fit, original_val, original_train = [read(args.original_split_dir, n) for n in ('fit_ids.json', 'val_ids.json', 'train_ids.json')]
    repair_train, dev, check = [read(args.split_dir, n) for n in ('train_ids.json', 'development_ids.json', 'internal_check_ids.json')]
    validate_populations(original_fit, original_val, original_train, repair_train, dev, check)
    if len(cache['ids']) != 9181 or set(cache['ids']) != set(original_train) or cache['D'] != 512:
        raise ValueError('Wrong cache population or dimension')
    populations = {'train': repair_train, 'development': dev, 'internal_check': check}
    return cache, signature, populations, paths, actual


def setup_runtime(args):
    if args.gpu is not None:
        if args.gpu not in (0, 1, 2, 3, 4, 5):
            raise ValueError('Only the six allocated GPUs are allowed')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', RS_RAG_MODEL_ROOT=str(args.models))
    sys.path[:0] = [str(ROOT / 'base_code'), str(ROOT / 'code')]
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def environment(torch):
    import importlib.metadata
    return {'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in ['torch', 'open_clip_torch', 'numpy', 'Pillow', 'scipy']},
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'tf32': False, 'deterministic_algorithms': True, 'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}


def pairs_for(cache, populations, captions, stage):
    index = {name: row for row, name in enumerate(cache['ids'])}
    train = set(populations['train'])
    evaluation = set(populations['development' if stage in ('train', 'selected-control') else 'internal_check'])
    pairs, queries = [], []
    for name, strings in captions.items():
        if name not in index:
            continue
        if stage in ('train', 'selected-control') and name in train:
            pairs.extend((index[name], text) for text in strings)
        elif name in evaluation:
            queries.append((index[name], strings[0]))
    gallery = sorted(index[name] for name in evaluation)
    if len(queries) != len(evaluation) or {row for row, _ in queries} != set(gallery):
        raise ValueError('Missing first-caption evaluation queries')
    if stage in ('train', 'selected-control') and {cache['ids'][row] for row, _ in pairs} != train:
        raise ValueError('Training caption population mismatch')
    return pairs, queries, gallery


def embed(plan, pairs, queries, torch):
    import open_clip
    import torch.nn.functional as F
    from runtime import create_backbone
    unique = sorted({text for _, text in pairs} | {text for _, text in queries})
    model, _, _ = create_backbone(plan['backbone']['model'], plan['backbone']['pretrained'])
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(plan['backbone']['model'])
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), 256):
            tokens = tokenizer(unique[start:start + 256]).cuda()
            vectors.append(F.normalize(model.encode_text(tokens), dim=-1).cpu())
    vectors = torch.cat(vectors)
    if not torch.isfinite(vectors).all():
        raise FloatingPointError('Non-finite frozen text embeddings')
    digest = state_sha({'text_embeddings': vectors})
    del model
    torch.cuda.empty_cache()
    return dict(zip(unique, vectors)), digest


def rank_metrics(ranks):
    if not ranks or any(type(rank) is not int or rank < 1 or rank > len(ranks) for rank in ranks):
        raise ValueError('Invalid full-gallery ranks')
    return {**{f'R@{k}': 100 * sum(rank <= k for rank in ranks) / len(ranks) for k in (1, 5, 10)},
            'MRR@10': 100 * sum(1 / rank if rank <= 10 else 0 for rank in ranks) / len(ranks)}


def selection_payload(plan, all_metrics):
    wanted = {(seed, c['id']) for seed in plan['seeds'] for c in plan['configurations']}
    if set(all_metrics) != wanted:
        raise ValueError('Selection requires every prespecified seed/configuration')
    means = {c['id']: {metric: sum(all_metrics[(seed, c['id'])][metric] for seed in plan['seeds']) / len(plan['seeds'])
                      for metric in ('R@1', 'R@5', 'R@10', 'MRR@10')} for c in plan['configurations']}
    selection = plan['selection']
    priority = {name: index for index, name in enumerate(selection['priority_order'])}
    ranked = sorted(selection['eligible_configuration_ids'], key=lambda name: (-means[name][selection['primary_metric']], -means[name][selection['secondary_metric']], priority[name]))
    return {'selected_configuration_id': ranked[0], 'eligible_ordered_by_locked_rule': ranked,
            'all_configuration_development_means': means, 'rule': selection}


def verify_training(args, plan, frozen, replay=True):
    """Read only stage-one artifacts; never touch internal-check outputs."""
    if replay:
        import torch
    metrics, manifests, artifacts = {}, {}, {}
    expected_names = {c['id'] for c in plan['configurations']}
    train_ids = set(json.loads((args.split_dir / 'train_ids.json').read_text()))
    dev_ids = set(json.loads((args.split_dir / 'development_ids.json').read_text()))
    for seed in plan['seeds']:
        folder = args.out / f'seed_{seed}'
        mp = folder / 'manifest.json'
        manifest = json.loads(mp.read_text())
        if manifest['status'] != 'complete_training_and_development' or set(manifest['runs']) != expected_names:
            raise ValueError('All candidates and seeds must complete before selection')
        if manifest['identity'].get('stage') != 'train':
            raise ValueError('Selection cannot consume post-selection controls or check results')
        if manifest['identity']['protocol_sha256'] != sha(args.protocol) or manifest['identity']['freeze_sha256'] != sha(args.freeze):
            raise ValueError('Training uses a different protocol or source freeze')
        if set(manifest['train_ids']) != train_ids or set(manifest['development_ids']) != dev_ids:
            raise ValueError('Training/development split mismatch')
        manifests[str(seed)] = sha(mp)
        for name, run in manifest['runs'].items():
            if run['status'] != 'complete':
                raise ValueError('Incomplete training run')
            expected_config = next(c for c in plan['configurations'] if c['id'] == name)
            if run['configuration'] != expected_config or run['seed'] != seed:
                raise ValueError('Training run configuration/seed mismatch')
            if set(run['artifacts_sha256']) != {'head.pt', 'training_progress.json', 'development_scores.pt', 'development.json'}:
                raise ValueError('Incomplete or unexpected training artifact set')
            directory = folder / name
            if any(sha(directory / file) != digest for file, digest in run['artifacts_sha256'].items()):
                raise ValueError('A training artifact changed')
            result = json.loads((directory / 'development.json').read_text())
            if result['configuration'] != expected_config or result['seed'] != seed or result['freeze_sha256'] != sha(args.freeze):
                raise ValueError('Development configuration/seed/source mismatch')
            if result['checkpoint_sha256'] != sha(directory / 'head.pt') or result['protocol_sha256'] != sha(args.protocol):
                raise ValueError('Development result is not bound to its head/protocol')
            if set(result['query_ids']) != dev_ids or set(result['gallery_ids']) != dev_ids or len(result['query_ids']) != 918 or len(result['gallery_ids']) != 918:
                raise ValueError('Wrong development evaluation population')
            recomputed = rank_metrics(result['ranks'])
            if recomputed != result['metrics_percent'] or recomputed != run['metrics_percent']:
                raise ValueError('Development metric/rank mismatch')
            if replay:
                head = torch.load(directory / 'head.pt', map_location='cpu', weights_only=True)
                if head['n_images'] != 6610 or head['n_pairs'] != plan['training']['n_train_pairs'] or set(head['provenance']['train_ids']) != train_ids or head['config'] != run['configuration']['config'] or head['provenance']['protocol_sha256'] != sha(args.protocol) or head['provenance']['freeze_sha256'] != sha(args.freeze):
                    raise ValueError('Checkpoint has wrong train population/configuration')
                matrix = torch.load(directory / 'development_scores.pt', map_location='cpu', weights_only=True)
                if matrix['query_ids'] != result['query_ids'] or matrix['gallery_ids'] != result['gallery_ids'] or matrix['gold_gallery_index'] != result['gold_gallery_index']:
                    raise ValueError('Score matrix identities mismatch')
                scores = matrix['scores']
                if tuple(scores.shape) != (918, 918) or not torch.isfinite(scores).all():
                    raise ValueError('Invalid development score matrix')
                # Replay on CUDA, matching the recorded original sorting backend.
                order = torch.argsort(scores.cuda(), dim=1, descending=True)
                gold = torch.tensor(result['gold_gallery_index'], device='cuda')[:, None]
                ranks = (order == gold).nonzero(as_tuple=True)[1].add(1).cpu().tolist()
                if ranks != result['ranks']:
                    raise ValueError('Development score/rank replay mismatch')
            metrics[(seed, name)] = recomputed
            artifacts[f'{seed}/{name}'] = run['artifacts_sha256']
    return selection_payload(plan, metrics), manifests, artifacts


def lock_selection(args, plan, frozen):
    path = args.out / 'SELECTION_LOCK.json'
    if any((args.out / f'seed_{seed}' / 'internal_check').exists() for seed in plan['seeds']) and not path.exists():
        raise ValueError('Internal-check artifacts exist before a selection lock')
    payload, manifests, artifacts = verify_training(args, plan, frozen, replay=True)
    body = {'schema': 'pare.repair-selection-lock.v1', 'protocol_sha256': sha(args.protocol), 'freeze_sha256': sha(args.freeze),
            'training_manifest_sha256': manifests, 'all_training_artifacts_sha256': artifacts, **payload}
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: existing[k] for k in body} != body:
            raise ValueError('Existing immutable selection differs')
    else:
        # Exclusive creation: a second worker cannot silently overwrite a lock.
        with path.open('x') as stream:
            json.dump({**body, 'locked_utc': now(), 'internal_check_used_for_selection': False}, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'selection_locked': str(path), 'sha256': sha(path), 'selected': payload['selected_configuration_id']}), flush=True)


def verify_selection(args, plan, frozen):
    path = args.out / 'SELECTION_LOCK.json'
    if not path.exists():
        raise ValueError('No internal-check access before all-seed selection is locked')
    locked = json.loads(path.read_text())
    payload, manifests, artifacts = verify_training(args, plan, frozen, replay=False)
    expected = {'protocol_sha256': sha(args.protocol), 'freeze_sha256': sha(args.freeze),
                'training_manifest_sha256': manifests, 'all_training_artifacts_sha256': artifacts, **payload}
    if any(locked.get(k) != v for k, v in expected.items()) or locked.get('internal_check_used_for_selection') is not False:
        raise ValueError('Selection lock or its underlying artifacts changed')
    return sha(path)



def selected_control_configuration(args, plan):
    locked = json.loads((args.out / 'SELECTION_LOCK.json').read_text())
    selected = next(c for c in plan['configurations'] if c['id'] == locked['selected_configuration_id'])
    config = dict(selected['config'])
    if (config['A'], config['B'], config['C'], config['T']) != (1, 0, 1, 1):
        raise ValueError('Selected control must match the locked fine-preserving candidate')
    if selected['id'].count('_T1_') != 1:
        raise ValueError('Selected ID must contain exactly one _T1_ marker')
    config['T'] = 0
    return {'id': selected['id'].replace('_T1_', '_T0_'), 'config': config,
            'matched_selected_control': True, 'selected_configuration_id': selected['id']}


def verify_controls(args, plan):
    """All three post-selection controls must complete before any check scoring."""
    expected = selected_control_configuration(args, plan)
    binding = {'selection_lock_sha256': sha(args.out / 'SELECTION_LOCK.json'), 'seeds': {}}
    train_ids = set(json.loads((args.split_dir / 'train_ids.json').read_text()))
    dev_ids = set(json.loads((args.split_dir / 'development_ids.json').read_text()))
    for seed in plan['seeds']:
        folder = args.out / f'seed_{seed}' / 'selected_control'
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['status'] != 'complete_selected_control' or set(manifest['runs']) != {expected['id']}:
            raise ValueError('All three matched controls must finish before the internal check')
        identity = manifest['identity']
        if identity['protocol_sha256'] != sha(args.protocol) or identity['freeze_sha256'] != sha(args.freeze) or identity['selection_lock_sha256'] != binding['selection_lock_sha256']:
            raise ValueError('Matched control has stale protocol/source/selection')
        run = manifest['runs'][expected['id']]
        if run['status'] != 'complete' or run['configuration'] != expected or run['seed'] != seed or identity.get('stage') != 'selected-control':
            raise ValueError('Matched control identity mismatch')
        if set(manifest['train_ids']) != train_ids or set(manifest['development_ids']) != dev_ids or set(run['artifacts_sha256']) != {'head.pt', 'training_progress.json', 'development_scores.pt', 'development.json'}:
            raise ValueError('Matched control population/artifact set mismatch')
        directory = folder / expected['id']
        if any(sha(directory / f) != h for f, h in run['artifacts_sha256'].items()):
            raise ValueError('Matched control artifact changed')
        result = json.loads((directory / 'development.json').read_text())
        if result['checkpoint_sha256'] != sha(directory / 'head.pt') or result['metrics_percent'] != rank_metrics(result['ranks']):
            raise ValueError('Matched control development identity/metrics mismatch')
        if result['configuration'] != expected or result['seed'] != seed or result['protocol_sha256'] != sha(args.protocol) or result['freeze_sha256'] != sha(args.freeze):
            raise ValueError('Matched control result provenance mismatch')
        if len(result['query_ids']) != 918 or len(result['gallery_ids']) != 918 or set(result['query_ids']) != dev_ids or set(result['gallery_ids']) != dev_ids:
            raise ValueError('Matched control result has wrong development population')
        binding['seeds'][str(seed)] = {'manifest_sha256': sha(folder / 'manifest.json'), 'artifacts_sha256': run['artifacts_sha256']}
    return binding


def save_evaluation(folder, label, headsha, configuration, seed, cache, gallery, queries, matrix, ranks, metrics, gold, protocolsha, freeze_sha, torch):
    query_ids = [cache['ids'][row] for row, _ in queries]
    query_hashes = [hashlib.sha256(text.encode()).hexdigest() for _, text in queries]
    gallery_ids = [cache['ids'][row] for row in gallery]
    identity = {'query_ids': query_ids, 'query_caption_sha256': query_hashes, 'gallery_ids': gallery_ids,
                'gold_gallery_index': gold, 'protocol_sha256': protocolsha, 'freeze_sha256': freeze_sha}
    torch.save({'scores': matrix, **identity}, folder / f'{label}_scores.pt')
    write_json(folder / f'{label}.json', {**identity, 'configuration': configuration, 'seed': seed,
               'ranks': ranks, 'metrics_percent': metrics, 'n_queries': len(queries), 'n_gallery': len(gallery),
               'checkpoint_sha256': headsha, 'posthoc_exploratory': True, 'official_test_accessed': False,
               'ranking': 'torch.argsort descending on CUDA, full within-split gallery'})


def run_stage(args, plan, frozen, torch):
    from train_saqre import load_captions
    from repair_head import RepairHead, RepairConfig
    from reference_kernels import train_one, validate_one
    from saqre_phys import SAQREHead
    if args.seed not in plan['seeds']:
        raise ValueError('Prespecified seed required')
    if args.command == 'train' and (args.out / 'SELECTION_LOCK.json').exists():
        raise ValueError('Stage-one artifacts are immutable after global selection')
    if args.command == 'selected-control' and any((args.out / f'seed_{seed}' / 'internal_check').exists() for seed in plan['seeds']):
        raise ValueError('Matched controls are immutable once internal checking begins')
    if not torch.cuda.is_available():
        raise ValueError('Formal experiment and rank replay require CUDA')
    selection_sha = verify_selection(args, plan, frozen) if args.command in ('selected-control', 'check') else None
    controls_sha = verify_controls(args, plan) if args.command == 'check' else None
    configurations = list(plan['configurations'])
    if args.command == 'selected-control':
        configurations = [selected_control_configuration(args, plan)]
    elif args.command == 'check':
        configurations.append(selected_control_configuration(args, plan))
    cache, signature, populations, input_paths, input_sha = load_inputs(args, plan)
    # Reading split IDs above does not evaluate any internal sample. Caption
    # selection and encoding below are stage-restricted by pairs_for().
    pairs, queries, gallery = pairs_for(cache, populations, load_captions(args.captions), args.command)
    if args.command in ('train', 'selected-control') and len(pairs) != plan['training']['n_train_pairs']:
        raise ValueError('Pair count differs from the frozen protocol')
    identity = {'protocol_sha256': sha(args.protocol), 'freeze_sha256': sha(args.freeze), 'input_sha256': input_sha,
                'source_sha256': frozen['source_sha256'], 'seed': args.seed, 'stage': args.command,
                'selection_lock_sha256': selection_sha, 'selected_control_artifacts_sha256': controls_sha}
    folder = args.out / f'seed_{args.seed}'
    if args.command == 'check':
        folder = folder / 'internal_check'
    elif args.command == 'selected-control':
        folder = folder / 'selected_control'
    if folder.exists() and not args.resume:
        raise FileExistsError('Use a new output or explicit --resume')
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mp = folder / 'manifest.json'
    if mp.exists():
        manifest = json.loads(mp.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Resume identity/source/input mismatch')
    else:
        manifest = {'schema': 'pare.repair-stage.v1', 'identity': identity, 'status': 'preflight', 'runs': {},
                    'failure_retry_history': [], 'created_utc': now(), 'train_ids': sorted(populations['train']),
                    'development_ids': sorted(populations['development']), 'internal_check_ids': sorted(populations['internal_check']),
                    'n_train_images': 6610, 'n_train_pairs': plan['training']['n_train_pairs'], 'settings': plan['training'],
                    'posthoc_exploratory': True, 'official_test_accessed': False}
        write_json(mp, manifest)
    def immutable():
        check_freeze(args, plan)
        if any(sha(path) != input_sha[key] for key, path in input_paths.items()):
            raise RuntimeError('An original input changed during execution')
        if selection_sha is not None and sha(args.out / 'SELECTION_LOCK.json') != selection_sha:
            raise RuntimeError('Selection lock changed during internal check')
    if args.preflight_only:
        immutable()
        write_json(folder / f'preflight_{time.time_ns()}.json', {'passed': True, 'identity': identity, 'n_pairs': len(pairs), 'n_queries': len(queries), 'n_gallery': len(gallery)})
        print(json.dumps({'preflight_passed': True, 'seed': args.seed, 'stage': args.command, 'n_pairs': len(pairs), 'n_queries': len(queries)}), flush=True)
        return
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cap2vec, embeddingsha = embed(plan, pairs, queries, torch)
        env = environment(torch)
        manifest.setdefault('execution_attempts', []).append({'started_utc': now(), 'environment': env})
        manifest['text_embeddings_sha256'] = embeddingsha
        order, expected_orders, rng = list(range(len(pairs))), [], random.Random(args.seed)
        for _ in range(plan['training']['epochs']):
            rng.shuffle(order)
            expected_orders.append(objsha(order))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        base = SAQREHead(512)
        initial_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        initial_sha = state_sha(initial_state)
        del base
        manifest['base_initial_state_sha256'] = initial_sha
        manifest['initial_pairs_sha256'] = objsha([[cache['ids'][row], hashlib.sha256(text.encode()).hexdigest()] for row, text in pairs])
        for configuration in configurations:
            name = configuration['id']
            directory = folder / name
            prior = manifest['runs'].get(name)
            if prior and prior['status'] == 'complete':
                if prior['text_embeddings_sha256'] != embeddingsha or any(sha(directory / f) != h for f, h in prior['artifacts_sha256'].items()):
                    raise ValueError('Completed artifact or embeddings changed')
                continue
            if directory.exists():
                if not args.retry_incomplete:
                    raise FileExistsError('Incomplete run requires explicit retry; old attempt retained')
                archive = folder / 'attempts' / f'{name}_{time.time_ns()}'
                archive.parent.mkdir(exist_ok=True)
                shutil.move(str(directory), archive)
                manifest['failure_retry_history'].append({'configuration': name, 'archived': str(archive.relative_to(folder)), 'prior': prior})
            directory.mkdir()
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            head = RepairHead(512, RepairConfig(**configuration['config']))
            head.load_base_state_dict(initial_state)
            if state_sha({k: head.state_dict()[k] for k in initial_state}) != initial_sha:
                raise RuntimeError('Common base initialization changed')
            head = head.cuda()
            run = {'status': 'running', 'configuration': configuration, 'seed': args.seed,
                   'started_utc': now(), 'base_initial_state_sha256': initial_sha,
                   'full_initial_state_sha256': state_sha(head.state_dict()), 'text_embeddings_sha256': embeddingsha,
                   'execution_environment': env}
            manifest['runs'][name] = run
            manifest.update(status='running', active_configuration=name)
            write_json(mp, manifest)
            if args.command in ('train', 'selected-control'):
                def progress(history):
                    write_json(directory / 'training_progress.json', {'history': history, 'updated_utc': now()})
                history = train_one(head, cache, pairs, cap2vec, 'cuda', args.seed, plan['training'], expected_orders, progress)
                checkpoint = {'schema': 'pare.repair-head.v1', 'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                              'D': 512, 'config': configuration['config'], 'seed': args.seed, 'n_images': 6610, 'n_pairs': len(pairs),
                              'history': history, 'settings': plan['training'],
                              'provenance': {**identity, 'train_ids': sorted(populations['train']), 'development_ids': sorted(populations['development']),
                                             'excluded_internal_check_ids': sorted(populations['internal_check']), 'cache_signature': signature,
                                             'base_initial_state_sha256': initial_sha, 'full_initial_state_sha256': run['full_initial_state_sha256'],
                                             'text_embeddings_sha256': embeddingsha, 'initial_pairs_sha256': manifest['initial_pairs_sha256']}}
                torch.save(checkpoint, directory / 'head.pt')
                label, headpath = 'development', directory / 'head.pt'
            else:
                headpath = args.out / f'seed_{args.seed}' / name / 'head.pt'
                if configuration.get('matched_selected_control'):
                    headpath = args.out / f'seed_{args.seed}' / 'selected_control' / name / 'head.pt'
                saved = torch.load(headpath, map_location='cpu', weights_only=True)
                if saved['config'] != configuration['config'] or saved['seed'] != args.seed or saved['n_images'] != 6610:
                    raise ValueError('Internal check attempted with wrong head')
                head.load_state_dict(saved['state_dict'], strict=True)
                label = 'internal_check'
                run['checkpoint_sha256'] = sha(headpath)
            matrix, ranks, metrics, gold = validate_one(head, cache, gallery, queries, cap2vec, 'cuda')
            immutable()
            if args.command == 'check' and sha(headpath) != run['checkpoint_sha256']:
                raise RuntimeError('Loaded checkpoint changed during evaluation')
            save_evaluation(directory, label, sha(headpath), configuration, args.seed, cache, gallery, queries, matrix, ranks, metrics, gold, sha(args.protocol), sha(args.freeze), torch)
            run.update(status='complete', completed_utc=now(), metrics_percent=metrics,
                       artifacts_sha256={p.name: sha(p) for p in directory.iterdir() if p.is_file()})
            write_json(mp, manifest)
            print(json.dumps({'complete': name, 'seed': args.seed, 'stage': args.command, 'metrics_percent': metrics}), flush=True)
            del head, matrix
        immutable()
        if set(manifest['runs']) != {c['id'] for c in configurations} or any(r['status'] != 'complete' for r in manifest['runs'].values()):
            raise RuntimeError('Prespecified grid not complete')
        manifest.update(status={'train': 'complete_training_and_development', 'selected-control': 'complete_selected_control', 'check': 'complete_internal_check'}[args.command], active_configuration=None, completed_utc=now())
        write_json(mp, manifest)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['failure_retry_history'].append({'failed_utc': now(), 'configuration': manifest.get('active_configuration'), 'type': type(error).__name__, 'message': str(error)})
        write_json(mp, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'train', 'lock-selection', 'selected-control', 'check'])
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocol.json')
    parser.add_argument('--freeze', type=Path, default=ROOT / 'REPAIR_SOURCE_FROZEN.json')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'plans/splits')
    parser.add_argument('--original-split-dir', type=Path, default=P / 'data/ret_clean_20261007')
    parser.add_argument('--cache', type=Path, default=P / 'formal_experiments_20261103/results/vitb32/cache_physics_train.pt')
    parser.add_argument('--captions', type=Path, default=P / 'data/ret_clean_20261007/Caption_train.csv')
    parser.add_argument('--models', type=Path, default=P / 'models')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--acknowledge-reviewed-source', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text())
    validate_plan(plan)
    if args.command == 'freeze':
        if not args.acknowledge_reviewed_source or args.freeze.exists() or args.out.exists():
            raise ValueError('Freeze requires reviewed-source acknowledgement, a new freeze file, and no prior results')
        write_json(args.freeze, {'schema': 'pare.repair-source-freeze.v1', 'frozen_utc': now(), 'protocol_sha256': sha(args.protocol),
                   'protocol_markdown_sha256': sha(args.protocol.with_suffix('.md')), 'source_sha256': source_files(),
                   'repair_split_sha256': {p.name: sha(p) for p in sorted(args.split_dir.glob('*_ids.json'))}})
        print(json.dumps({'frozen': str(args.freeze), 'sha256': sha(args.freeze)}))
        return
    frozen = check_freeze(args, plan)
    torch = setup_runtime(args)
    if args.command == 'lock-selection':
        lock_selection(args, plan, frozen)
    else:
        run_stage(args, plan, frozen, torch)


if __name__ == '__main__':
    main()
