#!/usr/bin/env python3
"""Post-lock, development-only train-mean-pi intervention on all 42 T1 heads.

External to the frozen experiment. This script never trains, selects a head,
encodes internal-check text, or writes within the training-results directory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_runner(workspace):
    path = workspace / 'scripts/run_repair.py'
    spec = importlib.util.spec_from_file_location('constant_probe_frozen_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_own_source(path):
    data = json.loads(path.read_text())
    here = Path(__file__).resolve().parent
    if data.get('schema') != 'pare.constant-token-probe-source.v1':
        raise ValueError('An explicit external-probe source manifest is required')
    for name, expected in data['file_sha256'].items():
        if Path(name).name != name or sha(here / name) != expected:
            raise ValueError('Probe source changed after its independent freeze')
    if data['file_sha256'].get(Path(__file__).name) != sha(__file__):
        raise ValueError('Executing probe is not pinned')
    return data


def require_population(rows, expected, cache):
    ids = [cache['ids'][r] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Wrong or repeated population in token means/diagnostics')


def branch_statistics(head, cache, rows, branch, device='cuda', chunk=256):
    """Float32 model pi; float64 accumulation; equal weight per valid crop."""
    import torch
    prefix = {'coarse': 'c', 'fine': 'f'}[branch]
    total = torch.zeros(3, dtype=torch.float64)
    occupancy = torch.zeros(3, dtype=torch.int64)
    entropy_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            idx = torch.tensor(rows[start:start + chunk])
            tf = cache[prefix + '_tf'][idx].to(device)
            if branch == 'coarse' and not head.config.C:
                tf = tf.clone()
                tf[..., 3] = 0.0
            # Persistent caches store counts, not the masks built by move_docs.
            slots = cache['Mc' if branch == 'coarse' else 'Mf']
            counts = cache[prefix + '_n'][idx]
            if tf.shape[1] != slots or (counts < 0).any() or (counts > slots).any():
                raise ValueError('Invalid persistent cache crop counts/capacity')
            mask = (torch.arange(slots)[None, :] < counts[:, None]).to(device)
            pi = torch.softmax(head.h_psi(tf), dim=-1)[mask].double().cpu()
            if not torch.isfinite(pi).all():
                raise FloatingPointError('Non-finite prototype probabilities')
            total += pi.sum(0)
            count += len(pi)
            entropy_sum += float(-(pi * pi.clamp_min(1e-300).log()).sum())
            occupancy += torch.bincount(pi.argmax(-1), minlength=3)
    if count == 0:
        raise ValueError('Cannot define a training branch mean without valid crops')
    mean = total / count
    if abs(float(mean.sum()) - 1.0) > 1e-6:
        raise ValueError('Invalid mean prototype probability')
    return {'branch': branch, 'n_images': len(rows), 'n_valid_crops': count,
            'mean_pi_float64': mean.tolist(), 'applied_mean_pi_float32': mean.float().tolist(),
            'mean_crop_entropy_nats': entropy_sum / count,
            'entropy_of_mean_pi_nats': float(-(mean * mean.clamp_min(1e-300).log()).sum()),
            'argmax_prototype_counts': occupancy.tolist(),
            'argmax_prototype_fractions': (occupancy.double() / count).tolist(),
            'equal_weight_per_valid_crop': True,
            'branch_used_in_fusion': bool(head.config.A) if branch == 'fine' else True}


def make_probe_class(base_class):
    """Override only augmentation; retain the frozen fuse and attention methods."""
    class ConstantTokenHead(base_class):
        def set_training_means(self, means, coarse_slots, fine_slots):
            if not self.config.T:
                raise ValueError('Only the 42 original T1 candidates are probed')
            if (coarse_slots, fine_slots) != (6, 8):
                raise ValueError('This cache-bound probe requires distinct Mc=6, Mf=8')
            self._constant_pi_by_slots = {6: means['coarse'], 8: means['fine']}

        def _augment_b(self, emb, tf):
            import torch
            import torch.nn.functional as F
            if emb.ndim != 3 or tf.shape != (*emb.shape[:-1], 4):
                raise ValueError('Unexpected augmentation layout')
            if emb.shape[1] not in self._constant_pi_by_slots:
                raise ValueError('Cannot identify the cache-bound branch')
            mean = torch.as_tensor(self._constant_pi_by_slots[emb.shape[1]],
                                   dtype=emb.dtype, device=emb.device)
            pi = mean.expand(*emb.shape[:-1], 3)
            delta = self.type_scale * (pi @ self.U)
            if self.config.token_mode == 'bounded':
                delta = delta / (1 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                                 / self.TOKEN_DELTA_CAP)
            return F.normalize(emb + delta, dim=-1), pi
    return ConstantTokenHead


def check_identity(result, cache, queries, gallery, gold):
    if result['query_ids'] != [cache['ids'][r] for r, _ in queries]:
        raise ValueError('Development query order changed')
    if result['gallery_ids'] != [cache['ids'][r] for r in gallery]:
        raise ValueError('Development gallery order changed')
    if result['query_caption_sha256'] != [hashlib.sha256(t.encode()).hexdigest() for _, t in queries]:
        raise ValueError('Development query caption identities changed')
    if result['gold_gallery_index'] != gold:
        raise ValueError('Development positive identities changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--captions', type=Path, required=True)
    parser.add_argument('--original-split-dir', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--probe-source-manifest', type=Path,
                        default=Path(__file__).with_name('PROBE_SOURCE_FROZEN.json'))
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.results = args.results.resolve()
    args.output = args.output.resolve()
    if args.output == args.results or args.results in args.output.parents or args.output.exists():
        raise ValueError('Probe output must be a new directory outside training results')
    own_source = check_own_source(args.probe_source_manifest)
    if not (args.results / 'SELECTION_LOCK.json').is_file():
        raise ValueError('No probe execution before a global selection lock')
    runner = load_runner(args.workspace)
    args.protocol = args.workspace / 'protocol.json'
    args.freeze = args.workspace / 'REPAIR_SOURCE_FROZEN.json'
    args.split_dir = args.workspace / 'plans/splits'
    args.out = args.results  # only passed to read-only validation helpers
    if own_source.get('scientific_protocol_sha256') != sha(args.protocol):
        raise ValueError('Probe source manifest targets a different scientific protocol')
    plan = json.loads(args.protocol.read_text())
    frozen = runner.check_freeze(args, plan)
    torch = runner.setup_runtime(args)
    if not torch.cuda.is_available():
        raise ValueError('Formal rank replay and probe require the CUDA backend')
    selection_sha = runner.verify_selection(args, plan, frozen)
    cache, signature, populations, input_paths, input_sha = runner.load_inputs(args, plan)
    from train_saqre import load_captions
    from repair_head import RepairHead, RepairConfig
    from reference_kernels import validate_one
    if Path(sys.modules['repair_head'].__file__).resolve() != args.workspace / 'code/repair_head.py':
        raise ValueError('Wrong RepairHead import source')
    if any(c['config']['T'] != 1 for c in plan['configurations']) or len(plan['configurations']) != 14:
        raise ValueError('Probe expects all 14 initial T1 configurations')
    if (cache['c_emb'].shape[1], cache['f_emb'].shape[1]) != (6, 8):
        raise ValueError('Unexpected cache branch layout')
    pairs, queries, gallery = runner.pairs_for(cache, populations, load_captions(args.captions), 'train')
    train_ids = set(populations['train'])
    train_rows = [i for i, name in enumerate(cache['ids']) if name in train_ids]
    require_population(train_rows, populations['train'], cache)
    require_population(gallery, populations['development'], cache)
    if len(train_rows) != 6610 or len(pairs) != 23016 or len(queries) != len(gallery) or len(queries) != 918:
        raise ValueError('Formal train/development population mismatch')
    cap2vec, embedding_sha = runner.embed(plan, pairs, queries, torch)
    for seed in plan['seeds']:
        m = json.loads((args.results / f'seed_{seed}/manifest.json').read_text())
        if m['text_embeddings_sha256'] != embedding_sha:
            raise ValueError('Re-encoded train/development text differs from original execution')
    ProbeHead = make_probe_class(RepairHead)
    manifest = {'schema': 'pare.constant-token-probe.v1', 'status': 'running',
                'created_utc': runner.now(), 'selection_impact': False,
                'official_test_accessed': False, 'internal_check_evaluated': False,
                'n_heads_expected': 42, 'protocol_sha256': sha(args.protocol),
                'training_freeze_sha256': sha(args.freeze), 'selection_lock_sha256': selection_sha,
                'probe_source_manifest_sha256': sha(args.probe_source_manifest),
                'probe_script_sha256': sha(__file__), 'runner_sha256': sha(args.workspace / 'scripts/run_repair.py'),
                'input_sha256': input_sha, 'text_embeddings_sha256': embedding_sha,
                'cache_signature': signature, 'train_ids': populations['train'],
                'development_ids': populations['development'],
                'method': 'replace pi by branchwise training-valid-crop mean; all other paths unchanged',
                'baseline_replay_requirement': 'identical ranks/metrics and maximum score error <=1e-6',
                'environment': runner.environment(torch), 'runs': []}
    args.output.mkdir(parents=True)
    mp = args.output / 'manifest.json'
    runner.write_json(mp, manifest)
    try:
        for seed in plan['seeds']:
            for cfg in plan['configurations']:
                name = cfg['id']
                src = args.results / f'seed_{seed}' / name
                headpath = src / 'head.pt'
                original_hash = sha(headpath)
                saved = torch.load(headpath, map_location='cpu', weights_only=True)
                if saved['seed'] != seed or saved['config'] != cfg['config'] or saved['n_images'] != 6610 or saved['n_pairs'] != 23016:
                    raise ValueError('Checkpoint configuration/population mismatch')
                if set(saved['provenance']['train_ids']) != set(populations['train']):
                    raise ValueError('Checkpoint mean source differs from actual training images')
                result = json.loads((src / 'development.json').read_text())
                original = RepairHead(512, RepairConfig(**cfg['config'])).cuda().eval()
                original.load_state_dict(saved['state_dict'], strict=True)
                matrix, ranks, metrics, gold = validate_one(original, cache, gallery, queries, cap2vec, 'cuda')
                check_identity(result, cache, queries, gallery, gold)
                archived = torch.load(src / 'development_scores.pt', map_location='cpu', weights_only=True)['scores']
                error = float((matrix - archived).abs().max())
                if ranks != result['ranks'] or metrics != result['metrics_percent'] or error > 1e-6:
                    raise ValueError('Original-head development replay failed')
                stats = {split: {branch: branch_statistics(original, cache, rows, branch)
                                 for branch in ('coarse', 'fine')}
                         for split, rows in [('training', train_rows), ('development', gallery)]}
                means = {branch: stats['training'][branch]['applied_mean_pi_float32']
                         for branch in ('coarse', 'fine')}
                probe = ProbeHead(512, RepairConfig(**cfg['config'])).cuda().eval()
                probe.load_state_dict(saved['state_dict'], strict=True)
                probe.set_training_means(means, 6, 8)
                pmatrix, pranks, pmetrics, pgold = validate_one(probe, cache, gallery, queries, cap2vec, 'cuda')
                if pgold != gold or sha(headpath) != original_hash:
                    raise ValueError('Probe changed target identity or source checkpoint')
                if sha(args.results / 'SELECTION_LOCK.json') != selection_sha:
                    raise ValueError('Selection lock changed during probe')
                folder = args.output / f'seed_{seed}' / name
                folder.mkdir(parents=True)
                identity = {k: result[k] for k in ('query_ids', 'query_caption_sha256', 'gallery_ids', 'gold_gallery_index')}
                torch.save({'scores': pmatrix, **identity, 'checkpoint_sha256': original_hash,
                            'selection_lock_sha256': selection_sha, 'probe_script_sha256': sha(__file__)},
                           folder / 'constant_trainmean_pi_scores.pt')
                report = {**identity, 'seed': seed, 'configuration': cfg,
                          'checkpoint_sha256': original_hash, 'source_development_sha256': sha(src / 'development.json'),
                          'source_development_scores_sha256': sha(src / 'development_scores.pt'),
                          'baseline_replay_max_absolute_score_error': error,
                          'baseline_replay_ranks_identical': True, 'baseline_metrics_percent': metrics,
                          'baseline_ranks': ranks,
                          'probe_metrics_percent': pmetrics, 'probe_ranks': pranks,
                          'delta_probe_minus_original_pp': {k: pmetrics[k] - metrics[k] for k in metrics},
                          'prototype_statistics': stats, 'selection_impact': False,
                          'official_test_accessed': False, 'internal_check_evaluated': False,
                          'scope': 'fixed-head intervention; not matched T0 retraining or physical mechanism identification'}
                runner.write_json(folder / 'constant_trainmean_pi.json', report)
                manifest['runs'].append({'seed': seed, 'configuration': name,
                                         'baseline_metrics_percent': metrics, 'probe_metrics_percent': pmetrics,
                                         'delta_probe_minus_original_pp': report['delta_probe_minus_original_pp'],
                                         'outputs': {str(p.relative_to(args.output)): sha(p) for p in folder.iterdir()}})
                runner.write_json(mp, manifest)
                print(json.dumps({'probe_complete': name, 'seed': seed}), flush=True)
                del original, probe, matrix, pmatrix, archived, saved
        if len(manifest['runs']) != 42:
            raise ValueError('Incomplete probe grid')
        runner.check_freeze(args, plan)
        runner.verify_selection(args, plan, frozen)
        check_own_source(args.probe_source_manifest)
        if any(sha(path) != input_sha[name] for name, path in input_paths.items()):
            raise ValueError('A frozen input changed during probe')
        manifest.update(status='complete_all_42_development_probes', completed_utc=runner.now())
        runner.write_json(mp, manifest)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        runner.write_json(mp, manifest)
        raise


if __name__ == '__main__':
    main()
