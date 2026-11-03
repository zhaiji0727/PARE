"""Read-only verification of all 42 outputs; write a new verification report."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    root = a.root
    freeze = json.loads((root / 'FOLLOWUP_SOURCE_FROZEN.json').read_text())
    for rel, digest in freeze['file_sha256'].items():
        assert sha(root / rel) == digest, rel
    plan = json.loads((root / 'plans/protocol.json').read_text())
    configs = {x['id']: x for x in plan['configurations']}
    fit = set(json.loads((root / 'data/ret_clean_20261007/fit_ids.json').read_text()))
    val = set(json.loads((root / 'data/ret_clean_20261007/val_ids.json').read_text()))
    assert len(fit) == 8263 and len(val) == 918 and not fit & val
    rows, seed_initials, seed_orders, seed_embeddings, seed_manifests = [], {}, {}, {}, {}
    first_query_identity = None
    for seed in [0, 1, 2]:
        directory = root / 'results' / f'seed_{seed}'
        m = json.loads((directory / 'manifest.json').read_text())
        seed_manifests[str(seed)] = {'relative_path': f'results/seed_{seed}/manifest.json',
                                    'sha256': sha(directory / 'manifest.json')}
        assert m['status'] == 'complete_all_14_configurations' and set(m['runs']) == set(configs)
        assert m['identity']['seed'] == seed and m['n_fit_images'] == 8263 and m['n_fit_pairs'] == 28759
        assert set(m['train_ids']) == fit and set(m['validation_ids']) == val
        assert m['identity']['protocol_sha256'] == sha(root / 'plans/protocol.json')
        seed_initials[str(seed)] = m['initial_state_sha256']
        seed_orders[str(seed)] = m['identity']['epoch_order_sha256']
        seed_embeddings[str(seed)] = m['text_embeddings_sha256']
        for name, run in m['runs'].items():
            d = directory / name
            assert run['status'] == 'complete' and run['configuration'] == configs[name] and run['seed'] == seed
            assert run['initial_state_sha256'] == m['initial_state_sha256']
            assert run['text_embeddings_sha256'] == m['text_embeddings_sha256']
            assert run['epoch_order_sha256'] == m['identity']['epoch_order_sha256']
            for rel, digest in run['artifacts_sha256'].items():
                assert sha(d / rel) == digest, (seed, name, rel)
            h = torch.load(d / 'head.pt', weights_only=True, map_location='cpu')
            assert h['schema'] == 'pare.factorial-head.v1' and h['D'] == 512
            assert h['factors'] == configs[name]['factors'] and h['seed'] == seed
            assert h['n_images'] == 8263 and h['n_pairs'] == 28759
            assert set(h['provenance']['train_ids']) == fit and set(h['provenance']['validation_ids']) == val
            assert h['provenance']['input_sha256'] == m['identity']['input_sha256']
            assert h['provenance']['source_sha256'] == m['identity']['source_sha256']
            assert h['provenance']['initial_state_sha256'] == m['initial_state_sha256']
            assert h['settings'] == plan['training']
            assert len(h['history']) == 10
            for epoch, item in enumerate(h['history']):
                assert item['epoch'] == epoch + 1 and item['pairs'] == 28759
                assert item['batches'] == 450 and item['tail_batch_pairs'] == 23
                assert item['pair_order_sha256'] == m['identity']['epoch_order_sha256'][epoch]
            assert all(torch.isfinite(t).all() for t in h['state_dict'].values())
            z = torch.load(d / 'validation_scores.pt', weights_only=True, map_location='cpu')
            r = json.loads((d / 'validation.json').read_text())
            assert r['checkpoint_sha256'] == sha(d / 'head.pt')
            assert z['scores'].shape == (918, 918) and torch.isfinite(z['scores']).all()
            for key in ['query_ids', 'gallery_ids', 'query_caption_sha256', 'gold_gallery_index']:
                assert z[key] == r[key], (seed, name, key)
            assert len(set(r['query_ids'])) == 918 and set(r['query_ids']) == val
            assert len(set(r['gallery_ids'])) == 918 and set(r['gallery_ids']) == val
            query_identity = {key: r[key] for key in ['query_ids', 'query_caption_sha256', 'gallery_ids', 'gold_gallery_index']}
            if first_query_identity is None:
                first_query_identity = query_identity
            assert query_identity == first_query_identity
            assert all(r['gallery_ids'][g] == q for q, g in zip(r['query_ids'], r['gold_gallery_index']))
            order = torch.argsort(z['scores'].to(a.device), dim=1, descending=True)
            gold = torch.tensor(z['gold_gallery_index'], device=a.device).unsqueeze(1)
            ranks = (order == gold).nonzero(as_tuple=True)[1].add(1).tolist()
            assert ranks == r['ranks'], (seed, name, 'rank replay')
            metrics = {f'R@{k}': 100 * sum(x <= k for x in ranks) / 918 for k in [1, 5, 10]}
            metrics['MRR@10'] = 100 * sum(1 / x if x <= 10 else 0 for x in ranks) / 918
            assert metrics == r['metrics_percent'] == run['metrics_percent']
            rows.append({'seed': seed, 'configuration': name, 'metrics_percent': metrics,
                         'head_sha256': sha(d / 'head.pt'), 'matrix_sha256': sha(d / 'validation_scores.pt'),
                         'validation_sha256': sha(d / 'validation.json'),
                         'relative_directory': f'results/seed_{seed}/{name}',
                         'initial_state_sha256': m['initial_state_sha256'], 'all_checks_passed': True})
    assert len(rows) == 42
    assert len(set(seed_initials.values())) == 3
    assert len({tuple(order) for order in seed_orders.values()}) == 3
    assert sha(root / 'inputs/cache_physics_train.pt') == plan['input_bindings']['training_cache_sha256']
    a.out.write_text(json.dumps({'all_passed': True, 'runs': rows, 'n_runs': 42,
                                'source_freeze_sha256': sha(root / 'FOLLOWUP_SOURCE_FROZEN.json'),
                                'protocol_sha256': sha(root / 'plans/protocol.json'),
                                'seed_manifests': seed_manifests,
                                'verifier_sha256': sha(__file__), 'rank_replay_device': a.device,
                                'initial_state_sha256_by_seed': seed_initials,
                                'frozen_text_embeddings_sha256_by_seed': seed_embeddings,
                                'original_cache_unchanged': True}, indent=2) + '\n')
    print(json.dumps({'all_passed': True, 'n_runs': 42, 'report_sha256': sha(a.out)}))


if __name__ == '__main__':
    main()
