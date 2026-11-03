"""Compare new full/seed-0 endpoint to the already-selected fit-only sweep trial."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    old = torch.load(a.prior, weights_only=True, map_location='cpu')
    new = torch.load(a.new / 'head.pt', weights_only=True, map_location='cpu')
    previous = json.loads(a.prior.with_suffix('.json').read_text())
    current = json.loads((a.new / 'validation.json').read_text())
    assert old['config']['seed'] == new['seed'] == 0
    assert old['config']['lr'] == new['settings']['lr'] == .002
    assert old['config']['kappa'] == new['settings']['kappa'] == .03
    assert old['config']['epochs'] == new['settings']['epochs'] == 10
    assert old['config']['bs'] == new['settings']['batch_size'] == 64
    assert old['ablate'] == [] and new['factors'] == dict(A=1, B=1, C=1, R=1, T=1)
    assert set(old['provenance']['train_ids']) == set(new['provenance']['train_ids'])
    assert len(new['provenance']['train_ids']) == 8263
    assert previous['query_ids'] == current['query_ids']
    assert previous['gallery_ids'] == current['gallery_ids']
    assert previous['checkpoint_sha256'] == sha(a.prior)
    errors = {key: float((value - new['state_dict'][key]).abs().max()) for key, value in old['state_dict'].items()}
    disagreements = sum(x != y for x, y in zip(previous['ranks'], current['ranks']))
    def metrics(ranks):
        z = {f'R@{k}': 100 * sum(x <= k for x in ranks) / len(ranks) for k in [1, 5, 10]}
        z['MRR@10'] = 100 * sum(1 / x if x <= 10 else 0 for x in ranks) / len(ranks)
        return z
    result = {'comparison': 'fit-only sweep vs new factorial full endpoint, identical split and seed',
              'prior_head_sha256': sha(a.prior), 'new_head_sha256': sha(a.new / 'head.pt'),
              'prior_ranks_sha256': sha(a.prior.with_suffix('.json')),
              'new_ranks_sha256': sha(a.new / 'validation.json'),
              'all_parameters_bitwise_equal': all(torch.equal(v, new['state_dict'][k]) for k, v in old['state_dict'].items()),
              'parameter_max_abs_errors': errors, 'max_parameter_abs_error': max(errors.values()),
              'rank_disagreement_count': disagreements, 'n_queries': 918,
              'prior_metrics_percent': metrics(previous['ranks']), 'new_metrics_percent': metrics(current['ranks']),
              'fit_only_group_verified': True, 'official_test_accessed': False,
              'checker_sha256': sha(Path(__file__))}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
