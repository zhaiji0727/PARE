#!/usr/bin/env python3
"""Independent complete-probe hash/rank checks and nonselective summaries."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def require(test, message):
    if not test: raise ValueError(message)
def metrics(ranks):
    require(len(ranks) == 918 and all(type(r) is int and 1 <= r <= 918 for r in ranks), 'Invalid ranks')
    return {**{f'R@{k}': 100*sum(r <= k for r in ranks)/918 for k in (1, 5, 10)},
            'MRR@10': 100*sum(1/r if r <= 10 else 0 for r in ranks)/918}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    require(not a.out.exists(), 'Output must be new')
    w, probe = a.workspace.resolve(), a.probe.resolve()
    plan, manifest = read(w/'protocol.json'), read(probe/'manifest.json')
    require(manifest['status'] == 'complete_all_42_development_probes', 'Probe must be fully complete')
    expected = {(s, c['id']) for s in plan['seeds'] for c in plan['configurations']}
    observed = {(r['seed'], r['configuration']) for r in manifest['runs']}
    require(len(manifest['runs']) == len(expected) == 42 and observed == expected, 'Need exact 42 probe runs')
    require(manifest['selection_impact'] is False and manifest['official_test_accessed'] is False
            and manifest['internal_check_evaluated'] is False, 'Scope changed')
    lock_sha = sha(w/'results/SELECTION_LOCK.json')
    require(manifest['selection_lock_sha256'] == lock_sha, 'Selection lock mismatch')
    require(manifest['protocol_sha256'] == sha(w/'protocol.json'), 'Protocol mismatch')
    require(manifest['training_freeze_sha256'] == sha(w/'REPAIR_SOURCE_FROZEN.json'), 'Training freeze mismatch')
    require(manifest['probe_script_sha256'] == sha(w/'audit/run_constant_token_probe.py'), 'Probe source mismatch')
    require(manifest['probe_source_manifest_sha256'] == sha(w/'audit/PROBE_SOURCE_FROZEN.json'), 'Probe source manifest mismatch')
    require(sha(a.cache) == manifest['input_sha256']['training_cache'], 'Cache mismatch')
    manifest_sha = sha(probe/'manifest.json')
    train = set(read(w/'plans/splits/train_ids.json')); dev = set(read(w/'plans/splits/development_ids.json'))
    require(len(train) == 6610 and len(dev) == 918 and not train & dev, 'Wrong population')
    require(set(manifest['train_ids']) == train and set(manifest['development_ids']) == dev, 'Probe population mismatch')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    require(torch.cuda.is_available(), 'CUDA needed for exact rank replay')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cache = torch.load(a.cache, map_location='cpu', mmap=True, weights_only=True)
    counts = {}
    for split, ids in [('training', train), ('development', dev)]:
        rows = torch.tensor([i for i, name in enumerate(cache['ids']) if name in ids])
        for branch, key in [('coarse', 'c'), ('fine', 'f')]: counts[split, branch] = int(cache[key+'_n'][rows].sum())
    del cache
    configs = {c['id']: c for c in plan['configurations']}
    rows, bindings = [], {}
    for entry in manifest['runs']:
        seed, name = entry['seed'], entry['configuration']
        directory = probe/f'seed_{seed}'/name
        expected_paths = {str((directory/n).relative_to(probe)) for n in ('constant_trainmean_pi.json', 'constant_trainmean_pi_scores.pt')}
        require(set(entry['outputs']) == expected_paths, 'Unexpected artifact set')
        for rel, digest in entry['outputs'].items():
            require(sha(probe/rel) == digest, 'Output SHA mismatch: '+rel); bindings[rel] = digest
        r = read(directory/'constant_trainmean_pi.json')
        require(r['seed'] == seed and r['configuration'] == configs[name], 'Config/seed mismatch')
        original = w/'results'/f'seed_{seed}'/name
        require(r['checkpoint_sha256'] == sha(original/'head.pt'), 'Head mismatch')
        require(r['source_development_sha256'] == sha(original/'development.json'), 'Original JSON mismatch')
        require(r['source_development_scores_sha256'] == sha(original/'development_scores.pt'), 'Original matrix mismatch')
        base = read(original/'development.json')
        for key in ('query_ids', 'query_caption_sha256', 'gallery_ids', 'gold_gallery_index'):
            require(r[key] == base[key], 'Identity/order mismatch: '+key)
        require(len(r['query_ids']) == len(set(r['query_ids'])) == 918 and set(r['query_ids']) == dev, 'Wrong queries')
        require(len(r['gallery_ids']) == len(set(r['gallery_ids'])) == 918 and set(r['gallery_ids']) == dev, 'Wrong gallery')
        require(r['baseline_replay_ranks_identical'] is True and 0 <= r['baseline_replay_max_absolute_score_error'] <= 1e-6, 'Original replay failed')
        require(r['baseline_ranks'] == base['ranks'] and metrics(r['baseline_ranks']) == r['baseline_metrics_percent'] == base['metrics_percent'], 'Baseline ranks/metrics mismatch')
        sm = torch.load(directory/'constant_trainmean_pi_scores.pt', map_location='cpu', weights_only=True)
        for key in ('query_ids', 'query_caption_sha256', 'gallery_ids', 'gold_gallery_index'):
            require(sm[key] == r[key], 'Probe matrix identity mismatch')
        require(sm['checkpoint_sha256'] == r['checkpoint_sha256'] and sm['selection_lock_sha256'] == lock_sha
                and sm['probe_script_sha256'] == manifest['probe_script_sha256'], 'Probe matrix bindings mismatch')
        scores = sm['scores']; require(scores.dtype == torch.float32 and tuple(scores.shape) == (918,918) and torch.isfinite(scores).all(), 'Invalid probe matrix')
        ranks = []
        for start in range(0, 918, 16):
            order = torch.argsort(scores[start:start+16].cuda(), dim=1, descending=True)
            gold = torch.tensor(r['gold_gallery_index'][start:start+16], device='cuda')[:,None]
            match = order.eq(gold);require(match.sum(1).eq(1).all(), 'Gold missing')
            ranks.extend(match.to(torch.int64).argmax(1).add(1).cpu().tolist())
        require(ranks == r['probe_ranks'] and metrics(ranks) == r['probe_metrics_percent'], 'Probe rank/metric mismatch')
        row = {'configuration': name, 'seed': seed}
        for metric in ('R@1','R@5','R@10','MRR@10'):
            delta = r['probe_metrics_percent'][metric]-r['baseline_metrics_percent'][metric]
            require(delta == r['delta_probe_minus_original_pp'][metric], 'Delta mismatch')
            row[metric+'_original'] = r['baseline_metrics_percent'][metric]
            row[metric+'_constant_pi'] = r['probe_metrics_percent'][metric]
            row[metric+'_delta_pp'] = delta
        require(entry['baseline_metrics_percent'] == r['baseline_metrics_percent'] and entry['probe_metrics_percent'] == r['probe_metrics_percent'] and entry['delta_probe_minus_original_pp'] == r['delta_probe_minus_original_pp'], 'Manifest summary mismatch')
        for split, n in [('training',6610),('development',918)]:
            for branch in ('coarse','fine'):
                stat = r['prototype_statistics'][split][branch]
                require(stat['n_images'] == n and stat['n_valid_crops'] == counts[split,branch], 'Mean crop population mismatch')
                require(sum(stat['argmax_prototype_counts']) == counts[split,branch], 'Occupancy mismatch')
                require(all(math.isfinite(x) and 0 <= x <= 1 for x in stat['mean_pi_float64']) and abs(sum(stat['mean_pi_float64'])-1)<1e-6, 'Invalid mean pi')
        rows.append(row)
    require(len(bindings)==84 and sha(probe/'manifest.json')==manifest_sha and sha(w/'results/SELECTION_LOCK.json')==lock_sha, 'Artifacts changed during verification')
    grouped = []
    for config in plan['configurations']:
        subset = [r for r in rows if r['configuration']==config['id']]
        require(len(subset)==3 and {r['seed'] for r in subset}=={0,1,2}, 'Missing seed')
        out = {'configuration': config['id'], 'config': config['config']}
        for metric in ('R@1','R@5','R@10','MRR@10'):
            out[metric+'_original_mean'] = statistics.mean(r[metric+'_original'] for r in subset)
            out[metric+'_constant_pi_mean'] = statistics.mean(r[metric+'_constant_pi'] for r in subset)
            out[metric+'_delta_mean_pp'] = statistics.mean(r[metric+'_delta_pp'] for r in subset)
            out[metric+'_delta_sample_sd_pp'] = statistics.stdev(r[metric+'_delta_pp'] for r in subset)
        grouped.append(out)
    report={'schema':'pare.constant-probe-independent-verification.v1','status':'complete_verified_all_42',
            'all_passed':True,'probe_matrices_replayed':42,'original_replays_verified':42,'output_files_verified':84,
            'probe_manifest_sha256':manifest_sha,'selection_lock_sha256':lock_sha,'verifier_sha256':sha(__file__),
            'probe_script_sha256':manifest['probe_script_sha256'],'artifact_sha256':bindings,
            'selected_configuration_id':read(w/'results/SELECTION_LOCK.json')['selected_configuration_id'],
            'selection_impact':False,'official_test_accessed':False,'internal_check_evaluated':False,
            'scope':'Fixed-head train-mean-pi intervention on development only, not T0 retraining or physical identification',
            'runs':rows,'configuration_summaries':grouped}
    a.out.mkdir(parents=True)
    (a.out/'verification_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,data in [('all_42_probes.csv',rows),('all_14_configuration_summaries.csv',[{k:v for k,v in r.items() if k!='config'} for r in grouped])]:
        with (a.out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({'all_passed':True,'runs':42,'verification_sha256':sha(a.out/'verification_summary.json')}))


if __name__=='__main__': main()
