#!/usr/bin/env python3
"""Verify public head/rank hashes and recompute all published metric/contrast tables.

No torch or dataset is required. This verifies the lightweight public evidence;
it does not recompute model scores or replace the archived matrix-replay audit.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

PROTOCOL_SHA = '42789ee8793b7c5bd59bf950dd8deefe8bf1c3caa2e9be2d54143630f63bf288'
AGGREGATOR_SHA = 'b132087d63dac8d3b476749e1c89cf1a62336b6257d4962da51bfe9590287fcb'
MANIFEST = 'benchmarks/followup_20261103/PUBLICATION_MANIFEST.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resolve(root, path):
    relative = Path(path)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe public path')
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), 'Public path leaves the repository')
    return result


def bound_bytes(root, item, key='sha256'):
    data = resolve(root, item['path']).read_bytes()
    require(digest(data) == item[key], 'File hash mismatch: ' + item['path'])
    return data


def compare(actual, expected, where='value'):
    if type(expected) in (float, int):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and abs(actual - expected) < 1e-10, 'Numeric mismatch: ' + where)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'Keys mismatch: ' + where)
        for key in expected:
            compare(actual[key], expected[key], where + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Length mismatch: ' + where)
        for i, value in enumerate(expected):
            compare(actual[i], value, where + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, 'Value mismatch: ' + where)


def verify(root, manifest_path=MANIFEST):
    root = Path(root).resolve()
    manifest = json.loads(resolve(root, manifest_path).read_bytes())
    require(manifest['posthoc_exploratory'] is True and manifest['official_test_used'] is False
            and manifest['validation_previously_used_for_selection'] is True, 'Incorrect study scope')
    seen_paths = set()
    for item in manifest['files']:
        require(item['path'] not in seen_paths, 'Duplicate file inventory entry')
        seen_paths.add(item['path'])
        bound_bytes(root, item)
    plan = json.loads(bound_bytes(root, manifest['protocol']))
    require(manifest['protocol']['sha256'] == PROTOCOL_SHA, 'Unexpected historical protocol')
    expected_keys = {(seed, c['id']) for seed in plan['seeds'] for c in plan['configurations']}
    require(len(expected_keys) == 42 and len(manifest['runs']) == 42, 'Expected all 42 runs')
    configs = {c['id']: c['factors'] for c in plan['configurations']}
    complete, identities = {}, None
    audit = json.loads(bound_bytes(root, manifest['original_audit']))
    require(audit['all_passed'] is True and audit['n_runs'] == 42, 'Original audit was not complete')
    audit_rows = {(x['seed'], x['configuration']): x for x in audit['runs']}
    for item in manifest['runs']:
        key = item['seed'], item['configuration_id']
        require(key in expected_keys and key not in complete, 'Missing, duplicate or unplanned run')
        require(item['factors'] == configs[key[1]], 'Factor mismatch')
        compressed = bound_bytes(root, item['record'], 'storage_sha256')
        require(item['record']['encoding'] == 'gzip', 'Expected gzip rank record')
        raw = gzip.decompress(compressed)
        require(digest(raw) == item['record']['original_sha256'], 'Original rank-record hash mismatch')
        record = json.loads(raw)
        bound_bytes(root, item['head'])
        ar = audit_rows[key]
        require(item['head']['sha256'] == ar['head_sha256']
                and digest(raw) == ar['validation_sha256']
                and item['matrix']['sha256'] == ar['matrix_sha256']
                and item['matrix']['published'] is False, 'Archive/audit binding mismatch')
        require(record['configuration'] == {'id': key[1], 'factors': configs[key[1]]}
                and record['seed'] == key[0] and record['checkpoint_sha256'] == item['head']['sha256'], 'Record/head identity mismatch')
        require(record['n_queries'] == record['n_gallery'] == 918 and record['official_test_accessed'] is False, 'Wrong query population')
        require(record['protocol_sha256'] == PROTOCOL_SHA and record['posthoc_exploratory'] is True, 'Wrong record protocol/scope')
        query_ids, gallery = record['query_ids'], record['gallery_ids']
        require(len(query_ids) == len(gallery) == len(set(query_ids)) == len(set(gallery)) == 918
                and set(query_ids) == set(gallery), 'Incorrect full validation identity set')
        hashes = record['query_caption_sha256']
        require(len(hashes) == 918 and all(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in hashes), 'Caption hashes must not contain caption bodies')
        identity = (query_ids, gallery, hashes, record['gold_gallery_index'])
        if identities is None:
            identities = identity
        require(identity == identities, 'Query/gallery identity drift across runs')
        require(all(gallery[g] == q for q, g in zip(query_ids, record['gold_gallery_index'])), 'Ground-truth index mismatch')
        ranks = record['ranks']
        require(len(ranks) == 918 and all(type(r) is int and 1 <= r <= 918 for r in ranks), 'Invalid ranks')
        metrics = {f'R@{k}': 100 * sum(r <= k for r in ranks) / 918 for k in (1,5,10)}
        metrics['MRR@10'] = 100 * sum(1/r if r <= 10 else 0 for r in ranks) / 918
        compare(record['metrics_percent'], metrics, str(key))
        complete[key] = metrics
    require(set(complete) == expected_keys, 'Incomplete run population')
    aggregator_path = resolve(root, 'experiments/followup_20261103/original/analysis/aggregate_followup.py')
    require(digest(aggregator_path.read_bytes()) == AGGREGATOR_SHA, 'Changed original summary arithmetic')
    spec = importlib.util.spec_from_file_location('pare_archived_aggregation', aggregator_path)
    aggregator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aggregator)
    tables = aggregator.summarize(plan, complete)
    summary = json.loads(bound_bytes(root, manifest['summary']))
    require(summary['posthoc_exploratory'] is True and summary['official_test_used'] is False, 'Incorrect summary scope')
    compare(summary['contrast_definitions'], plan['paired_contrasts'], 'contrast definitions')
    compare(summary['tables'], tables, 'all metric and contrast tables')
    for name, expected_rows in tables.items():
        path = resolve(root, 'benchmarks/followup_20261103/summary/' + name + '.csv')
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == len(expected_rows), 'CSV row count mismatch: ' + name)
        for actual, expected in zip(rows, expected_rows):
            converted = {}
            require(actual.keys() == expected.keys(), 'CSV columns mismatch')
            for key, value in expected.items():
                if type(value) is bool:
                    require(actual[key] in ('True','False'), 'Invalid CSV boolean')
                    converted[key] = actual[key] == 'True'
                else:
                    converted[key] = type(value)(actual[key])
            compare(converted, expected, name)
    return {'all_passed': True, 'runs': 42, 'heads_hash_verified': 42, 'query_rank_records': 42,
            'configurations': 14, 'paired_seed_rows': 108, 'contrasts': 36,
            'scope': 'Published byte hashes and rank-derived metrics/contrasts. Model scores are not recomputed.',
            'manifest_sha256': digest(resolve(root, manifest_path).read_bytes())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--manifest', default=MANIFEST)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = verify(a.repo, a.manifest)
    if a.out:
        require(not a.out.exists(), 'Refusing to overwrite a verification report')
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
