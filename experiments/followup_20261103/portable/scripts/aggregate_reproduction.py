#!/usr/bin/env python3
"""Summarize all 42 portable runs after byte-hash and exported-rank checks.

This does not load torch tensors or recompute scores/rankings from score matrices.
It verifies stored artifact bytes and recomputes metrics from exported ranks.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from protocol_adapter import ORIGINAL_PROTOCOL_SHA256, sha, validate_reproduction_protocol

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'original'
REFERENCE_SHA = 'b132087d63dac8d3b476749e1c89cf1a62336b6257d4962da51bfe9590287fcb'


def reference_module():
    path = ORIGINAL / 'analysis/aggregate_followup.py'
    if sha(path) != REFERENCE_SHA:
        raise ValueError('Original aggregation arithmetic changed')
    spec = importlib.util.spec_from_file_location('pare_original_aggregation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_sources():
    paths = {f'base_code/{p.name}': p for p in (ORIGINAL / 'base_code').glob('*.py')}
    paths.update({f'code/{p.name}': p for p in (ROOT / 'code').glob('*.py')})
    paths.update({f'scripts/{name}': ROOT / 'scripts' / name
                  for name in ['run_factorial_fitval.py', 'protocol_adapter.py']})
    return {name: sha(path) for name, path in paths.items()}


def load_complete(protocol, expected_sha, runs, split_dir):
    protocol, runs, split_dir = Path(protocol), Path(runs), Path(split_dir)
    plan = validate_reproduction_protocol(protocol, expected_sha, ORIGINAL)
    ref = reference_module()
    require = ref.require
    configs = {item['id']: item for item in plan['configurations']}
    require(len(configs) == 14 and plan['seeds'] == [0, 1, 2], 'Expected the complete 42-run design')
    bindings = plan['input_bindings']
    stable = {protocol: sha(protocol), protocol.with_suffix('.md'): sha(protocol.with_suffix('.md'))}
    for name, digest in bindings['split_file_sha256'].items():
        require(sha(split_dir / name) == digest, 'Fixed split changed: ' + name)
        stable[split_dir / name] = digest
    fit, val = set(ref.read(split_dir / 'fit_ids.json')), set(ref.read(split_dir / 'val_ids.json'))
    require(len(fit) == 8263 and len(val) == 918 and not fit & val, 'Invalid fixed populations')
    sources = expected_sources()
    inputs = {'training_cache': bindings['training_cache_sha256'],
              'training_captions': bindings['clean_training_captions_sha256'],
              **bindings['split_file_sha256'], 'original_split_manifest': bindings['original_split_manifest_sha256'],
              'pretrained_weight': plan['backbone']['pretrained_weight_sha256']}
    complete, records, initial_states, query_identity = {}, [], [], None
    common_pairs, common_embeddings = None, None
    for seed in plan['seeds']:
        directory = runs / f'seed_{seed}'
        path = directory / 'manifest.json'
        manifest = ref.read(path)
        stable[path] = sha(path)
        require(manifest.get('status') == 'complete_all_14_configurations'
                and manifest.get('active_configuration') is None, 'A seed is incomplete')
        require(set(manifest['runs']) == set(configs), 'A planned configuration is missing or duplicated')
        require(manifest['settings'] == plan['training'], 'Training settings changed')
        require(manifest.get('posthoc_exploratory') is True and manifest.get('uses_official_test') is False,
                'Expected exploratory validation without official test access')
        require((manifest['n_fit_images'], manifest['n_fit_pairs'], manifest['n_validation_queries'],
                 manifest['n_validation_gallery'], manifest['cache_contains_images']) == (8263, 28759, 918, 918, 9181),
                'Incorrect training or validation counts')
        require(len(manifest['train_ids']) == 8263 and set(manifest['train_ids']) == fit
                and len(manifest['validation_ids']) == 918 and set(manifest['validation_ids']) == val,
                'Fit or validation IDs differ')
        ident = manifest['identity']
        require(ident['protocol_sha256'] == expected_sha and ident['seed'] == seed, 'Seed/protocol mismatch')
        require(ident['configuration_ids'] == list(configs), 'Configuration order changed')
        require(ident['source_sha256'] == sources and ident['input_sha256'] == inputs, 'Source/input bindings differ')
        orders = ref.shuffled_orders(seed)
        require(ident['epoch_order_sha256'] == orders, 'Wrong seeded shuffle stream')
        require(ref.digest_string(manifest['initial_state_sha256']) and ref.digest_string(manifest['text_embeddings_sha256']),
                'Invalid initialization/embedding digest')
        initial_states.append(manifest['initial_state_sha256'])
        if common_pairs is None:
            common_pairs, common_embeddings = ident['initial_pairs_sha256'], manifest['text_embeddings_sha256']
        require(ident['initial_pairs_sha256'] == common_pairs and manifest['text_embeddings_sha256'] == common_embeddings,
                'Seeds did not use identical pairs and frozen text embeddings')
        for name, config in configs.items():
            run = manifest['runs'][name]
            require(run['status'] == 'complete' and run['configuration'] == config and run['seed'] == seed,
                    'Incomplete or mislabelled run')
            require(run['initial_state_sha256'] == manifest['initial_state_sha256']
                    and run['text_embeddings_sha256'] == common_embeddings
                    and run['initial_pairs_sha256'] == common_pairs, 'Within-seed pairing changed')
            require(run['epoch_order_sha256'] == orders and run['epochs'] == 10 and run['batches'] == 4500,
                    'Training extent or shuffle changed')
            require(set(run['artifacts_sha256']) == ref.ARTIFACTS, 'Unexpected run artifact set')
            run_dir = directory / name
            for filename, digest in run['artifacts_sha256'].items():
                require(sha(run_dir / filename) == digest, 'Stored artifact changed: ' + filename)
                stable[run_dir / filename] = digest
            history = ref.read(run_dir / 'training_progress.json')['history']
            require(len(history) == 10, 'Expected all ten epoch records')
            for epoch, row in enumerate(history):
                require((row['epoch'], row['batches'], row['pairs'], row['tail_batch_pairs']) == (epoch + 1, 450, 28759, 23)
                        and row['pair_order_sha256'] == orders[epoch], 'Epoch record differs')
            result = ref.read(run_dir / 'validation.json')
            require(result['configuration'] == config and result['seed'] == seed
                    and result['protocol_sha256'] == expected_sha, 'Exported result identity differs')
            require(result['posthoc_exploratory'] is True and result['official_test_accessed'] is False,
                    'Wrong evidence label')
            require(result['n_queries'] == result['n_gallery'] == 918, 'Wrong validation size')
            require(result['checkpoint_sha256'] == run['artifacts_sha256']['head.pt'], 'Result/head binding differs')
            query = {key: result[key] for key in ['query_ids', 'gallery_ids', 'query_caption_sha256', 'gold_gallery_index']}
            require(len(query['query_ids']) == len(set(query['query_ids'])) == 918 and set(query['query_ids']) == val
                    and len(query['gallery_ids']) == len(set(query['gallery_ids'])) == 918 and set(query['gallery_ids']) == val,
                    'Incomplete query/gallery identities')
            require(len(query['query_caption_sha256']) == 918 and all(ref.digest_string(x) for x in query['query_caption_sha256']),
                    'Invalid caption hashes')
            require(len(query['gold_gallery_index']) == 918 and all(type(g) is int and 0 <= g < 918
                    and query['gallery_ids'][g] == q for q, g in zip(query['query_ids'], query['gold_gallery_index'])),
                    'Incorrect relevant-image mapping')
            if query_identity is None:
                query_identity = query
            require(query == query_identity, 'Query identity differs between configurations/seeds')
            metrics = ref.from_ranks(result['ranks'])
            ref.check_metrics(result['metrics_percent'], metrics, name)
            ref.check_metrics(run['metrics_percent'], metrics, name)
            complete[seed, name] = metrics
            records.append({'seed': seed, 'configuration': name,
                            'artifacts_sha256': run['artifacts_sha256'], 'manifest_sha256': stable[path]})
    require(len(complete) == 42 and len(set(initial_states)) == 3, 'Expected all runs with three distinct seeded initializations')
    require(all(sha(path) == digest for path, digest in stable.items()), 'An input changed during aggregation')
    provenance = {'execution_kind': 'portable_reproduction', 'protocol_sha256': expected_sha,
                  'parent_protocol_sha256': ORIGINAL_PROTOCOL_SHA256, 'runs': records,
                  'validation_scope': {'artifact_bytes_checked': True, 'metrics_recomputed_from_exported_ranks': True,
                                       'torch_checkpoint_semantics_checked': False, 'scores_recomputed': False,
                                       'score_matrix_ranks_replayed': False},
                  'portable_aggregator_sha256': sha(__file__), 'reference_aggregator_sha256': REFERENCE_SHA}
    return plan, complete, provenance


def run(protocol, expected_sha, runs, split_dir, out):
    plan, complete, provenance = load_complete(protocol, expected_sha, runs, split_dir)
    ref = reference_module()
    tables = ref.summarize(plan, complete)
    manifest = ref.write_outputs(out, plan, tables, provenance)
    out = Path(out)
    document = ref.read(out / 'summary.json')
    document['execution_kind'] = 'portable_reproduction'
    document['validation_scope'] = provenance['validation_scope']
    (out / 'summary.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    (out / 'README.txt').write_text(
        'Portable reproduction of the fixed exploratory 42-run design, not the original historical execution.\n'
        'All 42 run artifact byte hashes were checked. Metrics were recomputed from exported query ranks.\n'
        'This tool did not load torch checkpoints, recompute scores, or replay rankings from score matrices.\n'
        'Aggregation and table arithmetic use the original unchanged summarize/write_outputs implementation.\n'
        'All 14 configurations and 36 contrasts are retained; no winner, p-value, or test replacement is selected.\n'
        'CSV/JSON retain unrounded values. TeX requires longtable, booktabs and amsmath.\n')
    manifest['execution_kind'] = 'portable_reproduction'
    manifest['portable_aggregator_sha256'] = sha(__file__)
    manifest['outputs'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                           for p in sorted(out.iterdir()) if p.name != 'summary_manifest.json'}
    (out / 'summary_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', required=True)
    parser.add_argument('--runs', type=Path, required=True, help='directory containing seed_0, seed_1 and seed_2')
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = run(args.protocol, args.expected_protocol_sha256, args.runs, args.split_dir, args.out)
    print(json.dumps({'status': 'complete', 'execution_kind': 'portable_reproduction', 'row_counts': output['row_counts']}))
