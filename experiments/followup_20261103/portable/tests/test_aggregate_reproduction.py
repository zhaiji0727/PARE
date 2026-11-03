from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import aggregate_reproduction as aggregate
from protocol_adapter import load_original, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.split = self.root / 'split'
        self.plan = deepcopy(load_original(ROOT.parent / 'original'))
        self.reference = aggregate.reference_module()
        fit = [f'fit_{i}' for i in range(8263)]
        val = [f'val_{i}' for i in range(918)]
        for name, value in [('fit_ids.json', fit), ('val_ids.json', val), ('train_ids.json', fit + val)]:
            write(self.split / name, value)
            self.plan['input_bindings']['split_file_sha256'][name] = sha(self.split / name)
        self.protocol = self.root / 'protocol.json'
        write(self.protocol, self.plan)
        self.protocol.with_suffix('.md').write_text('Synthetic fixture only; no scientific run.\n')
        self.protocol_sha = sha(self.protocol)
        self.patch = patch.object(aggregate, 'validate_reproduction_protocol', return_value=self.plan)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        # The immutable protocol adapter is tested separately. These synthetic
        # files test the complete-run/identity/hash/rank boundary without torch.
        bindings = self.plan['input_bindings']
        inputs = {'training_cache': bindings['training_cache_sha256'],
                  'training_captions': bindings['clean_training_captions_sha256'],
                  **bindings['split_file_sha256'], 'original_split_manifest': bindings['original_split_manifest_sha256'],
                  'pretrained_weight': self.plan['backbone']['pretrained_weight_sha256']}
        pairs = 'c' * 64
        embed = 'd' * 64
        self.expected = {}
        self.first_name = self.plan['configurations'][0]['id']
        for seed in self.plan['seeds']:
            initial = hashlib.sha256(str(seed).encode()).hexdigest()
            orders = self.reference.shuffled_orders(seed)
            manifest = {'status': 'complete_all_14_configurations', 'active_configuration': None,
                        'posthoc_exploratory': True, 'uses_official_test': False, 'settings': self.plan['training'],
                        'n_fit_images': 8263, 'n_fit_pairs': 28759, 'n_validation_queries': 918,
                        'n_validation_gallery': 918, 'cache_contains_images': 9181,
                        'train_ids': fit, 'validation_ids': val,
                        'initial_state_sha256': initial, 'text_embeddings_sha256': embed,
                        'identity': {'protocol_sha256': self.protocol_sha, 'seed': seed,
                                     'configuration_ids': [x['id'] for x in self.plan['configurations']],
                                     'source_sha256': aggregate.expected_sources(), 'input_sha256': inputs,
                                     'epoch_order_sha256': orders, 'initial_pairs_sha256': pairs}, 'runs': {}}
            for index, config in enumerate(self.plan['configurations']):
                name = config['id']
                d = self.runs / f'seed_{seed}' / name
                d.mkdir(parents=True)
                (d / 'head.pt').write_bytes(f'SYNTHETIC NON-TORCH HEAD {seed} {name}'.encode())
                (d / 'validation_scores.pt').write_bytes(f'SYNTHETIC NON-TORCH SCORES {seed} {name}'.encode())
                history = [{'epoch': ep + 1, 'batches': 450, 'pairs': 28759, 'tail_batch_pairs': 23,
                            'pair_order_sha256': orders[ep]} for ep in range(10)]
                write(d / 'training_progress.json', {'history': history})
                ranks = [1 + ((i * 3 + seed * 7 + index * 11) % 71) for i in range(918)]
                metrics = self.reference.from_ranks(ranks)
                self.expected[seed, name] = metrics
                result = {'configuration': config, 'seed': seed, 'protocol_sha256': self.protocol_sha,
                          'posthoc_exploratory': True, 'official_test_accessed': False,
                          'n_queries': 918, 'n_gallery': 918, 'checkpoint_sha256': sha(d / 'head.pt'),
                          'query_ids': val, 'gallery_ids': val, 'query_caption_sha256': ['e' * 64] * 918,
                          'gold_gallery_index': list(range(918)), 'ranks': ranks, 'metrics_percent': metrics}
                write(d / 'validation.json', result)
                manifest['runs'][name] = {'status': 'complete', 'configuration': config, 'seed': seed,
                                         'initial_state_sha256': initial, 'text_embeddings_sha256': embed,
                                         'initial_pairs_sha256': pairs, 'epoch_order_sha256': orders,
                                         'epochs': 10, 'batches': 4500, 'metrics_percent': metrics,
                                         'artifacts_sha256': {p.name: sha(p) for p in d.iterdir()}}
            write(self.runs / f'seed_{seed}' / 'manifest.json', manifest)

    def load(self):
        return aggregate.load_complete(self.protocol, self.protocol_sha, self.runs, self.split)

    def amend_manifest(self, mutator):
        path = self.runs / 'seed_0/manifest.json'
        value = json.loads(path.read_text())
        mutator(value)
        write(path, value)

    def test_all_tables_match_original_arithmetic_and_do_not_claim_tensor_replay(self):
        out = self.root / 'summary'
        result = aggregate.run(self.protocol, self.protocol_sha, self.runs, self.split, out)
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['tables'], self.reference.summarize(self.plan, self.expected))
        self.assertEqual(result['row_counts'], {'run_metrics': 42, 'configuration_summary': 14,
                                               'paired_contrasts_by_seed': 108, 'contrast_summary': 36,
                                               'primary_contrast_summary': 9})
        self.assertEqual(summary['execution_kind'], 'portable_reproduction')
        self.assertFalse(summary['validation_scope']['score_matrix_ranks_replayed'])
        for name, meta in result['outputs'].items():
            self.assertEqual(sha(out / name), meta['sha256'])

    def test_rejects_incomplete_seed_or_missing_configuration(self):
        self.amend_manifest(lambda m: m.__setitem__('status', 'running'))
        with self.assertRaises(ValueError):
            self.load()
        self.amend_manifest(lambda m: (m.__setitem__('status', 'complete_all_14_configurations'),
                                       m['runs'].pop(self.first_name)))
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_artifact_byte_change(self):
        (self.runs / 'seed_0' / self.first_name / 'head.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_within_seed_initialization_change(self):
        self.amend_manifest(lambda m: m['runs'][self.first_name].__setitem__('initial_state_sha256', 'f' * 64))
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_exported_metric_inconsistent_with_ranks(self):
        d = self.runs / 'seed_0' / self.first_name
        result = json.loads((d / 'validation.json').read_text())
        result['metrics_percent']['R@10'] += 1
        write(d / 'validation.json', result)
        self.amend_manifest(lambda m: m['runs'][self.first_name]['artifacts_sha256'].__setitem__('validation.json', sha(d / 'validation.json')))
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_query_identity_change_despite_updated_file_hash(self):
        d = self.runs / 'seed_0' / self.first_name
        result = json.loads((d / 'validation.json').read_text())
        result['query_ids'][0], result['query_ids'][1] = result['query_ids'][1], result['query_ids'][0]
        write(d / 'validation.json', result)
        self.amend_manifest(lambda m: m['runs'][self.first_name]['artifacts_sha256'].__setitem__('validation.json', sha(d / 'validation.json')))
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()
