"""Synthetic adversarial checks; no experiment results are read."""
import copy
import json
from pathlib import Path
import statistics
import tempfile
import unittest

import analyze_context48 as a

THRESHOLDS = {'mean_delta_R10_minimum': .5, 'each_seed_delta_R10_minimum': 0.,
              'minimum_strictly_positive_R10_seeds': 2, 'mean_delta_MRR10_minimum': 0.}


def records():
    rows = {}
    for stage in a.STAGES:
        for seed in a.SEEDS:
            for name in (*a.CONTEXT_IDS, *(f'parent/{k}' for k in a.PARENT_IDS)):
                values = {k: 10. for k in a.METRICS}
                if name == 'context48_A1T1':
                    values = {k: 10. + seed for k in a.METRICS}
                rows[stage, seed, name] = values
    return rows


class AnalysisTests(unittest.TestCase):
    def test_paired_mean_and_sample_sd_not_pooled_sd(self):
        rows = records()
        for seed, base in enumerate((1., 20., 80.)):
            rows['development', seed, 'parent/selected']['R@10'] = base
            rows['development', seed, 'context48_A1T1']['R@10'] = base + (1., 2., 3.)[seed]
        report = a.summarize(rows, THRESHOLDS)
        r = next(x for x in report['contrast_summaries'] if x['contrast'] == 'context_support' and x['stage'] == 'development')
        self.assertEqual(r['R@10_per_seed_pp'], [1., 2., 3.])
        self.assertEqual(r['R@10_mean_pp'], 2.)
        self.assertEqual(r['R@10_sample_sd_pp'], 1.)

    def test_all_results_and_signed_effects_retained(self):
        rows = records()
        rows['internal_check', 2, 'context48_A1T1']['R@10'] = 1.
        r = a.summarize(rows, THRESHOLDS)
        self.assertEqual(len(r['all_runs']), 48)
        self.assertEqual(len(r['configuration_summaries']), 16)
        self.assertEqual(len(r['contrast_summaries']), 12)
        self.assertEqual(len(r['paired_contrasts']), 36)
        self.assertTrue(any(x['R@10'] < 0 for x in r['paired_contrasts']))
        self.assertFalse(r['all_three_primary_pass_both_splits'])

    def test_missing_or_extra_run_rejected(self):
        rows = records(); rows.pop(next(iter(rows)))
        with self.assertRaises(ValueError): a.summarize(rows, THRESHOLDS)
        rows = records(); rows['development', 7, 'context48_A1T1'] = {}
        with self.assertRaises(ValueError): a.summarize(rows, THRESHOLDS)

    def test_thresholds_exact_and_all_conditions_needed(self):
        self.assertTrue(a.engineering_rule([0, .75, .75], [0, 0, 0], THRESHOLDS)['pass'])
        for r10, mrr in (([-1e-15, 1, 1], [0, 0, 0]), ([0, 0, 3], [0, 0, 0]),
                         ([.5-1e-12] * 3, [0, 0, 0]), ([1, 1, 1], [0, 0, -1e-15])):
            self.assertFalse(a.engineering_rule(r10, mrr, THRESHOLDS)['pass'])

    def test_both_splits_required(self):
        rows = records()
        for seed in a.SEEDS:
            rows['internal_check', seed, 'context48_A1T1'] = {k: 9. for k in a.METRICS}
        r = a.summarize(rows, THRESHOLDS)
        self.assertTrue(r['primary_acceptance']['context_support']['passes']['development'])
        self.assertFalse(r['primary_acceptance']['context_support']['pass_both_splits'])

    def test_partial_or_wrong_count_audit_rejected(self):
        audit = {'status': 'complete_verified_all_9', 'official_test_accessed': False,
                 'new_training_performed': False, 'protocol_sha256': 'p', 'freeze_sha256': 'f',
                 'counts': {'heads': 9, 'development_evaluations': 9, 'internal_check_evaluations': 9, 'score_matrices_replayed': 18},
                 'runs': [{}] * 9, 'all_scores_ranks_metrics_match': True,
                 'active_configurations_complete_no_extra': True}
        a.check_audit(audit, 9, 'p', 'f')
        for key, value in [('status', 'running'), ('all_scores_ranks_metrics_match', False)]:
            bad = copy.deepcopy(audit); bad[key] = value
            with self.assertRaises(ValueError): a.check_audit(bad, 9, 'p', 'f')
        audit['counts']['score_matrices_replayed'] = 17
        with self.assertRaises(ValueError): a.check_audit(audit, 9, 'p', 'f')

    def test_query_identity_and_metric_mutation_rejected(self):
        n = 918; ids = [str(i) for i in range(n)]; cfg = {'A': 1}
        r = {'configuration': {'id': 'x', 'config': cfg}, 'seed': 0, 'protocol_sha256': 'p', 'freeze_sha256': 'f',
             'checkpoint_sha256': 'h', 'official_test_accessed': False, 'posthoc_exploratory': True,
             'n_queries': n, 'n_gallery': n, 'query_ids': ids, 'gallery_ids': ids,
             'query_caption_sha256': ['f' * 64] * n, 'gold_gallery_index': list(range(n)),
             'ranks': [11] * n, 'metrics_percent': {k: 0. for k in a.METRICS}}
        kw = dict(name='x', config=cfg, seed=0, stage='development', ids=set(ids), protocol_sha='p', freeze_sha='f', head_sha='h')
        a.validate_evaluation(r, **kw)
        for mutation in ('rank', 'metric', 'gold', 'size', 'id', 'head'):
            bad = copy.deepcopy(r)
            if mutation == 'rank': bad['ranks'][0] = 0
            if mutation == 'metric': bad['metrics_percent']['R@10'] = 1.
            if mutation == 'gold': bad['gold_gallery_index'][0] = 1
            if mutation == 'size': bad['n_gallery'] = 917
            if mutation == 'id': bad['query_ids'][0] = 'not-present'
            if mutation == 'head': bad['checkpoint_sha256'] = 'other'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): a.validate_evaluation(bad, **kw)

    def test_artifact_escape_or_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'x').write_text('original')
            binding = {'x': a.sha(root / 'x')}; a.check_artifacts(root, binding)
            (root / 'x').write_text('changed')
            with self.assertRaises(ValueError): a.check_artifacts(root, binding)
            with self.assertRaises(ValueError): a.inside(root, '../outside')
            with self.assertRaises(ValueError): a.inside(root, '/outside')

    def test_output_refuses_overwrite_and_contains_all_comparisons(self):
        result = {**a.summarize(records(), THRESHOLDS), 'provenance': {}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'report'; a.write_outputs(out, result)
            self.assertEqual(len(json.loads((out / 'analysis.json').read_text())['contrast_summaries']), 12)
            self.assertIn('versus_parent_original_R_gate_only', (out / 'SUMMARY.md').read_text())
            with self.assertRaises(ValueError): a.write_outputs(out, result)


if __name__ == '__main__': unittest.main()
