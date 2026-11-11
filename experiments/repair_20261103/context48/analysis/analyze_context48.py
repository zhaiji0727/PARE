#!/usr/bin/env python3
"""Nonselective analysis of the complete, independently audited context48 study.

No model imports, training, candidate choice, or score recomputation. Numerical
input is read back from bound per-query JSON and checked against audit records.
The independent auditor, whose source is hashed here, checks weights/matrices.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SCIENCE_SHA = '217545b95394076aa4dd11c17988c6637bf2869fc24ad100867af2d94feae80f'
PARENT_SCIENCE_SHA = 'ad4339e9d1d2a1e67f989747160bf318c2f1fe6c9a062ad7eee5a0419c83974f'
PARENT_LOCK_SHA = '2f0df93aa09b0af7a88d064a7675f3a4ed3f2a4a316c2035f5f3da5250489f07'
METRICS = ('R@1', 'R@5', 'R@10', 'MRR@10')
SEEDS = (0, 1, 2)
CONTEXT_IDS = ('context48_A1T1', 'context48_A0T1', 'context48_A1T0')
PARENT_IDS = {
    'selected': 'ABC101_T1_Rconstant_Tokoriginal',
    'full': 'ABC111_T1_Roriginal_Tokoriginal',
    'strict': 'ABC000_T1_Roriginal_Tokoriginal',
    'matched_A0': 'ABC001_T1_Rconstant_Tokoriginal',
    'original_R_gate_only': 'ABC101_T1_Roriginal_Tokoriginal',
}
STAGES = ('development', 'internal_check')
N = {'development': 918, 'internal_check': 1653}
# The first three are the protocol's exact primary contrasts. The other three
# are reported comparators, not additional selection criteria or hypotheses.
CONTRASTS = (
    ('context_support', 'primary', 'context48_A1T1', 'parent/selected'),
    ('fine_contribution', 'primary', 'context48_A1T1', 'context48_A0T1'),
    ('token_contribution', 'primary', 'context48_A1T1', 'context48_A1T0'),
    ('versus_parent_full', 'descriptive', 'context48_A1T1', 'parent/full'),
    ('versus_parent_strict', 'descriptive', 'context48_A1T1', 'parent/strict'),
    ('versus_parent_original_R_gate_only', 'descriptive', 'context48_A1T1', 'parent/original_R_gate_only'),
)


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Artifact must be relative')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact escapes its results directory')
    return path


def check_artifacts(root, expected):
    require(isinstance(expected, dict) and bool(expected), 'Missing artifact bindings')
    for relative, digest in expected.items():
        require(sha(inside(root, relative)) == digest, 'Artifact changed: ' + relative)


def rank_metrics(ranks, n):
    require(len(ranks) == n and all(type(r) is int and 1 <= r <= n for r in ranks), 'Invalid ranks')
    return {**{f'R@{k}': 100 * sum(r <= k for r in ranks) / n for k in (1, 5, 10)},
            'MRR@10': 100 * sum(1 / r if r <= 10 else 0 for r in ranks) / n}


def validate_evaluation(r, *, name, config, seed, stage, ids, protocol_sha, freeze_sha, head_sha):
    n = N[stage]
    require(r['configuration']['id'] == name and r['configuration']['config'] == config, 'Wrong evaluated configuration')
    require(r['seed'] == seed and r['protocol_sha256'] == protocol_sha and r['freeze_sha256'] == freeze_sha, 'Evaluation identity mismatch')
    require(r['checkpoint_sha256'] == head_sha, 'Evaluation head binding mismatch')
    require(r['official_test_accessed'] is False and r['posthoc_exploratory'] is True, 'Evaluation scope mismatch')
    require(r['n_queries'] == r['n_gallery'] == n, 'Wrong evaluation size')
    for key in ('query_ids', 'gallery_ids'):
        require(len(r[key]) == len(set(r[key])) == n and set(r[key]) == ids, 'Wrong evaluation population/order')
    require(len(r['query_caption_sha256']) == n and all(isinstance(h, str) and len(h) == 64 for h in r['query_caption_sha256']), 'Missing first-caption bindings')
    gold = r['gold_gallery_index']
    require(len(gold) == n and all(type(i) is int and 0 <= i < n for i in gold), 'Invalid gold index')
    require([r['gallery_ids'][i] for i in gold] == r['query_ids'], 'Gold identities mismatch')
    m = rank_metrics(r['ranks'], n)
    require(set(r['metrics_percent']) == set(METRICS), 'Wrong metric fields')
    require(all(math.isclose(m[k], r['metrics_percent'][k], abs_tol=1e-10, rel_tol=0) for k in METRICS), 'Metrics differ from ranks')
    return m, {k: r[k] for k in ('query_ids', 'query_caption_sha256', 'gallery_ids', 'gold_gallery_index')}


def check_audit(audit, count, protocol_sha, freeze_sha):
    require(audit['status'] == f'complete_verified_all_{count}', 'Independent audit is incomplete')
    require(audit['official_test_accessed'] is False and audit['new_training_performed'] is False, 'Wrong independent audit scope')
    require(audit['protocol_sha256'] == protocol_sha and audit['freeze_sha256'] == freeze_sha, 'Audit provenance mismatch')
    require(audit['counts']['heads'] == count and audit['counts']['development_evaluations'] == count
            and audit['counts']['internal_check_evaluations'] == count
            and audit['counts']['score_matrices_replayed'] == count * 2, 'Incomplete independent audit coverage')
    require(len(audit['runs']) == count, 'Audit run count differs')
    require(audit['all_scores_ranks_metrics_match'] is True, 'Independent replay did not pass')
    require(audit['active_configurations_complete_no_extra'] is True, 'Incomplete/unexpected grid')


def engineering_rule(r10, mrr, thresholds):
    require(len(r10) == len(mrr) == 3 and all(math.isfinite(x) for x in [*r10, *mrr]), 'Need three finite paired effects')
    checks = {
        'mean_R10_at_least_threshold': statistics.mean(r10) >= thresholds['mean_delta_R10_minimum'],
        'every_seed_R10_nonnegative': min(r10) >= thresholds['each_seed_delta_R10_minimum'],
        'at_least_two_positive_R10_seeds': sum(x > 0 for x in r10) >= thresholds['minimum_strictly_positive_R10_seeds'],
        'mean_MRR_nonnegative': statistics.mean(mrr) >= thresholds['mean_delta_MRR10_minimum'],
    }
    return {'pass': all(checks.values()), 'checks': checks}


def summarize(records, thresholds):
    names = (*CONTEXT_IDS, *(f'parent/{k}' for k in PARENT_IDS))
    expected = {(stage, seed, name) for stage in STAGES for seed in SEEDS for name in names}
    require(set(records) == expected, 'Analysis requires all 18 context and all 30 parent comparator evaluations')
    runs, groups, contrasts, paired = [], [], [], []
    for stage in STAGES:
        for name in names:
            group = {'stage': stage, 'configuration': name}
            for seed in SEEDS:
                runs.append({'stage': stage, 'configuration': name, 'seed': seed, **records[stage, seed, name]})
            for metric in METRICS:
                values = [records[stage, seed, name][metric] for seed in SEEDS]
                group[metric + '_mean'] = statistics.mean(values)
                group[metric + '_sample_sd'] = statistics.stdev(values)
            groups.append(group)
        for cid, kind, positive, negative in CONTRASTS:
            row = {'stage': stage, 'contrast': cid, 'kind': kind, 'positive': positive, 'negative': negative}
            effects = {k: [] for k in METRICS}
            for seed in SEEDS:
                delta = {k: records[stage, seed, positive][k] - records[stage, seed, negative][k] for k in METRICS}
                paired.append({'stage': stage, 'contrast': cid, 'kind': kind, 'seed': seed, **delta})
                for k in METRICS:
                    effects[k].append(delta[k])
            for k, values in effects.items():
                row[k + '_mean_pp'] = statistics.mean(values)
                row[k + '_sample_sd_pp'] = statistics.stdev(values)
                row[k + '_per_seed_pp'] = values
            row['engineering_rule'] = engineering_rule(effects['R@10'], effects['MRR@10'], thresholds)
            contrasts.append(row)
    primary = {cid: {'pass_both_splits': all(r['engineering_rule']['pass'] for r in contrasts if r['contrast'] == cid),
                     'passes': {r['stage']: r['engineering_rule']['pass'] for r in contrasts if r['contrast'] == cid}}
               for cid, kind, _, _ in CONTRASTS if kind == 'primary'}
    return {'all_runs': runs, 'configuration_summaries': groups, 'paired_contrasts': paired,
            'contrast_summaries': contrasts, 'primary_acceptance': primary,
            'all_three_primary_pass_both_splits': all(x['pass_both_splits'] for x in primary.values())}


def load_complete(args):
    w, parent = args.workspace.resolve(), args.parent.resolve()
    root, proot = w / 'results', parent / 'results'
    require(sha(w / 'protocol.json') == SCIENCE_SHA, 'Wrong fixed context protocol')
    require(sha(parent / 'protocol.json') == PARENT_SCIENCE_SHA, 'Wrong fixed parent protocol')
    require(sha(proot / 'SELECTION_LOCK.json') == PARENT_LOCK_SHA, 'Parent selection changed')
    plan, pp = read(w / 'protocol.json'), read(parent / 'protocol.json')
    configs = {c['id']: c['config'] for c in plan['configurations']}
    pcfg = {c['id']: c['config'] for c in pp['configurations']}
    require(tuple(configs) == CONTEXT_IDS and plan['seeds'] == list(SEEDS), 'Wrong context grid')
    f, pf = w / 'CONTEXT_SOURCE_FROZEN.json', parent / 'REPAIR_SOURCE_FROZEN.json'
    frozen, parent_frozen = read(f), read(pf)
    require(frozen['protocol_sha256'] == SCIENCE_SHA and parent_frozen['protocol_sha256'] == PARENT_SCIENCE_SHA, 'Source/protocol mismatch')
    for relative, digest in frozen['source_sha256'].items():
        require(sha(inside(w, relative)) == digest, 'Context source changed')
    for relative, digest in parent_frozen['source_sha256'].items():
        require(sha(inside(parent, relative)) == digest, 'Parent source changed')
    audit, paudit = read(args.audit), read(args.parent_audit)
    check_audit(audit, 9, SCIENCE_SHA, sha(f))
    check_audit(paudit, 45, PARENT_SCIENCE_SHA, sha(pf))
    require(audit['audit_script_sha256'] == sha(args.audit_script), 'Independent context auditor changed')
    require(paudit['audit_script_sha256'] == sha(args.parent_audit_script), 'Independent parent auditor changed')
    require(audit['parent_audit_sha256'] == sha(args.parent_audit), 'Wrong independently audited parent')
    require(audit['input_bindings_sha256'] == sha(w / 'INPUTS_FROZEN.json'), 'Input bindings changed')
    require(audit['training_complete_lock_sha256'] == sha(root / 'TRAINING_COMPLETE_LOCK.json'), 'Completion lock changed')
    bindings = read(w / 'INPUTS_FROZEN.json')
    require(bindings['protocol_sha256'] == SCIENCE_SHA and bindings['freeze_sha256'] == sha(f), 'Input protocol/source changed')
    require(audit['overlay_sha256'] == bindings['input_sha256']['fine_overlay'], 'Wrong fine overlay')
    require(paudit['selection_lock_sha256'] == PARENT_LOCK_SHA and paudit['selected_configuration_id'] == PARENT_IDS['selected'], 'Wrong parent comparator')
    # All audited outputs must still match, including weights and matrices. The
    # parser does not import torch or redundantly redo the independent GPU audit.
    check_artifacts(root, audit['artifact_sha256'])
    check_artifacts(proot, paudit['artifact_sha256'])
    amap = {(r['seed'], r['configuration_id']): r for r in audit['runs']}
    pmap = {(r['seed'], r['configuration_id']): r for r in paudit['runs']}
    require(len(amap) == 9 and set(amap) == {(s, c) for s in SEEDS for c in CONTEXT_IDS}, 'Missing/duplicate context audit run')
    control = PARENT_IDS['selected'].replace('_T1_', '_T0_')
    require(len(pmap) == 45 and set(pmap) == {(s, c) for s in SEEDS for c in [*pcfg, control]}, 'Missing/duplicate parent audit run')
    expected_artifacts = set()
    for seed in SEEDS:
        for name in CONTEXT_IDS:
            expected_artifacts.update(f'seed_{seed}/{name}/{f}' for f in ('head.pt', 'development.json', 'development_scores.pt', 'training_progress.json'))
            expected_artifacts.update(f'seed_{seed}/internal_check/{name}/{f}' for f in ('internal_check.json', 'internal_check_scores.pt'))
        expected_artifacts.add(f'seed_{seed}/context48_A0T1/parent_A0_anchor.json')
    require(set(audit['artifact_sha256']) == expected_artifacts, 'Independent context audit did not bind every artifact')
    populations = {}
    for stage, basename in [('development', 'development_ids.json'), ('internal_check', 'internal_check_ids.json'), ('train', 'train_ids.json')]:
        path = w / 'plans/splits' / basename
        require(sha(path) == plan['input_bindings']['repair_split_file_sha256'][basename], 'Split changed')
        require(sha(path) == sha(parent / 'plans/splits' / basename), 'Parent/context split differs')
        ids = read(path)
        require(len(ids) == len(set(ids)) == ({'train': 6610}.get(stage, N.get(stage))), 'Wrong split population')
        populations[stage] = set(ids)
    require(not populations['train'] & populations['development'] and not populations['train'] & populations['internal_check']
            and not populations['development'] & populations['internal_check'], 'Splits overlap')
    records, identities, raw, sources, anchor_reports = {}, {}, {}, {}, []
    completion = read(root / 'TRAINING_COMPLETE_LOCK.json')
    require(completion['selection_performed'] is False, 'Context must not reselect')
    require(completion['protocol_sha256'] == SCIENCE_SHA and completion['freeze_sha256'] == sha(f)
            and completion['input_bindings_sha256'] == sha(w / 'INPUTS_FROZEN.json'), 'Completion lock identity mismatch')
    for seed in SEEDS:
        train_manifest = read(root / f'seed_{seed}/manifest.json')
        pm = read(proot / f'seed_{seed}/manifest.json')
        require(completion['training_manifest_sha256'][str(seed)] == sha(root / f'seed_{seed}/manifest.json'), 'Completion lock does not bind final train manifest')
        for parent_stage, relative in [('train', f'seed_{seed}/manifest.json'), ('selected-control', f'seed_{seed}/selected_control/manifest.json'), ('check', f'seed_{seed}/internal_check/manifest.json')]:
            require(paudit['manifest_sha256'][f'{seed}/{parent_stage}'] == sha(proot / relative), 'Parent stage manifest changed')
        for field in ('base_initial_state_sha256', 'initial_pairs_sha256', 'text_embeddings_sha256'):
            require(train_manifest[field] == pm[field], 'Parent/context pairing mismatch: ' + field)
        for name in CONTEXT_IDS:
            require(amap[seed, name]['epoch_order_sha256'] == pmap[seed, PARENT_IDS['matched_A0']]['epoch_order_sha256'], 'Paired epoch order differs')
            require(completion['all_training_artifacts_sha256'][f'{seed}/{name}'] == train_manifest['runs'][name]['artifacts_sha256'], 'Training completion lock artifact mismatch')
        for stage in STAGES:
            relative = f'seed_{seed}/' + ('manifest.json' if stage == 'development' else 'internal_check/manifest.json')
            label = f'{seed}/' + ('train' if stage == 'development' else 'check')
            manifest = read(root / relative)
            require(sha(root / relative) == audit['manifest_sha256'][label], 'Context manifest changed')
            require(manifest['status'] == ('complete_training_and_development' if stage == 'development' else 'complete_internal_check'), 'Context stage incomplete')
            require(set(manifest['runs']) == set(CONTEXT_IDS), 'Unexpected stage grid')
            identity = manifest['identity']
            require(identity['protocol_sha256'] == SCIENCE_SHA and identity['freeze_sha256'] == sha(f)
                    and identity['input_bindings_sha256'] == sha(w / 'INPUTS_FROZEN.json') and identity['overlay_sha256'] == audit['overlay_sha256'], 'Stage bindings mismatch')
            require(identity['seed'] == seed and identity['stage'] == ('train' if stage == 'development' else 'check'), 'Wrong stage/seed')
            if stage == 'internal_check':
                require(identity['training_complete_lock_sha256'] == sha(root / 'TRAINING_COMPLETE_LOCK.json'), 'Internal evaluation predates completion lock')
            for name in CONTEXT_IDS:
                run = manifest['runs'][name]
                require(run['status'] == 'complete', 'Context run incomplete')
                rel_dir = f'seed_{seed}/' + ('' if stage == 'development' else 'internal_check/') + name
                for filename, digest in run['artifacts_sha256'].items():
                    require(audit['artifact_sha256'].get(rel_dir + '/' + filename) == digest, 'Stage/audit artifact differs')
                path = root / rel_dir / (stage + '.json')
                r = read(path); ar = amap[seed, name]
                require(r['input_bindings_sha256'] == sha(w / 'INPUTS_FROZEN.json') and r['overlay_sha256'] == audit['overlay_sha256'], 'Evaluation overlay/input mismatch')
                m, ids = validate_evaluation(r, name=name, config=configs[name], seed=seed, stage=stage, ids=populations[stage], protocol_sha=SCIENCE_SHA, freeze_sha=sha(f), head_sha=ar['head_sha256'])
                require(m == ar[stage + '_metrics_percent'], 'Context audit metric differs')
                if stage not in identities:
                    identities[stage] = ids
                require(identities[stage] == ids, 'Context query/caption/gallery ordering differs')
                records[stage, seed, name], raw[stage, seed, name] = m, r
                sources[str(path)] = sha(path)
            for alias, name in PARENT_IDS.items():
                path = proot / f'seed_{seed}' / ('' if stage == 'development' else 'internal_check') / name / (stage + '.json')
                r, ar = read(path), pmap[seed, name]
                m, ids = validate_evaluation(r, name=name, config=pcfg[name], seed=seed, stage=stage, ids=populations[stage], protocol_sha=PARENT_SCIENCE_SHA, freeze_sha=sha(pf), head_sha=ar['head_sha256'])
                require(m == ar[stage + '_metrics_percent'], 'Parent audit metric differs')
                require(identities[stage] == ids, 'Parent/context query/caption/gallery ordering differs')
                records[stage, seed, 'parent/' + alias], raw[stage, seed, 'parent/' + alias] = m, r
                sources[str(path)] = sha(path)
            require(raw[stage, seed, 'context48_A0T1']['ranks'] == raw[stage, seed, 'parent/matched_A0']['ranks'], 'A0 anchor ranks differ on ' + stage)
        anchor = read(root / f'seed_{seed}/context48_A0T1/parent_A0_anchor.json')
        require(anchor['all_parameter_tensors_bitwise_equal'] is True and anchor['development_ranks_identical'] is True, 'A0 producer anchor failed')
        require(anchor['parent_head_sha256'] == pmap[seed, PARENT_IDS['matched_A0']]['head_sha256'], 'A0 parent head differs')
        require(anchor['parent_development_sha256'] == sources[str(proot / f'seed_{seed}' / PARENT_IDS['matched_A0'] / 'development.json')], 'A0 parent evaluation differs')
        anchor_reports.append({'seed': seed, 'parameter_tensors_bitwise_equal': True,
                               'development_ranks_identical': True, 'internal_check_ranks_identical': True,
                               'context_head_sha256': amap[seed, 'context48_A0T1']['head_sha256'],
                               'parent_head_sha256': pmap[seed, PARENT_IDS['matched_A0']]['head_sha256']})
    # Independently audited anchors, not only the producer's self-report.
    aa = {x['seed']: x for x in audit['parent_a0_anchors']}
    require(set(aa) == set(SEEDS), 'Missing independent A0 anchor')
    for seed in SEEDS:
        require(all(aa[seed][k] is True for k in ('all_parameter_tensors_bitwise_equal', 'development_ranks_identical', 'internal_check_ranks_identical')), 'Independent A0 anchor failed')
    provenance = {'protocol_sha256': SCIENCE_SHA, 'parent_protocol_sha256': PARENT_SCIENCE_SHA,
                  'context_freeze_sha256': sha(f), 'parent_freeze_sha256': sha(pf),
                  'context_audit_sha256': sha(args.audit), 'parent_audit_sha256': sha(args.parent_audit),
                  'context_auditor_sha256': sha(args.audit_script), 'parent_auditor_sha256': sha(args.parent_audit_script),
                  'input_bindings_sha256': sha(w / 'INPUTS_FROZEN.json'), 'overlay_sha256': audit['overlay_sha256'],
                  'training_complete_lock_sha256': sha(root / 'TRAINING_COMPLETE_LOCK.json'),
                  'parent_selection_lock_sha256': PARENT_LOCK_SHA, 'analysis_source_sha256': sha(__file__),
                  'consumed_evaluation_sha256': sources, 'context_artifacts_verified': len(audit['artifact_sha256']),
                  'parent_artifacts_verified': len(paudit['artifact_sha256'])}
    return plan, records, provenance, anchor_reports


def save_csv(path, rows):
    cooked = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cooked[0]))
        writer.writeheader(); writer.writerows(cooked)


def write_outputs(out, result):
    require(not out.exists(), 'Use a new output directory')
    out.mkdir(parents=True)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for key in ('all_runs', 'configuration_summaries', 'paired_contrasts', 'contrast_summaries'):
        save_csv(out / (key + '.csv'), result[key])
    lines = ['# Complete fixed context48 study', '',
             'All 9 context heads, 18 evaluations, and all five fixed Phase 1 comparator groups are included.',
             'Three paired seeds; SD is sample SD, not a confidence interval. Engineering criteria are descriptive and use unrounded effects. No p-values or candidate selection.',
             'The old 918-image validation set is development data. The 1,653-image internal check was excluded from these new fits, but is not external or historically unseen data.',
             'Context48 and the selected Phase 1 comparator use constant reliability. The original-R gate-only comparator retains the original reliability prior; these are different changes.', '',
             '| Split | Contrast | ΔR@10 mean ± SD (pp) | Seed 0 / 1 / 2 | ΔMRR@10 mean (pp) | Rule |',
             '|---|---|---:|---|---:|---|']
    for r in result['contrast_summaries']:
        values = ' / '.join(f'{v:+.6f}' for v in r['R@10_per_seed_pp'])
        lines.append(f"| {r['stage']} | {r['contrast']} ({r['kind']}) | {r['R@10_mean_pp']:+.6f} ± {r['R@10_sample_sd_pp']:.6f} | {values} | {r['MRR@10_mean_pp']:+.6f} | {'pass' if r['engineering_rule']['pass'] else 'fail'} |")
    lines += ['', 'Each mechanism claim requires its own matched contrast to pass on both splits. Failure of one contrast cannot be replaced by a positive comparator result. A0 must reproduce the Phase 1 matched A0 parameter tensors and both rank lists exactly; checkpoint files need not be byte-identical.', '',
              'All-three-primary engineering outcome: ' + ('PASS' if result['all_three_primary_pass_both_splits'] else 'FAIL') + '.']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    # Self-contained table fragments; no automatic emphasis of favorable values.
    esc = lambda s: str(s).replace('_', r'\_').replace('%', r'\%')
    tex = [r'% All values are percentages; paired effects are percentage points.', r'\begin{longtable}{llrrrr}', r'\hline Split & Configuration & R@1 & R@5 & R@10 & MRR@10 \\ \hline']
    for r in result['configuration_summaries']:
        cells = [f"{r[k + '_mean']:.3f} $\\pm$ {r[k + '_sample_sd']:.3f}" for k in METRICS]
        tex.append(' & '.join([esc(r['stage']), esc(r['configuration']), *cells]) + r' \\')
    tex += [r'\hline\end{longtable}', r'\begin{longtable}{llrrrrl}', r'\hline Split & Contrast & $\Delta$R@1 & $\Delta$R@5 & $\Delta$R@10 & $\Delta$MRR@10 & Rule \\ \hline']
    for r in result['contrast_summaries']:
        cells = [f"{r[k + '_mean_pp']:+.3f} $\\pm$ {r[k + '_sample_sd_pp']:.3f}" for k in METRICS]
        tex.append(' & '.join([esc(r['stage']), esc(r['contrast']), *cells, 'pass' if r['engineering_rule']['pass'] else 'fail']) + r' \\')
    tex += [r'\hline\end{longtable}']
    (out / 'all_tables.tex').write_text('\n'.join(tex) + '\n')
    manifest = {'schema': 'pare.context48-analysis-manifest.v1', 'provenance': result['provenance'],
                'outputs_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'parent', 'audit', 'parent-audit', 'audit-script', 'parent-audit-script', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    require(not args.out.exists(), 'Output must be new; no partial/overwrite results')
    plan, records, provenance, anchors = load_complete(args)
    result = {'schema': 'pare.context48-complete-analysis.v1', 'status': 'complete_verified',
              'selection_performed': False, 'official_test_accessed': False,
              'new_training_performed': False, 'thresholds': plan['acceptance'],
              'scope': 'Exploratory development and internal check only; no external blind or statistical significance claim.',
              'provenance': provenance, 'parent_A0_anchors': anchors,
              **summarize(records, plan['acceptance'])}
    write_outputs(args.out, result)
    print(json.dumps({'status': result['status'], 'primary_acceptance': result['primary_acceptance'],
                      'manifest_sha256': sha(args.out / 'manifest.json')}))


if __name__ == '__main__':
    main()
