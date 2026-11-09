#!/usr/bin/env python3
"""Read-only independent audit of75 heads/105 matrices; never trains a model.

Only the hash-verified original SAQREHead is imported, for initial parameter
bytes. No experiment ranking, metrics, training or validation code is called.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path

from audit_core import (ABLATIONS, TAGS, SCALES, VARIANTS, epoch_order_hashes,
                        evaluation_identity, expected_run_keys, metrics_from_ranks,
                        objsha, replay_ranks, replay_legacy_full_gallery_ranks,
                        require, same_state_bytes, same_tensor_bytes, sha, state_sha,
                        training_identity)

SETTINGS = dict(epochs=10, batch_size=64, weight_decay=.0001, n_train_images=9181,
                n_train_pairs=31945, batches_per_epoch=500, tail_batch_pairs=9,
                optimizer='AdamW', precision='float32')
TEMPERATURES = dict(vitb32=.03, vitb16=.05, remoteclip=.05, siglip=.07, georsclip=.05)


def read(path):
    return json.loads(Path(path).read_text())


def metrics_equal(left, right):
    return set(left) == set(right) and all(abs(left[k] - right[k]) < 1e-9 for k in left)


def history_check(head, progress, run, seed):
    history = head['history']; orders = epoch_order_hashes(seed)
    require(len(history) == 10 and progress['history'] == history, 'Incomplete/inconsistent history')
    elapsed = -1
    for epoch, row in enumerate(history, 1):
        require((row['epoch'], row['batches'], row['pairs'], row['tail_batch_pairs'], row['pair_order_sha256']) ==
                (epoch, 500, 31945, 9, orders[epoch - 1]), 'Wrong training order/population')
        require(math.isfinite(row['mean_batch_loss']) and row['mean_batch_loss'] >= 0 and
                math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] >= elapsed, 'Invalid training progress')
        elapsed = row['elapsed_seconds']
    states = progress['epoch_parameter_states']
    require(len(states) == 10 and [r['epoch'] for r in states] == list(range(1, 11)), 'Missing epoch parameter hashes')
    require(states[-1]['parameter_state_sha256'] == run['final_parameter_sha256'] == state_sha(head['state_dict']),
            'Final parameter bytes differ')
    require(run['expected_pair_order_sha256'] == head['provenance']['expected_pair_order_sha256'] == orders,
            'Expected shuffle mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args(); root = args.workspace.resolve(); output = args.output.resolve()
    require(not output.exists() and root / 'results' not in output.parents, 'Audit output must be new and outside results')
    plan, frozen, bound = [read(root / name) for name in ('protocol.json', 'SOURCE_FROZEN.json', 'INPUTS_BOUND.json')]
    require(plan['schema'] == 'pare.simplified-matched-study.v1' and plan['seeds'] == list(range(5)) and
            plan['variants'] == ABLATIONS and set(plan['backbones']) == set(TAGS) and plan['expected_heads'] == 75,
            'Wrong fixed study')
    require(all(plan['training'][k] == v for k, v in SETTINGS.items()) and
            all(plan['training'][k] is False for k in ('amp', 'early_stopping', 'warm_start', 'new_hyperparameter_search')),
            'Training recipe differs')
    require(plan['evaluation']['main_evaluations'] == 75 and plan['evaluation']['additional_scale_evaluations'] == 30 and
            plan['data_history']['original_test_previously_seen'] is True and plan['data_history']['candidate_chosen_after_prior_results'] is True,
            'Population/history differs')
    ps, fs = sha(root / 'protocol.json'), sha(root / 'SOURCE_FROZEN.json')
    require(frozen['protocol_sha256'] == bound['protocol_sha256'] == ps and
            frozen['protocol_markdown_sha256'] == bound['protocol_markdown_sha256'] == sha(root / 'PROTOCOL.md') and
            frozen['inputs_bound_sha256'] == sha(root / 'INPUTS_BOUND.json'), 'Frozen binding changed')
    source_hashes = {name: sha(root / name) for name in frozen['source_sha256']}
    require(source_hashes == frozen['source_sha256'] == bound['source_sha256'], 'Frozen source changed')
    actual_py = {str(p.relative_to(root)) for folder in ('base_code','reference','code','scripts','tests','analysis_scripts')
                 for p in (root / folder).glob('*.py')}
    require(actual_py == {name for name in source_hashes if name.endswith('.py')}, 'Frozen Python source set differs')
    copied = read(root/'REFERENCE_COPY_PROVENANCE.json')
    require(all(source_hashes[name] == row['sha256'] for name,row in copied.items()), 'Reference copy differs')
    for backbone in plan['backbones'].values():
        require(all(source_hashes['base_code/'+name] == digest for name,digest in backbone['cache_signature']['source_sha256'].items()),
                'Original cache/code signature differs')
    r2 = plan['r2_bindings']; groups = read(root / r2['groups_file'])['groups']
    require(sha(root / r2['groups_file']) == r2['groups_sha256'] and
            {k: len(v) for k, v in groups.items()} == r2['counts'], 'R2 fixed groups differ')
    require(all(sha(root / name) == digest for name, digest in r2['scale_manifests'].items()), 'Scale manifests differ')
    entries = [('common/' + key, value) for key, value in bound['common'].items()]
    entries += [(tag + '/' + key, value) for tag in TAGS for key, value in bound['backbones'][tag]['files'].items()]
    input_hashes = {key: sha(value['path']) for key, value in entries}
    require(all(input_hashes[key] == value['sha256'] for key, value in entries), 'Input file changed')
    common_paths = {key: Path(value['path']) for key, value in bound['common'].items()}
    fixed_groups = read(root/r2['groups_file']); original_groups = read(common_paths['spacing_groups'])
    for key in ('test_ids','test_ids_sha256','n_gallery_retained_for_all_subgroups','rounding',
                'bin_edges_m_per_pixel_after_rounding','groups','nominal_spacing_m_by_id','counts'):
        require(fixed_groups[key] == original_groups[key], 'Original R2 group definition differs: '+key)
    audit_provenance = read(common_paths['raster_spacing_audit_provenance'])
    require(audit_provenance['audit_sha256'] == input_hashes['common/raster_spacing_audit'] == fixed_groups['source_audit_sha256'],
            'R2 copied audit provenance differs')
    require(fixed_groups['source_grouping_code_sha256'] == input_hashes['common/spacing_groups_source'] and
            fixed_groups['source_queries_sha256'] == input_hashes['vitb32/legacy_queries'], 'R2 source code/query binding differs')
    train_ids, test_ids = read(common_paths['train_ids']), read(common_paths['test_ids'])
    require(len(train_ids) == len(set(train_ids)) == 9181 and len(test_ids) == len(set(test_ids)) == 3688 and
            not set(train_ids) & set(test_ids), 'Wrong or overlapping split population')
    members = [name for values in groups.values() for name in values]
    require(len(members) == len(set(members)) == 3688 and set(members) == set(test_ids), 'Groups must partition test IDs')

    # Require every prespecified run/evaluation before reading performance values.
    require({p.name for p in (root/'results').iterdir() if p.is_dir()} == set(TAGS), 'Extra/missing backbone directories')
    artifacts, runs = {}, {}
    for tag in TAGS:
        require({p.name for p in (root/'results'/tag).iterdir() if p.is_dir()} == {f'seed_{s}' for s in range(5)},
                'Extra/missing seed directories')
    for tag, seed, variant in expected_run_keys():
        folder = root/'results'/tag/f'seed_{seed}'/variant
        require({p.name for p in folder.parent.iterdir() if p.is_dir()} == set(VARIANTS), 'Extra/missing variants')
        labels = SCALES if tag == 'vitb32' else ('test',)
        expected_files = {'head.pt','training_progress.json','run.json'} | {label + suffix for label in labels for suffix in ('.json','_scores.pt')}
        require({p.name for p in folder.iterdir() if p.is_file()} == expected_files, 'Extra/missing run files')
        run = read(folder/'run.json'); identity = dict(backbone=tag,seed=seed,variant=variant,protocol_sha256=ps,freeze_sha256=fs)
        require(run['identity'] == identity and run['training_status'] == 'complete' and set(run['evaluations']) == set(labels),
                'Run incomplete/wrong identity')
        require(set(run['training_artifacts_sha256']) == {'head.pt','training_progress.json'}, 'Wrong training artifact set')
        recorded = {**run['training_artifacts_sha256'], 'run.json':sha(folder/'run.json')}
        for label in labels:
            part = run['evaluations'][label]['artifacts_sha256']
            require(set(part) == {label+'.json',label+'_scores.pt'}, 'Wrong evaluation artifact set')
            recorded.update(part)
        require(all(sha(folder/name) == digest for name,digest in recorded.items()), 'Recorded run hash mismatch')
        artifacts.update({str((folder/name).relative_to(root)):digest for name,digest in recorded.items()})
        runs[tag,seed,variant] = run

    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    require(torch.cuda.is_available(), 'CUDA required to reproduce original tie ordering')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('independent_simplified_base', root/'base_code/saqre_phys.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    results, anchors, baseline_results = [], {}, []
    for tag in TAGS:
        b = plan['backbones'][tag]; info = bound['backbones'][tag]
        paths = {key:Path(value['path']) for key,value in info['files'].items()}
        require(b['lr'] == .002 and b['kappa'] == TEMPERATURES[tag], 'Backbone hyperparameters changed')
        train = torch.load(paths['train_cache'],map_location='cpu',weights_only=True,mmap=True)
        test = torch.load(paths['test_cache'],map_location='cpu',weights_only=True,mmap=True)
        dimension = train['D']
        for cache, ids in ((train,train_ids),(test,test_ids)):
            require(cache['ids'] == ids and cache['D'] == dimension and cache['Mc'] == 6 and cache['Mf'] == 8,
                    'Cache shape/population contract differs')
            require(cache['provenance']['signature'] == b['cache_signature'] and
                    cache['provenance']['signature_sha256'] == objsha(b['cache_signature']), 'Original cache signature differs')
        tids = training_identity(train_ids,common_paths['train_captions'])
        qids = evaluation_identity(test_ids,common_paths['test_captions'])
        require(tids['pairs_sha256'] == info['initial_pairs_sha256'] and qids == info['query_identity'], 'Preflight identity mismatch')
        tm = read(root/'shared'/tag/'manifest.json'); tp = root/'shared'/tag/'text_embeddings.pt'
        text = torch.load(tp,map_location='cpu',weights_only=True)
        ti = dict(backbone=tag,protocol_sha256=ps,freeze_sha256=fs,training_text_sha256=objsha(tids['unique_caption_sha256']),
                  test_text_sha256=objsha(qids['query_caption_sha256']))
        require(tm['identity'] == ti and all(text[k] == v for k,v in ti.items()) and tm['artifact_sha256'] == sha(tp),
                'Text artifact binding differs')
        for label,n in (('train',tids['n_unique_captions']),('test',3688)):
            require(tuple(text[label].shape) == (n,dimension) and text[label].dtype == torch.float32 and
                    bool(torch.isfinite(text[label]).all()) and tm[label+'_tensor_sha256'] == state_sha({label:text[label]}),
                    'Text tensor/hash differs')
        require(same_tensor_bytes(text['test'],torch.from_numpy(np.load(paths['legacy_text'],allow_pickle=False))),
                'Original test text vector bytes differ')
        for name in ('manifest.json','text_embeddings.pt'):
            artifacts[str((root/'shared'/tag/name).relative_to(root))] = sha(root/'shared'/tag/name)
        scenes = {}
        for label in (SCALES if tag == 'vitb32' else ('test',)):
            source_key = ('test' if label == 'test' else label) + '_cache'
            expected_sha = info['files'][source_key]['sha256']
            require(bound['evaluation_cache_sha256'][tag][label] == expected_sha, 'Scale/cache mapping differs')
            if label != 'test':
                scaled = torch.load(paths[source_key],map_location='cpu',weights_only=True,mmap=True)
                require(scaled['ids'] == test_ids and scaled['scale_test']['reference_cache_sha256'] == info['files']['test_cache']['sha256'] and
                        scaled['scale_test']['factor'] == ('1.5' if label == 'spacing_1p5' else '2'), 'Wrong scale population/factor')
                base_signature = dict(scaled['provenance']['signature']); base_signature.pop('synthetic_spacing_test')
                require(base_signature == b['cache_signature'] and scaled['scale_test']['image_records_sha256'] == objsha(scaled['scale_test']['image_records']),
                        'Scale altered base signature/image manifest')
                local_manifest = root/next(name for name in r2['scale_manifests'] if ('_1.5.' if label=='spacing_1p5' else '_2.') in name)
                require(sha(local_manifest) == info['files'][label+'_manifest']['sha256'], 'Local/remote scale manifest mismatch')
                del scaled
            base_path = root/'baselines'/tag/label
            sj = base_path.with_suffix('.json'); sm = base_path.parent/(label+'_scores.pt')
            report = read(sj); saved = torch.load(sm,map_location='cpu',weights_only=True)
            ci = dict(backbone=tag,seed=None,variant='scene',protocol_sha256=ps,freeze_sha256=fs,
                      condition=label,evaluation_label=label,evaluation_cache_sha256=expected_sha,
                      text_embeddings_sha256=tm['test_tensor_sha256'],**qids)
            require(all(report[k] == saved[k] == v for k,v in ci.items()) and report['scores_artifact_sha256'] == sha(sm),
                    'Scene artifact identity/hash differs')
            require(tuple(saved['scores'].shape) == (3688,3688), 'Scene matrix shape differs')
            ranks = replay_ranks(saved['scores'],qids['gold_gallery_index'],torch)
            metric = metrics_from_ranks(ranks)
            require(ranks == report['ranks'] and metrics_equal(metric,report['metrics_percent']), 'Scene rank/metric mismatch')
            scenes[label] = saved['scores']; artifacts[str(sj.relative_to(root))]=sha(sj);artifacts[str(sm.relative_to(root))]=sha(sm)
            baseline_results.append(dict(backbone=tag,label=label,metrics=metric))
        require(same_tensor_bytes(scenes['test'],torch.from_numpy(np.load(paths['legacy_scene'],allow_pickle=False))),
                'Original scene score bytes differ')
        for seed in range(5):
            torch.manual_seed(seed); initial = module.SAQREHead(dimension).state_dict(); initial_sha = state_sha(initial)
            require(len(initial) == 15, 'Original parameter count changed')
            for variant in VARIANTS:
                folder = root/'results'/tag/f'seed_{seed}'/variant; run = runs[tag,seed,variant]
                head = torch.load(folder/'head.pt',map_location='cpu',weights_only=True); p = head['provenance']
                settings = {**SETTINGS,'lr':.002,'kappa':TEMPERATURES[tag]}
                require((head['schema'],head['D'],head['backbone'],head['seed'],head['variant'],head['ablate'],head['n_images'],head['n_pairs']) ==
                        ('pare.simplified-head.v1',dimension,tag,seed,variant,ABLATIONS[variant],9181,31945) and head['settings'] == settings,
                        'Wrong checkpoint recipe')
                require(set(head['state_dict']) == set(initial), 'Extra/missing model parameters')
                for name,tensor in head['state_dict'].items():
                    require(tensor.shape == initial[name].shape and tensor.dtype == initial[name].dtype and bool(torch.isfinite(tensor).all()),
                            'Invalid checkpoint tensor: '+name)
                require(all(p[k] == v for k,v in run['identity'].items()) and p['input_files'] == info['files'] and
                        p['common_input_files'] == bound['common'] and p['cache_signature'] == b['cache_signature'] and p['train_ids'] == train_ids,
                        'Checkpoint provenance differs')
                require(p['initial_state_sha256'] == run['initial_state_sha256'] == initial_sha and
                        p['initial_pairs_sha256'] == tids['pairs_sha256'] and
                        p['text_embeddings_sha256'] == run['text_embeddings_sha256'] == tm['train_tensor_sha256'] and
                        p['shared_text_artifact_sha256'] == tm['artifact_sha256'], 'Initial state/text/pairing differs')
                require(p['environment']['packages']['torch'] == str(torch.__version__) and
                        p['environment']['tf32'] is False and p['environment']['deterministic_algorithms'] is True, 'Runtime contract differs')
                history_check(head,read(folder/'training_progress.json'),run,seed)
                row = dict(backbone=tag,seed=seed,variant=variant,head_sha256=sha(folder/'head.pt'),metrics={})
                for label,scene in scenes.items():
                    report=read(folder/(label+'.json')); saved=torch.load(folder/(label+'_scores.pt'),map_location='cpu',weights_only=True)
                    ci={**run['identity'],**qids,'condition':label,'evaluation_label':label,
                        'evaluation_cache_sha256':bound['evaluation_cache_sha256'][tag][label],'checkpoint_sha256':row['head_sha256'],
                        'text_embeddings_sha256':tm['test_tensor_sha256'],'posthoc_exploratory':True,'original_test_previously_seen':True}
                    require(all(report[k] == saved[k] == v for k,v in ci.items()) and report['n_queries'] == report['n_gallery'] == 3688 and
                            tuple(saved['scores'].shape) == (3688,3688), 'Evaluation identity/population differs')
                    require(report['scores_artifact_sha256'] == sha(folder/(label+'_scores.pt')), 'Inner score hash differs')
                    ranks=replay_legacy_full_gallery_ranks(saved['scores'],scene,qids['gold_gallery_index'],torch)
                    metrics=metrics_from_ranks(ranks)
                    require(ranks == report['ranks'] and metrics_equal(metrics,report['metrics_percent']), 'Independent ranking/metric mismatch')
                    row['metrics'][label]=metrics
                    if seed==0 and variant in ('full','strict') and label=='test':
                        old=torch.load(paths['legacy_head_'+variant],map_location='cpu',weights_only=True)
                        old_result=read(paths['legacy_result_'+variant]); old_scores=torch.from_numpy(np.load(paths['legacy_scores_'+variant],allow_pickle=False))
                        require(same_state_bytes(head['state_dict'],old['state_dict']) and
                                [r['mean_batch_loss'] for r in head['history']] == [r['mean_batch_loss'] for r in old['history']], 'Legacy parameter/loss anchor differs')
                        require(same_tensor_bytes(saved['scores'],old_scores) and ranks == [r['natural_PARE_rank'] for r in old_result['queries']],
                                'Legacy matrix/rank anchor differs')
                        anchors[tag+'/'+variant]=dict(parameters_identical=True,epoch_losses_identical=True,scores_identical=True,ranks_identical=True)
                    del saved
                results.append(row)
                print(json.dumps({'audited':tag,'seed':seed,'variant':variant}),flush=True)
        del train,test,text,scenes
    require(len(results)==75 and sum(len(r['metrics']) for r in results)==105 and len(anchors)==10 and len(baseline_results)==7,
            'Incomplete audit grid')
    require(all(sha(root/name)==digest for name,digest in artifacts.items()), 'Result changed during audit')
    require(all(sha(value['path'])==input_hashes[key] for key,value in entries), 'Input changed during audit')
    require(all(sha(root/name)==digest for name,digest in source_hashes.items()) and sha(root/'protocol.json')==ps and
            sha(root/'SOURCE_FROZEN.json')==fs, 'Source/freeze changed during audit')
    result=dict(status='pass',heads=75,evaluations=105,baselines=baseline_results,runs=results,legacy_anchors=anchors,
                protocol_sha256=ps,freeze_sha256=fs,inputs_bound_sha256=sha(root/'INPUTS_BOUND.json'),source_sha256=source_hashes,
                input_sha256=input_hashes,artifact_sha256=artifacts,original_test_previously_seen=True,
                verification_code_sha256={name:sha(Path(__file__).parent/name) for name in ('verify_completed.py','audit_core.py')},
                completed_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'pass','heads':75,'evaluations':105,'output':str(output),'sha256':sha(output)}))


if __name__=='__main__':main()
