#!/usr/bin/env python3
"""Independent complete-case statistics audit; never selects or fits a model."""
import argparse
import json
import math
from pathlib import Path

from scipy.stats import t
from audit_core import (TAGS, VARIANTS, SCALES, expected_evaluation_keys,
                        expected_run_keys, metrics_from_ranks, require, sha)

METRICS = ('R@1','R@5','R@10','MRR@10')


def read(path):
    return json.loads(Path(path).read_text())


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)


def moments(values):
    require(len(values)==5 and all(math.isfinite(v) for v in values), 'Five finite paired seeds required')
    mean=math.fsum(values)/5
    return mean,math.sqrt(math.fsum((v-mean)**2 for v in values)/4)


def contrast(delta,mrr):
    mean,sd=moments(delta); mrr_mean,_=moments(mrr); se=sd/math.sqrt(5)
    raw=[mean-float(t.isf(.025,4))*se,mean+float(t.isf(.025,4))*se]
    family=[mean-float(t.isf(.0025,4))*se,mean+float(t.isf(.0025,4))*se]
    all_positive=all(value>0 for value in delta)
    return dict(mean=mean,sample_sd=sd,t95_interval=raw,bonferroni_family95_interval=family,
                mean_MRR10_delta=mrr_mean,all_R10_positive=all_positive,
                stable_positive_evidence=bool(all_positive and family[0]>0 and mrr_mean>=0))


def verify_rows(actual,expected,keys):
    mapping={tuple(row[k] for k in keys):row for row in actual}
    target={tuple(row[k] for k in keys):row for row in expected}
    require(len(mapping)==len(actual)==len(expected) and set(mapping)==set(target), 'Wrong summary identity set')
    for identity, row in target.items():
        supplied=mapping[identity]
        require(set(supplied)==set(row), 'Summary field set differs')
        for key,value in row.items():
            require((supplied[key]==value if key in keys else close(supplied[key],value)), 'Summary mismatch: '+str(identity)+'/'+key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--verification',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    require(not args.output.exists(), 'Use a new proof file')
    audit,analysis=read(args.verification),read(args.analysis)
    require(audit['status']=='pass' and audit['heads']==75 and audit['evaluations']==105 and
            len(audit['runs'])==75 and len(audit['legacy_anchors'])==10 and len(audit['baselines'])==7,
            'Complete independent numerical audit required before statistics')
    require(analysis['status']=='all_105_results_rank_verified' and analysis['n_heads']==75 and analysis['n_evaluations']==105,
            'Incomplete frozen analysis')
    binding={key:audit[key] for key in ('protocol_sha256','freeze_sha256')}
    require(analysis['bindings']==binding and binding['protocol_sha256']==sha(root/'protocol.json') and
            binding['freeze_sha256']==sha(root/'SOURCE_FROZEN.json'), 'Wrong frozen study binding')
    plan,frozen=read(root/'protocol.json'),read(root/'SOURCE_FROZEN.json')
    require(plan['seeds']==list(range(5)) and plan['statistics']['family_size']==10 and
            analysis['source_sha256']==frozen['source_sha256']['analysis_scripts/analyze.py']==sha(root/'analysis_scripts/analyze.py'),
            'Changed seeds/statistics/analysis source')
    primary={(row['backbone'],row['left'],row['right']) for row in plan['primary_contrasts']}
    require(len(plan['primary_contrasts'])==10 and primary=={(tag,'strict',control) for tag in TAGS for control in ('full','strict_token_off')},
            'Changed primary comparison grid')
    aruns={(row['backbone'],row['seed'],row['variant']):row for row in audit['runs']}
    require(set(aruns)==set(expected_run_keys()), 'Wrong audited run grid')
    input_paths={f'results/{tag}/seed_{seed}/{variant}/{label}.json' for tag,seed,variant,label in expected_evaluation_keys()}
    require(set(analysis['result_input_sha256'])==input_paths, 'Analysis omitted/added result inputs')
    records,metrics={},{}
    checked_hashes={}
    for tag,seed,variant,label in expected_evaluation_keys():
        relative=f'results/{tag}/seed_{seed}/{variant}/{label}.json'; path=root/relative; digest=sha(path)
        require(digest==analysis['result_input_sha256'][relative]==audit['artifact_sha256'][relative], 'Changed analyzed rank file')
        record=read(path)
        require((record['backbone'],record['seed'],record['variant'],record['evaluation_label'])==(tag,seed,variant,label),
                'Wrong record label')
        calculated=metrics_from_ranks(record['ranks'])
        require(all(close(calculated[key],record['metrics_percent'][key]) and
                    close(calculated[key],aruns[tag,seed,variant]['metrics'][label][key]) for key in METRICS),
                'Independent rank/metric disagreement')
        records[tag,seed,variant,label]=record;metrics[tag,seed,variant,label]=calculated;checked_hashes[relative]=digest
    summary=[]
    for tag in TAGS:
        for variant in VARIANTS:
            row=dict(backbone=tag,variant=variant)
            for metric in METRICS:
                mean,sd=moments([metrics[tag,seed,variant,'test'][metric] for seed in range(5)])
                row[metric+'_mean']=mean;row[metric+'_sd']=sd
            summary.append(row)
    verify_rows(analysis['summaries'],summary,('backbone','variant'))
    supplied={(r['backbone'],r['comparison']):r for r in analysis['contrasts']}
    require(len(supplied)==len(analysis['contrasts'])==10, 'Wrong primary comparison count')
    contrasts=[]
    for tag in TAGS:
        for control in ('full','strict_token_off'):
            name='strict_minus_'+control; actual=supplied[tag,name]
            delta={metric:[metrics[tag,s,'strict','test'][metric]-metrics[tag,s,control,'test'][metric] for s in range(5)]
                   for metric in METRICS}
            calculated=contrast(delta['R@10'],delta['MRR@10'])
            require(len(actual['all_seeds'])==5 and all(close(a,b) for a,b in zip(actual['all_seeds'],delta['R@10'])),
                    'Wrong paired R10 seed differences')
            require(set(actual['all_metric_paired_differences'])==set(METRICS), 'Missing secondary paired differences')
            for metric in METRICS:
                values=actual['all_metric_paired_differences'][metric]
                require(len(values)==5 and all(close(a,b) for a,b in zip(values,delta[metric])), 'Wrong paired metric differences')
            for key,value in calculated.items():
                if isinstance(value,list):
                    require(len(actual[key])==2 and all(close(a,b) for a,b in zip(actual[key],value)), 'Wrong confidence interval')
                elif isinstance(value,bool):require(actual[key] is value,'Wrong stable-positive evidence flag')
                else:require(close(actual[key],value),'Wrong comparison moment')
            contrasts.append(dict(backbone=tag,comparison=name,**calculated,all_metric_paired_differences=delta,verified=True))
    across={control:all(row['stable_positive_evidence'] for row in contrasts if row['comparison']=='strict_minus_'+control)
            for control in ('full','strict_token_off')}
    require(analysis['stable_across_all_five_backbones']==across,'Wrong across-backbone inference')
    scales=[]
    for label in SCALES:
        for variant in VARIANTS:
            row=dict(backbone='vitb32',scale=label,variant=variant)
            for metric in METRICS:
                values=[metrics['vitb32',s,variant,label][metric] for s in range(5)]
                differences=[metrics['vitb32',s,variant,label][metric]-metrics['vitb32',s,variant,'test'][metric] for s in range(5)]
                mean,sd=moments(values); change,change_sd=moments(differences)
                row.update({metric+'_mean':mean,metric+'_sd':sd,metric+'_change_mean':change,metric+'_change_sd':change_sd})
            scales.append(row)
    verify_rows(analysis['scales'],scales,('backbone','scale','variant'))
    group_path=root/plan['r2_bindings']['groups_file'];require(sha(group_path)==plan['r2_bindings']['groups_sha256'],'Changed fixed group definitions')
    groups=read(group_path)['groups'];subgroups=[]
    for group,members in groups.items():
        require(len(members)==len(set(members))==plan['r2_bindings']['counts'][group],'Wrong fixed group size')
        members=set(members)
        for variant in VARIANTS:
            samples=[]
            for seed in range(5):
                record=records['vitb32',seed,variant,'test']
                ranks=[rank for image,rank in zip(record['query_image_ids'],record['ranks']) if image in members]
                require(len(ranks)==len(members),'Wrong grouped query population')
                samples.append(metrics_from_ranks(ranks))
            row=dict(group=group,n_queries=len(members),variant=variant)
            for metric in METRICS:
                mean,sd=moments([sample[metric] for sample in samples]);row[metric+'_mean']=mean;row[metric+'_sd']=sd
            subgroups.append(row)
    require(len(subgroups)==12,'Wrong subgroup grid');verify_rows(analysis['subgroups'],subgroups,('group','variant'))
    abases={(row['backbone'],row['label']):row['metrics'] for row in audit['baselines']};baselines=[]
    for tag in TAGS:
        for label in (SCALES if tag=='vitb32' else ('test',)):
            relative=f'baselines/{tag}/{label}.json';path=root/relative;digest=sha(path)
            require(digest==audit['artifact_sha256'][relative],'Changed baseline file')
            record=read(path);calculated=metrics_from_ranks(record['ranks'])
            require(record['backbone']==tag and record['evaluation_label']==label and record['variant']=='scene' and record['seed'] is None,
                    'Wrong seedless baseline identity')
            require(all(close(calculated[key],record['metrics_percent'][key]) and close(calculated[key],abases[tag,label][key]) for key in METRICS),
                    'Baseline rank/metric mismatch')
            baselines.append(dict(backbone=tag,label=label,metrics=calculated));checked_hashes[relative]=digest
    require(len(baselines)==7,'Incomplete baseline verification')
    require(all(sha(root/name)==digest for name,digest in checked_hashes.items()),'Input changed during statistics verification')
    report=dict(status='independent_analysis_verified',selection_performed=False,analysis_sha256=sha(args.analysis),
                verification_sha256=sha(args.verification),comparisons=contrasts,summaries=summary,scales=scales,subgroups=subgroups,
                baselines=baselines,counts=dict(summaries=15,comparisons=10,scales=9,subgroups=12,baselines=7),
                protocol_sha256=binding['protocol_sha256'],freeze_sha256=binding['freeze_sha256'],
                audit_source_sha256=sha(__file__),input_sha256=checked_hashes,
                interpretation='Paired training seeds on fixed previously used data; five-seed t approximation, family10; no new-data generalization or equivalence inferred.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],sha256=sha(args.output),output=str(args.output))))


if __name__=='__main__':main()
