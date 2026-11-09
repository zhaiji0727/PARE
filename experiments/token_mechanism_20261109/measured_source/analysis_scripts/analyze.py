#!/usr/bin/env python3
"""Complete-case, rank-verified analysis of the fixed 15-head diagnostic."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
from scipy.stats import t
METRICS=('R@1','R@5','R@10','MRR@10')
VARIANTS=('conditional','constant_descriptor','token_off')
SPLITS=('development','internal_check')

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rank_metrics(ranks,n):
    if len(ranks)!=n or any(type(r) is not int or not 1<=r<=n for r in ranks):
        raise ValueError('Invalid rank vector')
    return {**{f'R@{k}':100*sum(r<=k for r in ranks)/n for k in (1,5,10)},
            'MRR@10':100*sum(1/r if r<=10 else 0 for r in ranks)/n}

def contrast(deltas,mrr):
    if len(deltas)!=5 or len(mrr)!=5 or not all(math.isfinite(x) for x in deltas+mrr):
        raise ValueError('Exactly five finite paired differences required')
    avg=statistics.mean(deltas); sd=statistics.stdev(deltas); se=sd/math.sqrt(5)
    ci=lambda probability:[avg-t.ppf(probability,4)*se,avg+t.ppf(probability,4)*se]
    ordinary=ci(.975); family=ci(1-.05/(2*4))
    flag=avg>=.5 and all(v>0 for v in deltas) and family[0]>0 and statistics.mean(mrr)>=0
    return {'seed_deltas_R10_pp':deltas,'mean_R10_delta_pp':avg,'sample_sd_R10_pp':sd,
            't95_interval_pp':ordinary,'bonferroni_family95_interval_pp':family,
            'mean_MRR10_delta_pp':statistics.mean(mrr),'all_seeds_R10_positive':all(v>0 for v in deltas),
            'predeclared_stronger_within_dataset_evidence':bool(flag)}

def analyze(root,out):
    root=Path(root);out=Path(out);plan=json.loads((root/'protocol.json').read_text())
    if plan['seeds']!=list(range(5)) or plan['variants']!=list(VARIANTS):raise ValueError('Wrong plan')
    inputs={};records={};identities={};bindings={key:set() for key in ('protocol_sha256','freeze_sha256','means_sha256')}
    for split,n in zip(SPLITS,(918,1653)):
        for seed in range(5):
            for variant in VARIANTS:
                p=root/'results'/f'seed_{seed}'/variant/f'{split}.json'
                d=json.loads(p.read_text());inputs[str(p.relative_to(root))]=digest(p)
                if d['seed']!=seed or d['variant']!=variant or d['n_queries']!=n or d['n_gallery']!=n:
                    raise ValueError(f'Identity/population mismatch: {p}')
                for k in bindings:
                    if not d[k]:raise ValueError('Missing provenance')
                    bindings[k].add(d[k])
                if not d['posthoc_exploratory'] or d['official_test_accessed']:raise ValueError('Scope mismatch')
                identity={k:d[k] for k in ('query_ids','query_caption_sha256','gallery_ids','gold_gallery_index')}
                if split in identities and identity!=identities[split]:raise ValueError('Unpaired populations')
                identities[split]=identity
                if len(set(d['query_ids']))!=n or set(d['query_ids'])!=set(d['gallery_ids']):raise ValueError('Duplicate/missing identity')
                for q,g in zip(d['query_ids'],d['gold_gallery_index']):
                    if d['gallery_ids'][g]!=q:raise ValueError('Wrong gold identity')
                values=rank_metrics(d['ranks'],n)
                if any(abs(values[m]-d['metrics_percent'][m])>1e-9 for m in METRICS):raise ValueError('Metric/rank mismatch')
                records[split,seed,variant]=values
    if any(len(v)!=1 for v in bindings.values()):raise ValueError('Mixed provenance')
    if next(iter(bindings['protocol_sha256']))!=digest(root/'protocol.json'):raise ValueError('Wrong protocol hash')
    if next(iter(bindings['freeze_sha256']))!=digest(root/'SOURCE_FROZEN.json'):raise ValueError('Wrong freeze hash')
    if next(iter(bindings['means_sha256']))!=digest(root/'training_descriptor_means.json'):raise ValueError('Wrong means hash')
    summaries=[];contrasts=[]
    for split in SPLITS:
        for variant in VARIANTS:
            row={'split':split,'variant':variant}
            for m in METRICS:
                vals=[records[split,s,variant][m] for s in range(5)]
                row[m+'_mean']=statistics.mean(vals);row[m+'_sample_sd']=statistics.stdev(vals)
            summaries.append(row)
        for control in VARIANTS[1:]:
            differences={m:[records[split,s,'conditional'][m]-records[split,s,control][m] for s in range(5)] for m in METRICS}
            row={'split':split,'comparison':f'conditional_minus_{control}',**contrast(differences['R@10'],differences['MRR@10']),
                 'all_metric_paired_differences_pp':differences}
            contrasts.append(row)
    conclusions={control:all(r['predeclared_stronger_within_dataset_evidence'] for r in contrasts if r['comparison']==f'conditional_minus_{control}') for control in VARIANTS[1:]}
    report={'status':'all_30_rank_vectors_verified','n_heads':15,'n_evaluations':30,'seeds':list(range(5)),
            'posthoc_exploratory':True,'official_test_accessed':False,'source_sha256':digest(Path(__file__)),
            'bindings':{k:next(iter(v)) for k,v in bindings.items()},'summaries':summaries,'contrasts':contrasts,
            'stronger_evidence_on_both_splits':conclusions,
            'interpretation':{'constant_descriptor':'Tests regional conditioning beyond branch-constant trainable augmentation within this recipe. Not physical scattering identification.',
                              'token_off':'Tests token module aggregate effect with original reliability retained.',
                              'limits':'Five seed pairs on fixed previously used data; Student-t normality approximation, not distribution-free confirmation or new-data generalization. No inference of independent fine-branch effect.'},
            'result_input_sha256':inputs}
    out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    rows=[]
    for r in contrasts:
        rows.append({k:v for k,v in r.items() if k!='all_metric_paired_differences_pp'})
    with (out/'contrasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['15次匹配训练：区域条件token机制诊断','完整核验30个排名向量，全部种子与预定比较保留。','']
    for r in contrasts:
        lo,hi=r['bonferroni_family95_interval_pp']
        lines.append(f"{r['split']} | {r['comparison']}: R@10差值 {r['mean_R10_delta_pp']:+.4f} ± {r['sample_sd_R10_pp']:.4f} 百分点；多重比较校正区间 [{lo:+.4f},{hi:+.4f}]；预定证据条件 {r['predeclared_stronger_within_dataset_evidence']}")
    lines+=['','未通过预定条件表示证据不足，并不等于证明没有效应。','两套数据均曾使用；结果是探索性匹配训练证据，不是独立新数据验证，也不证明真实散射类别或细粒度分支独立有益。']
    (out/'结果摘要.txt').write_text('\n'.join(lines)+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    r=analyze(args.root,args.out);print(json.dumps({'status':r['status'],'conclusions':r['stronger_evidence_on_both_splits']}))
