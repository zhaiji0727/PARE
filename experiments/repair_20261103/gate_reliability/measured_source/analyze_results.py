"""Independent rank-based analysis; never selects a model or edits results."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

METRICS = ('R@1', 'R@5', 'R@10', 'MRR@10')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(ranks):
    n=len(ranks)
    return {**{f'R@{k}':100*sum(r<=k for r in ranks)/n for k in (1,5,10)},
            'MRR@10':100*sum(1/r if r<=10 else 0 for r in ranks)/n}

def save_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stage',choices=['development','internal_check'],required=True)
    ap.add_argument('--include-control',action='store_true');args=ap.parse_args()
    plan=json.loads(args.protocol.read_text()); configs={x['id']:x for x in plan['configurations']}
    lock_path=args.results/'SELECTION_LOCK.json';lock=json.loads(lock_path.read_text()) if lock_path.exists() else None
    if args.include_control:
        assert lock is not None
        control_dirs=sorted((args.results/'seed_0/selected_control').glob('*/development.json'))
        assert len(control_dirs)==1
        control=json.loads(control_dirs[0].read_text())['configuration'];configs[control['id']]=control
    n=918 if args.stage=='development' else 1653
    records={};rows=[];sources=[];identity=None
    for seed in plan['seeds']:
        for name,config in configs.items():
            d=args.results/f'seed_{seed}'
            if args.stage=='internal_check': d=d/'internal_check'/name
            elif config.get('matched_selected_control'): d=d/'selected_control'/name
            else: d=d/name
            path=d/(args.stage+'.json');r=json.loads(path.read_text())
            assert r['configuration']==config and r['seed']==seed,(seed,name)
            assert r['n_queries']==r['n_gallery']==n
            assert r['official_test_accessed'] is False
            assert r['protocol_sha256']==sha(args.protocol)
            assert len(r['ranks'])==n and all(type(x)==int and 1<=x<=n for x in r['ranks'])
            assert len(set(r['query_ids']))==len(set(r['gallery_ids']))==n
            assert set(r['query_ids'])==set(r['gallery_ids'])
            assert [r['gallery_ids'][i] for i in r['gold_gallery_index']]==r['query_ids']
            current=[r['query_ids'],r['query_caption_sha256'],r['gallery_ids'],r['gold_gallery_index']]
            if identity is None: identity=current
            assert current==identity
            m=metrics(r['ranks'])
            assert all(abs(m[k]-r['metrics_percent'][k])<1e-10 for k in METRICS)
            records[(name,seed)]=m
            rows.append({'configuration':name,'seed':seed,**config['config'],**m})
            sources.append({'file':str(path),'sha256':sha(path),'head_sha256':r['checkpoint_sha256']})
    summary=[]
    for name,config in configs.items():
        item={'configuration':name,**config['config']}
        for k in METRICS:
            v=[records[(name,s)][k] for s in plan['seeds']]
            item[k+'_mean']=statistics.mean(v);item[k+'_sample_sd']=statistics.stdev(v)
        summary.append(item)
    def lookup(A,B,C,R,T='original',token=1):
        matching=[k for k,c in configs.items() if c['config']==dict(A=A,B=B,C=C,T=token,fusion='original',reliability=R,token_mode=T)]
        assert len(matching)==1,matching
        return matching[0]
    comparisons=list(plan['paired_contrasts'])
    def add(label,positive,negative):
        comparisons.append({'id':label,'kind':'locked_selected_comparison','terms':[
            {'configuration_id':positive,'coefficient':1},
            {'configuration_id':negative,'coefficient':-1}]})
    if lock:
        selected=lock['selected_configuration_id'];config=configs[selected]['config']
        add('selected_vs_original_full',selected,lookup(1,1,1,'original'))
        add('selected_vs_original_strict',selected,lookup(0,0,0,'original'))
        add('selected_fine_contribution',selected,lookup(0,0,1,config['reliability'],config['token_mode']))
        if args.include_control: add('selected_token_contribution',selected,control['id'])
    contrast_rows=[];paired=[]
    for contrast in comparisons:
        label=contrast['id'];terms=contrast['terms']
        item={'contrast':label,'kind':contrast['kind'],'terms':json.dumps(terms)}
        for k in METRICS:
            differences=[sum(t['coefficient']*records[(t['configuration_id'],seed)][k] for t in terms) for seed in plan['seeds']]
            item[k+'_mean_pp']=statistics.mean(differences);item[k+'_sample_sd_pp']=statistics.stdev(differences)
            for seed,diff in zip(plan['seeds'],differences):
                paired.append({'contrast':label,'metric':k,'seed':seed,'difference_pp':diff})
        d10=[sum(t['coefficient']*records[(t['configuration_id'],seed)]['R@10'] for t in terms) for seed in plan['seeds']]
        item['descriptive_improvement_rule_pass']=(statistics.mean(d10)>=.5 and min(d10)>=-1e-10 and sum(d>1e-10 for d in d10)>=2 and item['MRR@10_mean_pp']>=-1e-10)
        item['R@10_per_seed']=json.dumps(d10)
        contrast_rows.append(item)
    args.out.mkdir(parents=True,exist_ok=False)
    save_csv(args.out/'all_runs.csv',rows);save_csv(args.out/'configuration_summary.csv',summary)
    save_csv(args.out/'paired_contrasts.csv',paired);save_csv(args.out/'contrast_summary.csv',contrast_rows)
    report={'stage':args.stage,'status':'rank_metrics_verified','n_runs':len(rows),'n_queries_per_run':n,
            'protocol_sha256':sha(args.protocol),'selection_lock_sha256':sha(lock_path) if lock else None,
            'selected':lock['selected_configuration_id'] if lock else None,
            'configuration_summary':summary,'contrasts':contrast_rows,'sources':sources,
            'scope':'Internal development/held-out checks on previously used dataset, not independent external confirmation. All paired SDs describe three seeds, not confidence intervals.'}
    (args.out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('stage','status','n_runs','n_queries_per_run','selected')},indent=2))

if __name__=='__main__': main()
