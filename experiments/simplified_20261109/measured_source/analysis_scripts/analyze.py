#!/usr/bin/env python3
"""Complete-case analysis of 75 fixed heads and 105 full-gallery evaluations."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
from scipy.stats import t
METRICS=('R@1','R@5','R@10','MRR@10')
TAGS=('vitb32','vitb16','remoteclip','siglip','georsclip')
VARIANTS=('full','strict','strict_token_off')
SCALES=('test','spacing_1p5','spacing_2')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rank_metrics(ranks,gallery=3688):
 if not ranks or any(type(r) is not int or not 1<=r<=gallery for r in ranks):raise ValueError('Invalid ranks')
 n=len(ranks)
 return {**{f'R@{k}':100*sum(r<=k for r in ranks)/n for k in (1,5,10)},'MRR@10':100*sum(1/r if r<=10 else 0 for r in ranks)/n}
def summary(values):
 if len(values)!=5 or not all(math.isfinite(x) for x in values):raise ValueError('Exactly five finite seed values required')
 return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'all_seeds':values}
def contrast(deltas,mrr):
 s=summary(deltas);summary(mrr);se=s['sample_sd']/math.sqrt(5)
 def ci(q):return [s['mean']-float(t.ppf(q,4))*se,s['mean']+float(t.ppf(q,4))*se]
 adj=ci(.9975)
 return {**s,'t95_interval':ci(.975),'bonferroni_family95_interval':adj,'mean_MRR10_delta':statistics.mean(mrr),'all_R10_positive':all(x>0 for x in deltas),'stable_positive_evidence':bool(all(x>0 for x in deltas) and adj[0]>0 and statistics.mean(mrr)>=0)}
def expected_keys():
 return [(tag,s,v,scale) for tag in TAGS for s in range(5) for v in VARIANTS for scale in (SCALES if tag=='vitb32' else ('test',))]
def read_records(root):
 root=Path(root);plan=json.loads((root/'protocol.json').read_text())
 if plan['seeds']!=list(range(5)) or set(plan['variants'])!=set(VARIANTS) or set(plan['backbones'])!=set(TAGS):raise ValueError('Plan mismatch')
 bound=json.loads((root/'INPUTS_BOUND.json').read_text());cache_hashes=bound['evaluation_cache_sha256']
 records={};inputs={};identities={};bindings={'protocol_sha256':sha(root/'protocol.json'),'freeze_sha256':sha(root/'SOURCE_FROZEN.json')}
 for tag,s,v,scale in expected_keys():
  p=root/'results'/tag/f'seed_{s}'/v/f'{scale}.json';d=json.loads(p.read_text())
  if (d['backbone'],d['seed'],d['variant'],d['n_queries'],d['n_gallery'])!=(tag,s,v,3688,3688):raise ValueError(f'Wrong identity/count: {p}')
  if any(d[k]!=h for k,h in bindings.items()):raise ValueError('Wrong frozen provenance')
  if d['evaluation_label']!=scale or d['evaluation_cache_sha256']!=cache_hashes[tag][scale]:raise ValueError('Evaluation label/cache mismatch')
  ids={k:d[k] for k in ('query_ids','query_image_ids','query_caption_sha256','gallery_ids','gold_gallery_index')}
  if len(set(ids['query_ids']))!=3688 or len(ids['query_ids'])!=3688 or len(set(ids['gallery_ids']))!=3688 or len(ids['gallery_ids'])!=3688:raise ValueError('Duplicate or incomplete population')
  if len(d['ranks'])!=3688 or len(ids['gold_gallery_index'])!=3688 or len(ids['query_image_ids'])!=3688 or len(set(ids['query_image_ids']))!=3688:raise ValueError('Incomplete paired result')
  if any(type(g) is not int or not 0<=g<3688 or ids['gallery_ids'][g]!=q for q,g in zip(ids['query_image_ids'],ids['gold_gallery_index'])):raise ValueError('Gold identity mismatch')
  if tag in identities and ids!=identities[tag]:raise ValueError('Within-backbone unpaired identities')
  identities[tag]=ids
  m=rank_metrics(d['ranks'])
  if any(abs(m[k]-d['metrics_percent'][k])>1e-9 for k in METRICS):raise ValueError('Metric/rank mismatch')
  h=root/'results'/tag/f'seed_{s}'/v/'head.pt'
  if sha(h)!=d['checkpoint_sha256']:raise ValueError('Wrong trained head')
  records[tag,s,v,scale]=(d,m);inputs[str(p.relative_to(root))]=sha(p)
 if len(records)!=105:raise ValueError('Incomplete fixed study')
 if any(identities[k]['query_ids']!=identities[TAGS[0]]['query_ids'] or identities[k]['gallery_ids']!=identities[TAGS[0]]['gallery_ids'] for k in TAGS):raise ValueError('Different backbone populations')
 return records,inputs,bindings

def analyze(root,out):
 root=Path(root);out=Path(out);records,inputs,bindings=read_records(root)
 rows=[];contrasts=[];scales=[];subgroups=[]
 plan=json.loads((root/'protocol.json').read_text());gp=root/plan['r2_bindings']['groups_file']
 if sha(gp)!=plan['r2_bindings']['groups_sha256']:raise ValueError('Subgroup membership changed')
 groups=json.loads(gp.read_text())['groups']
 for tag in TAGS:
  for v in VARIANTS:
   row={'backbone':tag,'variant':v}
   for k in METRICS:
    a=summary([records[tag,s,v,'test'][1][k] for s in range(5)])
    row[k+'_mean']=a['mean'];row[k+'_sd']=a['sample_sd']
   rows.append(row)
  for control in ('full','strict_token_off'):
   delta={k:[records[tag,s,'strict','test'][1][k]-records[tag,s,control,'test'][1][k] for s in range(5)] for k in METRICS}
   contrasts.append({'backbone':tag,'comparison':'strict_minus_'+control,**contrast(delta['R@10'],delta['MRR@10']),'all_metric_paired_differences':delta})
 for scale in SCALES:
  for v in VARIANTS:
   row={'backbone':'vitb32','scale':scale,'variant':v}
   for k in METRICS:
    a=summary([records['vitb32',s,v,scale][1][k] for s in range(5)])
    change=summary([records['vitb32',s,v,scale][1][k]-records['vitb32',s,v,'test'][1][k] for s in range(5)])
    row[k+'_mean']=a['mean'];row[k+'_sd']=a['sample_sd'];row[k+'_change_mean']=change['mean'];row[k+'_change_sd']=change['sample_sd']
   scales.append(row)
 for label,names in groups.items():
  names=set(names)
  for v in VARIANTS:
   values=[]
   for seed in range(5):
    d=records['vitb32',seed,v,'test'][0]
    ranks=[r for q,r in zip(d['query_image_ids'],d['ranks']) if q in names]
    if len(ranks)!=len(names):raise ValueError('Incomplete subgroup')
    values.append(rank_metrics(ranks))
   row={'group':label,'n_queries':len(names),'variant':v}
   for m in METRICS:
    z=summary([x[m] for x in values]);row[m+'_mean']=z['mean'];row[m+'_sd']=z['sample_sd']
   subgroups.append(row)
 result={'subgroups':subgroups,'status':'all_105_results_rank_verified','n_heads':75,'n_evaluations':105,'bindings':bindings,'source_sha256':sha(__file__),'summaries':rows,'contrasts':contrasts,'scales':scales,'stable_across_all_five_backbones':{v:all(c['stable_positive_evidence'] for c in contrasts if c['comparison']=='strict_minus_'+v) for v in ('full','strict_token_off')},'interpretation':'Fixed previously used test set; n5 seed variability, small-sample t normality approximation; no new blind-test or equivalence claim. Secondary scale summaries are descriptive.','result_input_sha256':inputs}
 out.mkdir(parents=True,exist_ok=True);write(out/'analysis.json',result)
 for name,data in [('main_summary',rows),('scale_summary',scales),('subgroup_summary',subgroups)]:
  with (out/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 lines=['75头/105评估：固定简化方案匹配验证','数据已用于此前研究，本轮不构成新的独立盲测。','']
 for c in contrasts:
  lo,hi=c['bonferroni_family95_interval'];lines.append(f"{c['backbone']} {c['comparison']}: ΔR@10={c['mean']:+.4f}±{c['sample_sd']:.4f}pp; 调整区间[{lo:+.4f},{hi:+.4f}]; 稳定正向条件={c['stable_positive_evidence']}")
 (out/'结果摘要.txt').write_text('\n'.join(lines)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=analyze(a.root,a.out);print(json.dumps({'status':r['status'],'claims':r['stable_across_all_five_backbones']}))
