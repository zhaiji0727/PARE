"""Read-only numerical/identity audit of completed post-hoc JSON results."""
from pathlib import Path
import json, math, hashlib
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace');R=P/'report_revision_diagnostics_20261111';S=P/'simplified_confirm_20261109_v1'
issues=[];counts={};hashes={};sources={}
def read(p):
 b=p.read_bytes();sources[str(p.relative_to(P))]=hashlib.sha256(b).hexdigest();return json.loads(b)
def metric(r):
 return {**{f'R@{k}':100*sum(v<=k for v in r)/len(r) for k in [1,5,10]},'MRR@10':100*sum(1/v if v<=10 else 0 for v in r)/len(r)}
def check(label,got,expected):
 if got!=expected:issues.append({'label':label,'got':got,'expected':expected})
def metriccheck(label,got,expected):
 for k,v in expected.items():
  if not math.isclose(got[k],v,abs_tol=1e-9):issues.append({'label':label+':'+k,'got':got[k],'expected':v})
for mode,pattern in [('probe','*/seed_*.json'),('constant_training','*/seed_*/result.json'),('allcaptions','*/*seed_*.json')]:
 files=sorted((R/mode).glob(pattern));counts[mode]=len(files);hashes[mode]=set()
 for p in files:
  d=read(p);b=d['backbone'];seed=d['seed'];variant=d.get('variant','strict')
  if mode=='constant_training':variant='strict'
  old=read(S/f'results/{b}/seed_{seed}/{variant}/test.json')
  rr=d['ranks'];check(str(p)+':rank_count',len(rr),6190 if mode=='allcaptions' else 3688)
  check(str(p)+':unique_queryids',len(set(d['query_ids'])),len(rr))
  check(str(p)+':ranks_valid',all(type(r) is int and 1<=r<=3688 for r in rr),True)
  newm=metric(rr);oldm=metric(old['ranks'])
  metriccheck(str(p),d['constant_pi_metrics'] if mode=='probe' else d['metrics'],newm)
  if mode!='allcaptions':check(str(p)+':queryids',d['query_ids'],old['query_ids'])
  if mode=='probe':
   metriccheck(str(p)+':old',d['original_metrics'],oldm)
   metriccheck(str(p)+':delta',{'R@10':d['delta_R10_constant_minus_original']},{'R@10':newm['R@10']-oldm['R@10']})
  if mode=='constant_training':
   metriccheck(str(p)+':old',d['original_conditional_metrics'],oldm)
   metriccheck(str(p)+':delta',{'R@10':d['delta_R10_conditional_minus_constant']},{'R@10':oldm['R@10']-newm['R@10']})
   oldrun=read(S/f'results/{b}/seed_{seed}/strict/run.json')
   for key in ['initial_state_sha256','expected_pair_order_sha256']:check(str(p)+':'+key,d[key],oldrun[key])
  if mode=='allcaptions':
   lookup=dict(zip(d['query_ids'],rr))
   check(str(p)+':preserved_first_ranks',[lookup[x] for x in old['query_ids']],old['ranks'])
   check(str(p)+':extra_querycount',sum(not x.endswith('#caption=1') for x in d['query_ids']),2502)
  if 'script_sha256' in d:hashes[mode].add(d['script_sha256'])
proofs=[]
for p in sorted((R/'input_verification').glob('*.json')):
 d=read(p);proofs.append(p.stem)
 check(str(p)+':extra_vectors_bitwise',d.get('all_2502_extra_embeddings_equal'),True)
 check(str(p)+':numerical_first_abs',d.get('max_original_embedding_error',1)<1e-5,True)
 check(str(p)+':numerical_first_cosine',d.get('minimum_original_cosine_float64',0)>1-1e-10,True)
 for k,v in d.get('first_caption_rank_checks',{}).items():check(str(p)+':'+k+':R10_hits',v['R10_hit_disagreements'],0)
print(json.dumps({'audit':'read-only completed-result check; does not certify pending results or rerun model scoring','counts':counts,'input_verified_backbones':proofs,'script_hashes':{k:sorted(v) for k,v in hashes.items()},'issues':issues,'checked_files':len(sources),'source_sha256':sources},indent=2))
