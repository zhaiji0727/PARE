from pathlib import Path
import json,gzip,hashlib
P=Path(__file__).resolve().parent
manifest=json.loads((P/'MANIFEST.json').read_text());n=m=0;errors=[]
def metrics(r):
 return {'R@1':100*sum(x<=1 for x in r)/len(r),'R@5':100*sum(x<=5 for x in r)/len(r),'R@10':100*sum(x<=10 for x in r)/len(r),'MRR@10':100*sum(1/x if x<=10 else 0 for x in r)/len(r)}
for name,entry in manifest['files'].items():
 p=P/name;b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==entry['file_sha256']
 payload=gzip.decompress(b) if p.suffix=='.gz' else b
 assert hashlib.sha256(payload).hexdigest()==entry['projected_content_sha256']
 if not (name.endswith('.json') or name.endswith('.json.gz')):continue
 j=json.loads(payload)
 if 'ranks' in j and 'metrics_percent' in j:
  n+=1
  for k,v in metrics(j['ranks']).items():
   if abs(v-j['metrics_percent'][k])>1e-10:errors.append([name,k,v,j['metrics_percent'][k]])
 if 'baseline_ranks' in j and 'probe_ranks' in j:
  m+=1
  for key in ['baseline','probe']:
   for k,v in metrics(j[key+'_ranks']).items():
    if abs(v-j[key+'_metrics_percent'][k])>1e-10:errors.append([name,key,k])
assert n==108 and m==42 and not errors,(n,m,errors)
print(json.dumps({'files_hash_verified':len(manifest['files']),'retrieval_evaluations_recomputed':n,'probes_recomputed':m,'errors':errors,'status':'pass'}))
print('PASS',n,'retrieval and',m,'probe metrics')
