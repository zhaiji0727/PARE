from pathlib import Path
import json,math,hashlib
R=Path('/home/coder/my_workspace/rs_paper/pare_workspace/report_revision_diagnostics_20261111')
p=R/'new_diagnostics_summary.json';s=json.loads(p.read_text());issues=[];details=[];hashes={}
def load(f):
 p=R/f;b=p.read_bytes();hashes[f]=hashlib.sha256(b).hexdigest();return json.loads(b)
def r10(d):return 100*sum(r<=10 for r in d['ranks'])/len(d['ranks'])
def checkstat(label,expected,a):
 n=len(a);m=math.fsum(a)/n;sd=math.sqrt(math.fsum((x-m)**2 for x in a)/(n-1)) if n>1 else None
 q={'n':n,'mean':m,'sd':sd,'values':a}
 if n==5:
  q['ordinary95']=[m-2.7764451051977987*sd/math.sqrt(5),m+2.7764451051977987*sd/math.sqrt(5)]
  q['family95_over5']=[m-4.604094871349993*sd/math.sqrt(5),m+4.604094871349993*sd/math.sqrt(5)]
  q['positive']=sum(x>0 for x in a)
 for k,v in q.items():
  got=expected[k]
  if isinstance(v,list):ok=len(v)==len(got) and all(abs(x-y)<1e-9 for x,y in zip(v,got))
  elif v is None:ok=got is None
  else:ok=abs(v-got)<1e-9
  if not ok:issues.append({'label':label,'field':k,'got':got,'expected':v})
 return q
for x in s['probe']:
 ds=[load(f) for f in x['source_files']];a=[r10(d)-d['original_metrics']['R@10'] for d in ds]
 q=checkstat('probe/'+x['backbone'],x['delta_constant_pi_minus_original'],a);details.append({'kind':'probe','backbone':x['backbone'],**q})
for x in s['constant_training']:
 ds=[load(f) for f in x['source_files']];a=[r10(d) for d in ds];b=[d['original_conditional_metrics']['R@10'] for d in ds]
 checkstat('constant/'+x['backbone'],x['constant_R10'],a);checkstat('conditional/'+x['backbone'],x['conditional_R10'],b)
 q=checkstat('delta/'+x['backbone'],x['delta_conditional_minus_constant'],[y-z for y,z in zip(b,a)]);details.append({'kind':'constant','backbone':x['backbone'],**q})
for x in s['allcaptions']:
 ds=[load(f) for f in x['source_files']];a=[r10(d) for d in ds];checkstat('allcaption/'+x['backbone']+'/'+x['variant'],x['R10'],a)
print(json.dumps({'summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'summary_complete':s['complete'],'n_probe':len(s['probe']),'n_constant':len(s['constant_training']),'n_allcaption':len(s['allcaptions']),'issues':issues,'details':details,'source_sha256':hashes},indent=2))
