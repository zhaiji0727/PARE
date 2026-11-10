"""Descriptive exact-text overlap audit; does not change any split or model."""
from pathlib import Path
import sys,json,collections,statistics
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace');S=P/'simplified_confirm_20261109_v1';O=P/'report_revision_diagnostics_20261111';sys.path.insert(0,str(S/'base_code'))
from train_saqre import load_captions
from retrieval_queries import select_queries
bound=json.loads((S/'INPUTS_BOUND.json').read_text());train=load_captions(bound['common']['train_captions']['path']);test=load_captions(bound['common']['test_captions']['path'])
trids=set(json.loads(Path(bound['common']['train_ids']['path']).read_text()));teids=json.loads(Path(bound['common']['test_ids']['path']).read_text());q,_=select_queries(test,teids);allq,_=select_queries(test,teids,'all-captions')
tc=[c for k,cs in train.items() if k in trids for c in cs];ts=set(tc);ct=collections.Counter(r['caption'] for r in q)
d={'normalization':'Original load_captions: strip strings; exact text equality only','train_pairs':len(tc),'train_unique_texts':len(ts),'first_test_queries':len(q),'first_test_unique_texts':len(ct),'first_test_queries_in_duplicate_groups':sum(v for v in ct.values() if v>1),'first_test_duplicate_groups':sum(v>1 for v in ct.values()),'first_test_max_duplicate_group':max(ct.values()),'first_test_queries_matching_any_train_caption':sum(r['caption'] in ts for r in q),'first_test_distinct_texts_matching_any_train_caption':len(set(r['caption'] for r in q)&ts),'all_test_caption_rows':len(allq),'all_test_caption_rows_matching_any_train_caption':sum(r['caption'] in ts for r in allq)}
groups={'exact_text_overlap':[i for i,r in enumerate(q) if r['caption'] in ts],'no_exact_text_overlap':[i for i,r in enumerate(q) if r['caption'] not in ts]};rows=[]
for b in ['vitb32','vitb16','remoteclip','siglip','georsclip']:
 for v in ['full','strict','strict_token_off','scene']:
  files=[S/f'baselines/{b}/test.json'] if v=='scene' else [S/f'results/{b}/seed_{seed}/{v}/test.json' for seed in range(5)]
  ds=[json.loads(f.read_text()) for f in files]
  assert all(t['query_ids']==[r['query_id'] for r in q] for t in ds)
  for g,ix in groups.items():
   vals=[sum(t['ranks'][i]<=10 for i in ix)/len(ix)*100 for t in ds]
   rows.append({'backbone':b,'variant':v,'group':g,'n_queries':len(ix),'R10_mean':statistics.mean(vals),'R10_sd':statistics.stdev(vals) if len(vals)>1 else None,'R10_values':vals})
d.update(subgroups=rows,scope='Post-hoc descriptive subgroup summaries on unchanged full-gallery ranks. Repeated text does not imply repeated image or establish data leakage; semantic similarity is not audited.',query_groups={g:[q[i]['query_id'] for i in ix] for g,ix in groups.items()})
(O/'caption_overlap.json').write_text(json.dumps(d,indent=2)+'\n')
print('overlap',d['first_test_queries_matching_any_train_caption'],'nonoverlap',len(groups['no_exact_text_overlap']),'rows',len(rows))
