from pathlib import Path
import json,statistics as st,math,csv
from scipy.stats import t
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace/report_revision_diagnostics_20261111')
B=['vitb32','vitb16','remoteclip','siglip','georsclip']
def read(p):return json.loads(p.read_text())
def calc(a):
    m=st.mean(a);s=st.stdev(a) if len(a)>1 else None
    d={'n':len(a),'mean':m,'sd':s,'values':a}
    if s is not None:
        d.update(ordinary95=[m-t.ppf(.975,len(a)-1)*s/math.sqrt(len(a)),m+t.ppf(.975,len(a)-1)*s/math.sqrt(len(a))],family95_over5=[m-t.ppf(.995,len(a)-1)*s/math.sqrt(len(a)),m+t.ppf(.995,len(a)-1)*s/math.sqrt(len(a))],positive=sum(v>0 for v in a))
    return d
out={'note':'Post-hoc additions on reused benchmark. New training contrasts are S(conditional) minus constant-descriptor. Probe contrasts are fixed-mean-posterior minus original. Family intervals, if used, correct five backbones within this separate diagnostic family; they do not convert the diagnostic into an unseen-data confirmatory test.','probe':[],'constant_training':[],'allcaptions':[],'text_verification':[]}
for b in B:
    ds=[read(f) for f in sorted((P/'probe'/b).glob('seed_*.json'))]
    if len(ds)==5:out['probe'].append({'backbone':b,'delta_constant_pi_minus_original':calc([d['delta_R10_constant_minus_original'] for d in ds]),'token_norm_min':min(d['posterior']['simplex_min_norm'] for d in ds),'token_norm_max':max(d['posterior']['simplex_max_norm'] for d in ds),'source_files':[str(f.relative_to(P)) for f in sorted((P/'probe'/b).glob('seed_*.json'))]})
    ds=[read(f) for f in sorted((P/'constant_training'/b).glob('seed_*/result.json'))]
    if len(ds)==5:
        # Independently recompute metrics from full rank arrays.
        for d in ds:assert abs(sum(r<=10 for r in d['ranks'])/3688*100-d['metrics']['R@10'])<1e-10
        out['constant_training'].append({'backbone':b,'constant_R10':calc([d['metrics']['R@10'] for d in ds]),'conditional_R10':calc([d['original_conditional_metrics']['R@10'] for d in ds]),'delta_conditional_minus_constant':calc([d['delta_R10_conditional_minus_constant'] for d in ds]),'source_files':[str(f.relative_to(P)) for f in sorted((P/'constant_training'/b).glob('seed_*/result.json'))]})
    for v in ['full','strict','strict_token_off','scene']:
        fs=[P/f'allcaptions/{b}/scene.json'] if v=='scene' else sorted((P/'allcaptions'/b).glob(f'{v}_seed_*.json'))
        if len(fs)!=(1 if v=='scene' else 5) or not all(f.exists() for f in fs):continue
        ds=[read(f) for f in fs]
        for d in ds:assert len(d['ranks'])==6190 and abs(sum(r<=10 for r in d['ranks'])/6190*100-d['metrics']['R@10'])<1e-10
        out['allcaptions'].append({'backbone':b,'variant':v,'R10':calc([d['metrics']['R@10'] for d in ds]),'source_files':[str(f.relative_to(P)) for f in fs]})
    vf=P/f'input_verification/{b}.json'
    if vf.exists():out['text_verification'].append(read(vf))
out['allcaption_numerical_validation']={'accepted':len(out['text_verification'])==5 and all(
    v.get('max_original_embedding_error',1)<1e-5 and v.get('minimum_original_cosine_float64',0)>1-1e-10
    and v.get('all_2502_extra_embeddings_equal') is True
    and set(v.get('first_caption_rank_checks',{}))=={'scene','full_seed0','strict_seed0','strict_token_off_seed0'}
    and all(r.get('R10_hit_disagreements',1)==0 and r.get('n_checked')==3688 for r in v['first_caption_rank_checks'].values())
    for v in out['text_verification']),
    'criteria':'All five backbone/source file hashes validated by verify_text_numerical.py; 3688 original first-caption vectors within maxabs<1e-5 and float64 cosine>1-1e-10; 2502 extra vectors exact upon repeat; new first-caption scoring of seed0 F/S/noToken and Scene has zero R10 hit differences. This is numerical validation, not bitwise equality of all original vectors or ranks.',
    'retained_difference':'RemoteCLIP full seed0 has one non-top10 rank difference; R10 membership unchanged. All other checked rank arrays match.'}
out['complete']=len(out['probe'])==5 and len(out['constant_training'])==5 and len(out['allcaptions'])==20 and out['allcaption_numerical_validation']['accepted']
if (P/'caption_overlap.json').exists():out['caption_overlap']=read(P/'caption_overlap.json')
(P/'new_diagnostics_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'complete':out['complete'],'n_probe':len(out['probe']),'n_constant_training':len(out['constant_training']),'n_allcaptions':len(out['allcaptions']),'n_input_verification':len(out['text_verification'])}))
for d in out['constant_training']:print(d['backbone'],'S-minus-constant',d['delta_conditional_minus_constant'])
for d in out['allcaptions']:print(d['backbone'],d['variant'],d['R10']['mean'],d['R10']['sd'])
