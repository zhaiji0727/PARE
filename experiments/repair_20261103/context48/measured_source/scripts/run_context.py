#!/usr/bin/env python3
"""Fixed context48 study: nine fits, no search or reselection, then internal check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import fcntl

ROOT=Path(__file__).resolve().parents[1]
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace')
sys.path.insert(0,str(ROOT/'reference'))
import run_repair_phase1 as ref
SCIENCE_SHA='217545b95394076aa4dd11c17988c6637bf2869fc24ad100867af2d94feae80f'


def source_hashes():
    files={str(p.relative_to(ROOT)):ref.sha(p) for d in ['base_code','code','scripts','tests','reference'] for p in sorted((ROOT/d).glob('*.py'))}
    files['REFERENCE_SOURCE_PROVENANCE.json']=ref.sha(ROOT/'REFERENCE_SOURCE_PROVENANCE.json')
    return files


def check_plan(plan):
    expected={(1,1),(0,1),(1,0)}
    got=[]
    for c in plan['configurations']:
        f=c['config'];got.append((f['A'],f['T']))
        if c['id']!=f"context48_A{f['A']}T{f['T']}" or any(f[k]!=v for k,v in {'B':0,'C':1,'fusion':'original','reliability':'constant','token_mode':'original'}.items()):
            raise ValueError('Context study only permits the fixed three B0/C1/Rconstant configurations')
    if len(got)!=3 or set(got)!=expected or plan['seeds']!=[0,1,2] or plan['new_heads']!=9:raise ValueError('Wrong fixed grid')
    if plan['scope']['crop_half_pixels']!=24 or plan['scope']['suppression_half_pixels']!=12:raise ValueError('Wrong fixed geometry')
    return {**plan,'backbone':{'model':'ViT-B-32','pretrained':'laion2b_s34b_b79k','embedding_dimension':512}}


def verify_freeze(args):
    if ref.sha(ROOT/'protocol.json')!=SCIENCE_SHA:raise ValueError('Scientific protocol changed')
    f=json.loads(args.freeze.read_text())
    if f['protocol_sha256']!=SCIENCE_SHA or f['source_sha256']!=source_hashes():raise ValueError('Source freeze changed')
    if f['split_sha256']!={p.name:ref.sha(p) for p in sorted(args.split_dir.glob('*_ids.json'))}:raise ValueError('Split changed')
    return f


def parent_binding(args,plan):
    path=args.parent/'results/SELECTION_LOCK.json'
    if ref.sha(path)!=plan['parent_selection_lock_sha256']:raise ValueError('Original selection lock changed')
    selected=json.loads(path.read_text())['selected_configuration_id']
    if selected!='ABC101_T1_Rconstant_Tokoriginal':raise ValueError('Wrong fixed parent comparator')
    result={}
    for seed in [0,1,2]:
        seed_dir=args.parent/f'results/seed_{seed}'
        result[f'{seed}/manifest.json']=ref.sha(seed_dir/'manifest.json')
        for name,directory in [('A1T1',seed_dir/selected),('A0T1',seed_dir/'ABC001_T1_Rconstant_Tokoriginal'),
                               ('A1T0',seed_dir/'selected_control/ABC101_T0_Rconstant_Tokoriginal')]:
            for file in ['head.pt','development.json']:
                result[f'{seed}/{name}/{file}']={'path':str(directory/file),'sha256':ref.sha(directory/file)}
    return result



def validate_transform_binding(contract, original_repr):
    from build_fine_overlay import canonical_transform_repr
    expected=canonical_transform_repr(original_repr)
    if canonical_transform_repr(contract['eval_transform'])!=expected or canonical_transform_repr(contract['current_eval_transform_repr'])!=expected or contract['canonical_eval_transform']!=expected:
        raise ValueError('Overlay changed actual encoder preprocessing')


def load_bound_inputs(args,plan):
    import torch
    from fine_context import validate_overlay
    base,sig,pop,paths,hashes=ref.load_inputs(args,plan)
    overlay=torch.load(args.overlay,map_location='cpu',weights_only=True)
    expected={name:ref.sha(ROOT/name) for name in overlay['contract']['source_sha256']}
    validate_overlay(base,overlay,hashes['training_cache'],expected)
    if overlay['contract']['source_freeze_sha256']!=ref.sha(args.freeze):raise ValueError('Overlay produced under a different source freeze')
    if overlay['contract']['builder_sha256']!=ref.sha(ROOT/'scripts/build_fine_overlay.py') or overlay['contract']['protocol_sha256']!=SCIENCE_SHA:raise ValueError('Overlay builder/protocol differs')
    if overlay['contract']['base_signature']!=sig:raise ValueError('Overlay changed encoder/extraction signature')
    validate_transform_binding(overlay['contract'],base['provenance']['eval_transform'])
    paths['fine_overlay']=args.overlay;hashes['fine_overlay']=ref.sha(args.overlay)
    bindings={'input_sha256':hashes,'parent_comparators':parent_binding(args,plan),'overlay_contract':overlay['contract']}
    return base,overlay,sig,pop,paths,bindings


def verify_bound_file(args,bindings):
    old=json.loads(args.bindings.read_text())
    expected={'protocol_sha256':SCIENCE_SHA,'freeze_sha256':ref.sha(args.freeze),**bindings}
    if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Bound input/comparator bytes changed')


def validate_complete_training(args,plan):
    records={};manifests={}
    for seed in [0,1,2]:
        directory=args.out/f'seed_{seed}';path=directory/'manifest.json';m=json.loads(path.read_text())
        if m['status']!='complete_training_and_development' or set(m['runs'])!={c['id'] for c in plan['configurations']}:raise ValueError('All nine fits must complete')
        if m['identity']['freeze_sha256']!=ref.sha(args.freeze) or m['identity']['input_bindings_sha256']!=ref.sha(args.bindings):raise ValueError('Wrong training source/input bindings')
        manifests[str(seed)]=ref.sha(path)
        for name,run in m['runs'].items():
            if run['status']!='complete':raise ValueError('Incomplete context fit')
            if any(ref.sha(directory/name/file)!=digest for file,digest in run['artifacts_sha256'].items()):raise ValueError('Context fit artifact changed')
            if name=='context48_A0T1' and not (run.get('parent_A0_anchor',{}).get('all_parameter_tensors_bitwise_equal') and run.get('parent_A0_anchor',{}).get('development_ranks_identical')):raise ValueError('Mandatory A0 parameter/rank anchor missing')
            records[f'{seed}/{name}']=run['artifacts_sha256']
    return {'training_manifest_sha256':manifests,'all_training_artifacts_sha256':records,
            'protocol_sha256':SCIENCE_SHA,'freeze_sha256':ref.sha(args.freeze),'input_bindings_sha256':ref.sha(args.bindings)}


def run(args,plan,frozen,torch):
    from repair_head import RepairHead,RepairConfig
    from saqre_phys import SAQREHead
    from train_saqre import load_captions
    from reference_kernels import train_one,validate_one
    from fine_context import compose_overlay
    if args.seed not in [0,1,2]:raise ValueError('Seed required')
    completion_sha=None
    if args.command=='check':
        complete_path=args.out/'TRAINING_COMPLETE_LOCK.json'
        complete=json.loads(complete_path.read_text())
        expected=validate_complete_training(args,plan)
        if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Training completion lock changed')
        completion_sha=ref.sha(complete_path)
    elif (args.out/'TRAINING_COMPLETE_LOCK.json').exists():raise ValueError('Training frozen after completion lock')
    base,overlay,sig,pop,paths,bindings=load_bound_inputs(args,plan);verify_bound_file(args,bindings)
    cache=compose_overlay(base,overlay,plan['configurations'][0]['config'],bindings['input_sha256']['training_cache'])
    pairs,queries,gallery=ref.pairs_for(cache,pop,load_captions(args.captions),args.command)
    if args.command=='train' and len(pairs)!=23016:raise ValueError('Wrong train pairs')
    folder=args.out/f'seed_{args.seed}'
    if args.command=='check':folder=folder/'internal_check'
    if folder.exists() and not args.resume:raise FileExistsError('Use a new output or explicit --resume')
    folder.mkdir(parents=True,exist_ok=True);lock=(folder/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'protocol_sha256':SCIENCE_SHA,'freeze_sha256':ref.sha(args.freeze),'input_bindings_sha256':ref.sha(args.bindings),
              'overlay_sha256':bindings['input_sha256']['fine_overlay'],'stage':args.command,'seed':args.seed,'training_complete_lock_sha256':completion_sha}
    mp=folder/'manifest.json'
    if mp.exists():
        manifest=json.loads(mp.read_text())
        if manifest['identity']!=identity:raise ValueError('Resume identity mismatch')
    else:
        manifest={'schema':'pare.context48-stage.v1','identity':identity,'status':'starting','runs':{},'failure_retry_history':[],
                  'train_ids':sorted(pop['train']),'development_ids':sorted(pop['development']),'internal_check_ids':sorted(pop['internal_check']),
                  'n_train_images':6610,'n_train_pairs':23016,'base_nonfine_fields_reused_unchanged':True,'posthoc_exploratory':True,'official_test_accessed':False}
    def immutable():
        verify_freeze(args)
        if any(ref.sha(p)!=bindings['input_sha256'][name] for name,p in paths.items()):raise ValueError('Input bytes changed')
        if parent_binding(args,plan)!=bindings['parent_comparators']:raise ValueError('Parent comparator changed')
        verify_bound_file(args,bindings)
    try:
        random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        cap2vec,embedding_sha=ref.embed(plan,pairs,queries,torch)
        manifest['text_embeddings_sha256']=embedding_sha;manifest['environment']=ref.environment(torch)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);initial=SAQREHead(512)
        state={k:v.detach().clone() for k,v in initial.state_dict().items()};initial_sha=ref.state_sha(state);del initial
        manifest['base_initial_state_sha256']=initial_sha
        pair_sha=ref.objsha([[cache['ids'][row],hashlib.sha256(text.encode()).hexdigest()] for row,text in pairs]);manifest['initial_pairs_sha256']=pair_sha
        order=list(range(len(pairs)));rng=random.Random(args.seed);expected_orders=[]
        for _ in range(10):rng.shuffle(order);expected_orders.append(ref.objsha(order))
        if args.command=='train':
            parent=json.loads((args.parent/f'results/seed_{args.seed}/manifest.json').read_text())
            if parent['text_embeddings_sha256']!=embedding_sha or parent['base_initial_state_sha256']!=initial_sha or parent['initial_pairs_sha256']!=pair_sha:raise ValueError('Context fit confounds frozen text/init/pair order')
        for c in plan['configurations']:
            name=c['id'];directory=folder/name;prior=manifest['runs'].get(name)
            if prior and prior['status']=='complete':
                if any(ref.sha(directory/f)!=h for f,h in prior['artifacts_sha256'].items()):raise ValueError('Completed artifact changed')
                continue
            if directory.exists():
                if not args.retry_incomplete:raise FileExistsError('Explicit retry required, old attempt retained')
                archive=folder/'attempts'/f'{name}_{time.time_ns()}';archive.parent.mkdir(exist_ok=True);shutil.move(str(directory),archive)
                manifest['failure_retry_history'].append({'configuration':name,'archived':str(archive.relative_to(folder)),'prior':prior})
            directory.mkdir();torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
            head=RepairHead(512,RepairConfig(**c['config']));head.load_base_state_dict(state)
            if ref.state_sha({k:head.state_dict()[k] for k in state})!=initial_sha:raise ValueError('Paired initialization mismatch')
            head=head.cuda();run={'status':'running','configuration':c,'seed':args.seed,'base_initial_state_sha256':initial_sha,
                                  'full_initial_state_sha256':ref.state_sha(head.state_dict()),'text_embeddings_sha256':embedding_sha}
            manifest['runs'][name]=run;manifest.update(status='running',active_configuration=name);ref.write_json(mp,manifest)
            if args.command=='train':
                history=train_one(head,cache,pairs,cap2vec,'cuda',args.seed,plan['training'],expected_orders,
                                  lambda h:ref.write_json(directory/'training_progress.json',{'history':h,'updated_utc':ref.now()}))
                checkpoint={'schema':'pare.context48-head.v1','state_dict':{k:v.detach().cpu() for k,v in head.state_dict().items()},'D':512,'config':c['config'],'seed':args.seed,
                            'n_images':6610,'n_pairs':23016,'history':history,'settings':plan['training'],
                            'provenance':{**identity,'train_ids':sorted(pop['train']),'development_ids':sorted(pop['development']),'excluded_internal_check_ids':sorted(pop['internal_check']),
                            'input_sha256':bindings['input_sha256'],'base_cache_signature':sig,'fine_overlay_contract':overlay['contract'],
                            'base_initial_state_sha256':initial_sha,'text_embeddings_sha256':embedding_sha,'initial_pairs_sha256':pair_sha}}
                headpath=directory/'head.pt';torch.save(checkpoint,headpath);label='development'
            else:
                headpath=args.out/f'seed_{args.seed}'/name/'head.pt';checkpoint=torch.load(headpath,map_location='cpu',weights_only=True)
                if checkpoint['config']!=c['config'] or checkpoint['provenance']['input_bindings_sha256']!=ref.sha(args.bindings):raise ValueError('Wrong context head')
                head.load_state_dict(checkpoint['state_dict'],strict=True);label='internal_check'
            headsha=ref.sha(headpath)
            matrix,ranks,metrics,gold=validate_one(head,cache,gallery,queries,cap2vec,'cuda');immutable()
            if ref.sha(headpath)!=headsha:raise ValueError('Head changed during evaluation')
            ref.save_evaluation(directory,label,headsha,c,args.seed,cache,gallery,queries,matrix,ranks,metrics,gold,SCIENCE_SHA,ref.sha(args.freeze),torch)
            result=json.loads((directory/f'{label}.json').read_text());result.update(input_bindings_sha256=ref.sha(args.bindings),overlay_sha256=bindings['input_sha256']['fine_overlay'])
            ref.write_json(directory/f'{label}.json',result)
            if args.command=='train' and name=='context48_A0T1':
                olddir=args.parent/f'results/seed_{args.seed}/ABC001_T1_Rconstant_Tokoriginal'
                old=torch.load(olddir/'head.pt',map_location='cpu',weights_only=True);oldr=json.loads((olddir/'development.json').read_text())
                same=checkpoint['state_dict'].keys()==old['state_dict'].keys() and all(torch.equal(v,old['state_dict'][k]) for k,v in checkpoint['state_dict'].items())
                ranks_same=ranks==oldr['ranks'] and result['query_ids']==oldr['query_ids'] and result['gallery_ids']==oldr['gallery_ids']
                anchor={'all_parameter_tensors_bitwise_equal':same,'development_ranks_identical':ranks_same,'parent_head_sha256':ref.sha(olddir/'head.pt'),'parent_development_sha256':ref.sha(olddir/'development.json')}
                ref.write_json(directory/'parent_A0_anchor.json',anchor);run['parent_A0_anchor']=anchor
                if not same or not ranks_same:raise ValueError('Mandatory A0 bitwise/rank anchor failed; do not advance')
            run.update(status='complete',metrics_percent=metrics,artifacts_sha256={p.name:ref.sha(p) for p in directory.iterdir() if p.is_file()})
            ref.write_json(mp,manifest);print(json.dumps({'complete':name,'seed':args.seed,'stage':args.command}),flush=True)
            del head,matrix
        immutable()
        if set(manifest['runs'])!={c['id'] for c in plan['configurations']} or any(r['status']!='complete' for r in manifest['runs'].values()):raise ValueError('Incomplete fixed grid')
        manifest.update(status='complete_training_and_development' if args.command=='train' else 'complete_internal_check',active_configuration=None,completed_utc=ref.now());ref.write_json(mp,manifest)
    except BaseException as error:
        manifest['status']='failed';manifest['failure_retry_history'].append({'type':type(error).__name__,'message':str(error),'utc':ref.now()});ref.write_json(mp,manifest);raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['freeze','bind-inputs','train','lock-training','check'])
    ap.add_argument('--freeze',type=Path,default=ROOT/'CONTEXT_SOURCE_FROZEN.json');ap.add_argument('--bindings',type=Path,default=ROOT/'INPUTS_FROZEN.json')
    ap.add_argument('--split-dir',type=Path,default=ROOT/'plans/splits');ap.add_argument('--original-split-dir',type=Path,default=P/'data/ret_clean_20261007')
    ap.add_argument('--cache',type=Path,default=P/'formal_experiments_20261103/results/vitb32/cache_physics_train.pt');ap.add_argument('--overlay',type=Path,default=ROOT/'cache/fine_context48.pt')
    ap.add_argument('--captions',type=Path,default=P/'data/ret_clean_20261007/Caption_train.csv');ap.add_argument('--models',type=Path,default=P/'models')
    ap.add_argument('--parent',type=Path,default=P/'repair_experiments_20261103_v1');ap.add_argument('--out',type=Path,default=ROOT/'results')
    ap.add_argument('--seed',type=int,choices=[0,1,2]);ap.add_argument('--gpu',type=int);ap.add_argument('--resume',action='store_true');ap.add_argument('--retry-incomplete',action='store_true')
    ap.add_argument('--acknowledge-reviewed-source',action='store_true');args=ap.parse_args();plan=check_plan(json.loads((ROOT/'protocol.json').read_text()))
    if args.command=='freeze':
        if not args.acknowledge_reviewed_source or args.freeze.exists() or args.out.exists():raise ValueError('Reviewed fresh source freeze required')
        if ref.sha(ROOT/'protocol.json')!=SCIENCE_SHA:raise ValueError('Scientific protocol changed')
        ref.write_json(args.freeze,{'schema':'pare.context48-source-freeze.v1','protocol_sha256':SCIENCE_SHA,'source_sha256':source_hashes(),
                                   'split_sha256':{p.name:ref.sha(p) for p in sorted(args.split_dir.glob('*_ids.json'))},'frozen_utc':ref.now()});print(ref.sha(args.freeze));return
    frozen=verify_freeze(args);torch=ref.setup_runtime(args)
    if args.command=='bind-inputs':
        if args.bindings.exists() or args.out.exists():raise FileExistsError('Bindings immutable and must precede training')
        _,_,_,_,_,bindings=load_bound_inputs(args,plan)
        ref.write_json(args.bindings,{'protocol_sha256':SCIENCE_SHA,'freeze_sha256':ref.sha(args.freeze),**bindings,'bound_utc':ref.now()});print(ref.sha(args.bindings));return
    if args.command=='lock-training':
        payload=validate_complete_training(args,plan);path=args.out/'TRAINING_COMPLETE_LOCK.json'
        if path.exists():
            old=json.loads(path.read_text())
            if any(old.get(k)!=v for k,v in payload.items()):raise ValueError('Training lock immutable')
        else:
            with path.open('x') as f:json.dump({**payload,'locked_utc':ref.now(),'selection_performed':False},f,indent=2)
        print(ref.sha(path));return
    run(args,plan,frozen,torch)

if __name__=='__main__':main()
