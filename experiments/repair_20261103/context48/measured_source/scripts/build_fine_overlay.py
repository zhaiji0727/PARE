#!/usr/bin/env python3
"""Verify frozen fine geometry, then create an explicitly separate context overlay.

metadata: all 9181 original-training images, no encoder; exact old24 metadata.
smoke: 16 deterministic images, recreate old full encoder batch and test context48.
build: full 9181 fine-only overlay, requires completed metadata/smoke reports.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import inspect
import json
import multiprocessing
import os
import re
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace')
sys.path[:0]=[str(ROOT/'base_code'),str(ROOT/'code')]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical_transform_repr(value):
    return re.sub(r'(<function [A-Za-z_][A-Za-z_0-9.]* at )0x[0-9a-fA-F]+(>)',r'\1<process-address>\2',value)

def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def extraction_sources():return {n:sha(ROOT/n) for n in ['code/fine_context.py','base_code/saqre_phys.py','base_code/runtime.py','base_code/cache_primitives.py','base_code/artifacts.py']}

def metadata_worker(task):
    row,name,path,expected_hash=task
    import numpy as np
    from PIL import Image
    from fine_context import extract_fine_context,boxes_from_centers
    from saqre_phys import scatterer_patches
    if sha(path)!=expected_hash:raise ValueError('Image bytes changed: '+name)
    with Image.open(path) as f: old=extract_fine_context(f,12)
    reference=scatterer_patches(old['g'],old['det'],max_sc=8,half=12)
    if old['boxes']!=reference:raise ValueError('Old24 box mismatch: '+name)
    newboxes=boxes_from_centers(old['centers'],old['g'].shape,24)
    h,w=old['g'].shape
    for (x,y),box in zip(old['centers'],newboxes):
        if box!=(max(0,x-24),max(0,y-24),min(w,x+24),min(h,y+24)):raise ValueError('New geometry mismatch')
        if not(box[0]<=x<box[2] and box[1]<=y<box[3]):raise ValueError('Center not in new box')
    n=len(old['features']);mu=np.zeros(8,np.float32);t=np.zeros(8,np.float32);tf=np.zeros((8,4),np.float32)
    for i,v in enumerate(old['features']):mu[i]=v['mu'];t[i]=v['t'];tf[i]=v['typefeat']
    return {'row':row,'id':name,'n':n,'f_mu':mu,'f_t':t,'f_tf':tf,'L':np.float32(old['L']),'c_star':np.float32(old['c_star']),'nu':np.float32(old['nu']),
            'centers':old['centers'],'old_boxes':old['boxes'],'context_boxes':newboxes,'shape':[h,w]}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['metadata','smoke','build'])
    ap.add_argument('--base-cache',type=Path,default=P/'formal_experiments_20261103/results/vitb32/cache_physics_train.pt')
    ap.add_argument('--images',type=Path);ap.add_argument('--models',type=Path,default=P/'models');ap.add_argument('--gpu',type=int,default=5)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--metadata-report',type=Path,default=ROOT/'validation/all9181_old24_metadata_final.json')
    ap.add_argument('--smoke-report',type=Path,default=ROOT/'validation/old24_context48_smoke_final.json')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Use a new output; never overwrite a cache or report')
    if args.gpu not in range(6):raise ValueError('Only allocated GPUs0–5 may be used')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),RS_RAG_MODEL_ROOT=str(args.models),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from PIL import Image
    from cache_primitives import build_image_index,encode
    from artifacts import validate_cache
    from runtime import create_backbone
    from fine_context import extract_fine_context,validate_overlay
    from saqre_phys import extract_primitives
    torch.set_num_threads(4)
    plan=json.loads((ROOT/'protocol.json').read_text());base_sha=sha(args.base_cache);builder_sha=sha(Path(__file__))
    if args.mode=='build':
        from run_context import source_hashes
        frozen=json.loads((ROOT/'CONTEXT_SOURCE_FROZEN.json').read_text())
        if frozen['source_sha256']!=source_hashes() or frozen['protocol_sha256']!=sha(ROOT/'protocol.json'):raise ValueError('Full build requires the reviewed frozen source tree')
    if base_sha!=plan['input_bindings']['training_cache_sha256']:raise ValueError('Only the frozen training cache is allowed')
    base=torch.load(args.base_cache,map_location='cpu',weights_only=True)
    signature=validate_cache(base,'ViT-B-32','laion2b_s34b_b79k')
    if len(base['ids'])!=9181 or base['D']!=512 or base['Mf']!=8:raise ValueError('Unexpected base cache')
    images=args.images or Path(base['config']['img_root']);index=build_image_index(images)
    image_hashes=base['provenance']['image_file_sha256'];sources=extraction_sources()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='metadata':
        tasks=[(i,n,index[n],image_hashes[n]) for i,n in enumerate(base['ids'])]
        records=[];checked=['f_n','f_mu','f_t','f_tf','L','c_star','nu']
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for x in pool.map(metadata_worker,tasks,chunksize=8):
                row=x.pop('row');name=x['id']
                if int(base['f_n'][row])!=x['n']:raise ValueError('Old count mismatch '+name)
                for k in checked[1:]:
                    if not np.array_equal(base[k][row].numpy(),x.pop(k)):raise ValueError('Old metadata mismatch '+name+' '+k)
                records.append(x)
                if len(records)%500==0:print(json.dumps({'metadata_checked':len(records),'of':9181}),flush=True)
        if sha(args.base_cache)!=base_sha or extraction_sources()!=sources or sha(Path(__file__))!=builder_sha:raise RuntimeError('Input/source changed during metadata check')
        write(args.out,{'schema':'pare.context48-old24-metadata-check.v1','passed':True,'all_original_images':9181,'base_cache_sha256':base_sha,
                        'protocol_sha256':sha(ROOT/'protocol.json'),'source_sha256':sources,'builder_sha256':builder_sha,'image_hashes_verified':True,'old_boxes_exact':True,
                        'old_metadata_fields_bitwise_equal':checked,'new_geometry_checked':True,
                        'geometry_sha256':objsha([{'id':r['id'],'centers':r['centers'],'boxes':r['context_boxes'],'shape':r['shape']} for r in records]),'records':records,'seconds':time.monotonic()-start})
        print(json.dumps({'metadata_passed':9181,'report_sha256':sha(args.out)}));return
    model,_,transform=create_backbone('ViT-B-32','laion2b_s34b_b79k');model=model.cuda().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if canonical_transform_repr(repr(transform))!=canonical_transform_repr(base['provenance']['eval_transform']):raise ValueError('Encoder preprocessing differs from the original cache')
    # Keep the encoder backend defaults from the original cache-building program;
    # record them explicitly. No training precision setting is applied here.
    backend={'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
             'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()}
    contract={'base_cache_sha256':base_sha,'base_signature':signature,'base_signature_sha256':base['provenance']['signature_sha256'],
              'pretrained_weight_sha256':signature['pretrained_weight_sha256'],'eval_transform':base['provenance']['eval_transform'],'current_eval_transform_repr':repr(transform),
              'canonical_eval_transform':canonical_transform_repr(repr(transform)),'canonical_transform_rule':'replace only hex memory address in <function NAME at 0x...>','source_sha256':sources,
              'builder_sha256':builder_sha,'protocol_sha256':sha(ROOT/'protocol.json'),'crop_half':24,'suppression_half':12,
              'max_fine_crops':8,'encoder_batch_policy':'original scene and coarse crops precede context fine crops, capacity128',
              'transform_implementation_files':{(item.__module__ if inspect.isfunction(item) else type(item).__module__):sha(inspect.getsourcefile(item if inspect.isfunction(item) else type(item))) for item in transform.transforms},
              'backend':backend,'created_utc':datetime.now(timezone.utc).isoformat()}
    if args.mode=='smoke':
        selected=list(range(16));rows=[];maxerr={k:0. for k in ['scene','c_emb','f_emb']}
        for row in selected:
            name=base['ids'][row]
            if sha(index[name])!=image_hashes[name]:raise ValueError('Image changed')
            with Image.open(index[name]) as f:im=f.convert('RGB')
            reference=extract_primitives(im,max_regions=6,max_sc=8);old=extract_fine_context(im,12);new=extract_fine_context(im,24)
            original_crops=[reference['scene'][0]]+[c for c,_ in reference['regions']]+[c for c,_ in reference['scatterers']]
            factored_crops=[reference['scene'][0]]+[c for c,_ in reference['regions']]+old['crops']
            if len(original_crops)!=len(factored_crops) or any(not np.array_equal(np.array(a),np.array(b)) for a,b in zip(original_crops,factored_crops)):raise ValueError('Factored old24 crop pixels differ')
            vectors=encode(model,transform,'cuda',factored_crops,bs=128)
            nc=len(reference['regions']);nf=len(old['crops'])
            for k,a,b in [('scene',vectors[0],base['scene'][row]),('c_emb',vectors[1:1+nc],base['c_emb'][row,:nc]),('f_emb',vectors[1+nc:],base['f_emb'][row,:nf])]:
                diff=float((a-b).abs().max()) if a.numel() else 0.;maxerr[k]=max(maxerr[k],diff)
                if not torch.allclose(a,b,atol=1e-6,rtol=1e-6):raise ValueError('Old24 embedding mismatch '+k+' '+str(diff))
            newvectors=encode(model,transform,'cuda',[reference['scene'][0]]+[c for c,_ in reference['regions']]+new['crops'],bs=128)
            if not torch.isfinite(newvectors).all():raise ValueError('Nonfinite context embedding')
            rows.append({'id':name,'fine_count':nf,'centers':new['centers'],'old_boxes':old['boxes'],'new_boxes':new['boxes'],'old_pixels_exact':True})
        if sha(args.base_cache)!=base_sha or extraction_sources()!=sources or sha(Path(__file__))!=builder_sha:raise RuntimeError('Source/base cache changed')
        write(args.out,{'schema':'pare.context48-smoke.v1','passed':True,'n_images':16,'ids':[base['ids'][i] for i in selected],
                        'contract':contract,'old24_embedding_max_abs_error':maxerr,'tolerance':{'atol':1e-6,'rtol':1e-6},'rows':rows,
                        'all_metadata_required_separately':True,'official_test_accessed':False})
        print(json.dumps({'smoke_passed':True,'max_abs_error':maxerr,'report_sha256':sha(args.out)}));return
    for path,kind in [(args.metadata_report,'metadata'),(args.smoke_report,'smoke')]:
        report=json.loads(path.read_text())
        if not report['passed']:raise ValueError('Required prior validation failed')
        bound=report if kind=='metadata' else report['contract']
        if bound['source_sha256']!=sources or bound['base_cache_sha256']!=base_sha or bound['protocol_sha256']!=sha(ROOT/'protocol.json') or bound['builder_sha256']!=builder_sha:raise ValueError('Validation report binding mismatch')
        if kind=='metadata':
            if report['all_original_images']!=9181:raise ValueError('Full original metadata check required')
            contract['geometry_sha256']=report['geometry_sha256']
    contract['metadata_report_sha256']=sha(args.metadata_report);contract['smoke_report_sha256']=sha(args.smoke_report)
    contract['source_freeze_sha256']=sha(ROOT/'CONTEXT_SOURCE_FROZEN.json')
    n=len(base['ids']);overlay={'schema':'pare.fine-context-overlay.v1','ids':base['ids'],'D':512,'contract':contract,
        'f_emb':torch.zeros(n,8,512),'f_mu':torch.zeros(n,8),'f_t':torch.zeros(n,8),'f_tf':torch.zeros(n,8,4),
        'f_n':torch.zeros(n,dtype=torch.long),'centers':torch.full((n,8,2),-1,dtype=torch.long),'boxes':torch.full((n,8,4),-1,dtype=torch.long),'image_sizes':torch.zeros(n,2,dtype=torch.long)}
    start=time.monotonic()
    for row,name in enumerate(base['ids']):
        if sha(index[name])!=image_hashes[name]:raise ValueError('Image bytes changed '+name)
        with Image.open(index[name]) as f:im=f.convert('RGB')
        original=extract_primitives(im,max_regions=6,max_sc=8);fine=extract_fine_context(im,24);nf=len(fine['features']);nc=len(original['regions'])
        if nf!=int(base['f_n'][row]):raise ValueError('Changed center count')
        vectors=encode(model,transform,'cuda',[original['scene'][0]]+[c for c,_ in original['regions']]+fine['crops'],bs=128)
        overlay['f_n'][row]=nf;overlay['f_emb'][row,:nf]=vectors[1+nc:];overlay['image_sizes'][row]=torch.tensor(fine['g'].shape)
        if nf:overlay['centers'][row,:nf]=torch.tensor(fine['centers']);overlay['boxes'][row,:nf]=torch.tensor(fine['boxes'])
        for j,feat in enumerate(fine['features']):overlay['f_mu'][row,j]=feat['mu'];overlay['f_t'][row,j]=feat['t'];overlay['f_tf'][row,j]=torch.tensor(feat['typefeat'])
        if (row+1)%100==0:print(json.dumps({'context_images':row+1,'of':n,'seconds':time.monotonic()-start}),flush=True)
    if sha(args.base_cache)!=base_sha or extraction_sources()!=sources or sha(Path(__file__))!=builder_sha:raise RuntimeError('Input/source changed during context cache creation')
    validate_overlay(base,overlay,base_sha,sources);torch.save(overlay,args.out)
    write(args.out.with_suffix('.manifest.json'),{'complete':True,'overlay_sha256':sha(args.out),'images':n,'contract':contract,'seconds':time.monotonic()-start})
    print(json.dumps({'overlay_complete':True,'sha256':sha(args.out),'images':n}))

if __name__=='__main__':main()
