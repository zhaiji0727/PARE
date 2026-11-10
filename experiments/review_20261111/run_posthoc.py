"""2026-11-11 frozen-head diagnostics. Never trains or writes original study."""
import argparse, os, sys, json, types, time, hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True);ap.add_argument('--gpu',default='6');ap.add_argument('--mode',choices=['probe','allcaptions'],required=True);args=ap.parse_args()
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace');S=P/'simplified_confirm_20261109_v1';O=P/'report_revision_diagnostics_20261111'/args.mode/args.backbone;O.mkdir(parents=True,exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,RS_RAG_MODEL_ROOT=str(P/'models'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=[str(S/'base_code'),str(S/'reference'),str(S/'code')]
import torch, numpy as np
import torch.nn.functional as F
from scipy.optimize import minimize
from saqre_phys import SAQREHead
from train_saqre import move_docs,load_captions
from evaluate_multik import score_head_matrix,metrics
from batch_scoring import score_matrix
from study_core import head_ranks,sha,write_json
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
B=args.backbone;plan=json.loads((S/'protocol.json').read_text());bound=json.loads((S/'INPUTS_BOUND.json').read_text())
cache_path=P/f'formal_experiments_20261103/results/{B}/cache_physics_test.pt';cache=torch.load(cache_path,map_location='cpu',weights_only=True)
assert sha(cache_path)==bound['evaluation_cache_sha256'][B]['test']
texts=torch.load(S/f'shared/{B}/text_embeddings.pt',map_location='cpu',weights_only=True)['test']
original=json.loads((S/f'baselines/{B}/test.json').read_text());ids=cache['ids'];assert ids==original['gallery_ids']
scene=torch.load(S/f'baselines/{B}/test_scores.pt',map_location='cpu',weights_only=True)['scores']
def head_for(seed,variant):
    hp=S/f'results/{B}/seed_{seed}/{variant}/head.pt';ck=torch.load(hp,map_location='cpu',weights_only=True)
    assert ck['backbone']==B and ck['seed']==seed and ck['variant']==variant
    h=SAQREHead(cache['D']).cuda();h.load_state_dict(ck['state_dict']);h.ablate=set(ck['ablate']);h.eval()
    for p in h.parameters():p.requires_grad_(False)
    return h,hp
def provenance():return {'posthoc_exploratory':True,'backbone':B,'source_study':str(S),'script_sha256':sha(Path(__file__)),'test_cache_sha256':sha(cache_path),'source_protocol_sha256':sha(S/'protocol.json'),'training_performed':False,'gpu':torch.cuda.get_device_name(),'tf32':False}
def bounds(h):
    u=(h.U*h.type_scale).detach().cpu().double().numpy();gram=u@u.T
    opt=minimize(lambda a:float(a@gram@a),np.ones(3)/3,jac=lambda a:2*gram@a,bounds=[(0,1)]*3,constraints=[{'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(3)}],method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    assert opt.success
    return {'simplex_min_norm':float(np.sqrt(max(0,opt.fun))),'simplex_max_norm':float(np.linalg.norm(u,axis=1).max()),'min_coefficients':opt.x.tolist()}
@torch.no_grad()
def run_probe():
    train_path=P/f'formal_experiments_20261103/results/{B}/cache_physics_train.pt';train=torch.load(train_path,map_location='cpu',weights_only=True)
    mask=torch.arange(train['Mc'])[None,:]<train['c_n'][:,None];tf=train['c_tf'][mask].cuda().clone();tf[:,3]=0
    for seed in range(5):
        output=O/f'seed_{seed}.json'
        if output.exists():continue
        start=time.monotonic();h,hp=head_for(seed,'strict')
        baseline=json.loads((S/f'results/{B}/seed_{seed}/strict/test.json').read_text())
        if seed==0:
            test=score_head_matrix(h,texts,cache,move_docs,score_matrix,'cuda',32,512)
            rr=head_ranks(test.cuda(),scene.cuda(),original['gold_gallery_index'])
            assert rr==baseline['ranks'],'Unmodified evaluator did not reproduce every baseline rank'
            max_error=float((test-torch.load(S/f'results/{B}/seed_{seed}/strict/test_scores.pt',map_location='cpu',weights_only=True)['scores']).abs().max())
            write_json(O/'evaluator_check.json',{'n_ranks':len(rr),'all_equal':True,'max_score_abs_difference':max_error})
        pi=h.type_posterior(tf);mean_pi=pi.mean(0);norm_bounds=bounds(h)
        coeff_info={'train_regions':len(tf),'mean_pi':mean_pi.tolist(),'pi_sd':pi.std(0).tolist(),'dominant_counts':torch.bincount(pi.argmax(1),minlength=3).tolist(),**norm_bounds}
        def constant_augment(self,emb,tf):
            c=mean_pi.expand(*tf.shape[:-1],3);return F.normalize(emb+self.type_scale*(c@self.U),dim=-1),c
        h._augment_b=types.MethodType(constant_augment,h)
        scores=score_head_matrix(h,texts,cache,move_docs,score_matrix,'cuda',32,512)
        ranks=head_ranks(scores.cuda(),scene.cuda(),original['gold_gallery_index'])
        result={**provenance(),'seed':seed,'checkpoint_sha256':sha(hp),'intervention':'Replace each region posterior by the uniform-over-valid-training-regions mean posterior from this exact fitted head; retain region embeddings, reliability, gate and all learned parameters. No retraining.','posterior':coeff_info,'original_metrics':metrics(baseline['ranks']),'constant_pi_metrics':metrics(ranks),'ranks':ranks,'query_ids':original['query_ids'],'elapsed_seconds':time.monotonic()-start}
        result['delta_R10_constant_minus_original']=result['constant_pi_metrics']['R@10']-result['original_metrics']['R@10']
        write_json(output,result);print(B,seed,result['delta_R10_constant_minus_original'],flush=True)
@torch.no_grad()
def run_allcaptions():
    import open_clip
    from runtime import create_backbone
    from retrieval_queries import select_queries
    captions=load_captions(bound['common']['test_captions']['path']);records,coverage=select_queries(captions,ids,'all-captions')
    first,firstcov=select_queries(captions,ids,'first-caption');assert len(first)==3688 and len(records)==6190
    assert [r['query_id'] for r in first]==original['query_ids']
    extra=[r for r in records if r['caption_index_one_based']>1];assert len(extra)==2502
    ep=O/'extra_text_embeddings.pt';b=plan['backbones'][B]
    if not ep.exists():
        ext=None
        if B=='remoteclip':ext=P/'models/external_checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt'
        if B=='georsclip':ext=P/'models/external_checkpoints/georsclip/ckpt/RS5M_ViT-B-32.pt'
        model,_,preprocess=create_backbone(b['model'],b['pretrained'],ext);model=model.cuda().eval();tok=open_clip.get_tokenizer(b['model']);chunks=[]
        for start in range(0,len(extra),256):chunks.append(F.normalize(model.encode_text(tok([r['caption'] for r in extra[start:start+256]]).cuda()),dim=-1).cpu())
        vectors=torch.cat(chunks);torch.save(vectors,ep)
        write_json(O/'backbone_preprocessing.json',{'evaluation_transform':repr(preprocess),'model':b['model'],'pretrained':b['pretrained'],'external_checkpoint':str(ext) if ext else None})
        del model;torch.cuda.empty_cache()
    vectors=torch.load(ep,map_location='cpu',weights_only=True);gold=[r['gold_gallery_row'] for r in extra]
    extra_scene=(vectors.cuda()@F.normalize(cache['scene'].cuda(),dim=-1).T).cpu()
    from evaluate_multik import full_gallery_ranks
    # Scene baseline uses the original direct full-gallery tie convention.
    extra_scene_ranks=full_gallery_ranks(extra_scene.cuda(),gold)
    def combine(old,new):
        lookup={r['query_id']:v for r,v in zip(first,old)};lookup.update({r['query_id']:v for r,v in zip(extra,new)});return [lookup[r['query_id']] for r in records]
    br=combine(original['ranks'],extra_scene_ranks)
    write_json(O/'scene.json',{**provenance(),'variant':'scene','coverage':coverage,'ranks':br,'query_ids':[r['query_id'] for r in records],'metrics':metrics(br),'note':'First-caption ranks reused verbatim from original 3688-query experiment; 2502 additional caption rows encoded and evaluated with unchanged models. Each caption row has equal weight.'})
    for seed in range(5):
        for variant in ['full','strict','strict_token_off']:
            output=O/f'{variant}_seed_{seed}.json'
            if output.exists():continue
            start=time.monotonic();h,hp=head_for(seed,variant);scores=score_head_matrix(h,vectors,cache,move_docs,score_matrix,'cuda',32,512);r=head_ranks(scores.cuda(),extra_scene.cuda(),gold)
            old=json.loads((S/f'results/{B}/seed_{seed}/{variant}/test.json').read_text());ranks=combine(old['ranks'],r)
            write_json(output,{**provenance(),'seed':seed,'variant':variant,'checkpoint_sha256':sha(hp),'coverage':coverage,'ranks':ranks,'query_ids':[r['query_id'] for r in records],'metrics':metrics(ranks),'extra_only_metrics':metrics(r),'original_first_caption_R10':metrics(old['ranks'])['R@10'],'elapsed_seconds':time.monotonic()-start,'note':'First-caption ranks reused verbatim, additional rows independently evaluated under identical frozen model.'});print(B,variant,seed,metrics(ranks)['R@10'],flush=True)
if args.mode=='probe':run_probe()
else:run_allcaptions()
