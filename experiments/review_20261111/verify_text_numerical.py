"""Verify all new text encodings against the original frozen reference."""
import argparse,os,json,sys,importlib.metadata
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True);ap.add_argument('--gpu',default='4');a=ap.parse_args()
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace');S=P/'simplified_confirm_20261109_v1';O=P/'report_revision_diagnostics_20261111'/'input_verification';O.mkdir(exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,RS_RAG_MODEL_ROOT=str(P/'models'),HF_HOME=str(P/'models/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=[str(S/'base_code'),str(S/'code'),str(S/'reference')]
import torch,open_clip
import torch.nn.functional as F
from runtime import create_backbone
from artifacts import resolve_pretrained_path
from train_saqre import load_captions
from retrieval_queries import select_queries
from study_core import sha,write_json
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
tag=a.backbone;plan=json.loads((S/'protocol.json').read_text());bound=json.loads((S/'INPUTS_BOUND.json').read_text());b=plan['backbones'][tag];proof={}
for k,v in bound['common'].items():
    if k.endswith('captions') or k.endswith('_ids'):
        proof[k]=sha(v['path']);assert proof[k]==v['sha256']
for split in ['train','test']:
    cp=P/f'formal_experiments_20261103/results/{tag}/cache_physics_{split}.pt';proof[split+'_cache']=sha(cp)
    expected=bound['backbones'][tag]['files'][split+'_cache']['sha256'];assert proof[split+'_cache']==expected
wp=resolve_pretrained_path(b['model'],b['pretrained']);proof['pretrained_weight']=sha(wp);assert proof['pretrained_weight']==b['cache_signature']['pretrained_weight_sha256']
ext=None
if tag=='remoteclip':ext=P/'models/external_checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt'
if tag=='georsclip':ext=P/'models/external_checkpoints/georsclip/ckpt/RS5M_ViT-B-32.pt'
if ext:proof['external_checkpoint']=sha(ext);assert proof['external_checkpoint']==b['cache_signature']['external_checkpoint_sha256']
ids=json.loads(Path(bound['common']['test_ids']['path']).read_text());caps=load_captions(bound['common']['test_captions']['path']);first,_=select_queries(caps,ids);allcaps,_=select_queries(caps,ids,'all-captions');extra=[r for r in allcaps if r['caption_index_one_based']>1]
model,_,transform=create_backbone(b['model'],b['pretrained'],ext);model=model.cuda().eval();tokenizer=open_clip.get_tokenizer(b['model'])
@torch.no_grad()
def encode(rs):
    return torch.cat([F.normalize(model.encode_text(tokenizer([r['caption'] for r in rs[i:i+256]]).cuda()),dim=-1).cpu() for i in range(0,len(rs),256)])
old=torch.load(S/f'shared/{tag}/text_embeddings.pt',map_location='cpu',weights_only=True)['test'];new=encode(first);err=float((old-new).abs().max());assert err < 1e-5
cos64=float(F.cosine_similarity(old.double(),new.double(),dim=1).min());assert cos64 > 1-1e-10
print(tag,'first_max_abs_error',err,'first_min_cos_float64',cos64,flush=True)
ep=P/f'report_revision_diagnostics_20261111/allcaptions/{tag}/extra_text_embeddings.pt';oldextra=torch.load(ep,map_location='cpu',weights_only=True);newextra=encode(extra);extraerr=float((oldextra-newextra).abs().max());assert torch.equal(oldextra,newextra)
print(tag,'extra_error',extraerr,flush=True)
del model;torch.cuda.empty_cache()
from saqre_phys import SAQREHead
from train_saqre import move_docs
from batch_scoring import score_matrix
from evaluate_multik import score_head_matrix,full_gallery_ranks
from study_core import head_ranks
cache=torch.load(P/f'formal_experiments_20261103/results/{tag}/cache_physics_test.pt',map_location='cpu',weights_only=True)
gold=[r['gold_gallery_row'] for r in first];newscene=(new.cuda()@F.normalize(cache['scene'].cuda(),dim=-1).T).cpu()
sceneoriginal=json.loads((S/f'baselines/{tag}/test.json').read_text());sceneranks=full_gallery_ranks(newscene.cuda(),gold)
rankchecks={'scene':{'n_checked':3688,'rank_disagreements':sum(x!=y for x,y in zip(sceneranks,sceneoriginal['ranks'])),'R10_hit_disagreements':sum((x<=10)!=(y<=10) for x,y in zip(sceneranks,sceneoriginal['ranks']))}}
for variant in ['full','strict','strict_token_off']:
 ck=torch.load(S/f'results/{tag}/seed_0/{variant}/head.pt',map_location='cpu',weights_only=True);h=SAQREHead(cache['D']).cuda();h.load_state_dict(ck['state_dict']);h.ablate=set(ck['ablate']);h.eval()
 with torch.no_grad():matrix=score_head_matrix(h,new,cache,move_docs,score_matrix,'cuda',32,512);ranks=head_ranks(matrix.cuda(),newscene.cuda(),gold)
 original=json.loads((S/f'results/{tag}/seed_0/{variant}/test.json').read_text())['ranks'];rankchecks[variant+'_seed0']={'n_checked':3688,'rank_disagreements':sum(x!=y for x,y in zip(ranks,original)),'R10_hit_disagreements':sum((x<=10)!=(y<=10) for x,y in zip(ranks,original))}
print(tag,rankchecks,flush=True)
write_json(O/f'{tag}.json',{'backbone':tag,'all_3688_original_embeddings_equal':bool(torch.equal(old,new)), 'minimum_original_cosine_float64':cos64,'mean_original_abs_error':float((old-new).abs().mean()),'first_caption_rank_checks':rankchecks,'tolerance_note':'Same validated files and backbone. Floating-point check uses max abs <1e-5 and float64 cosine >1-1e-10; seed0 all3 model variants plus Scene are scored independently with freshly encoded first captions.','max_original_embedding_error':err,'all_2502_extra_embeddings_equal':bool(torch.equal(oldextra,newextra)),'max_extra_embedding_error':extraerr,'source_file_sha256':proof,'script_sha256':sha(Path(__file__)),'eval_preprocess':repr(transform),'packages':{n:importlib.metadata.version(n) for n in ['torch','open_clip_torch','numpy','Pillow','scipy','transformers']},'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'tf32':False})
print(tag,'input verification complete; first-caption equivalence is numerical, not bitwise',flush=True)
