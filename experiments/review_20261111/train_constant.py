"""Post-hoc fixed 25-head constant-input token control; no outcome-based selection."""
import os,sys,argparse,json,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True);ap.add_argument('--gpu',default='6');a=ap.parse_args()
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace');S=P/'simplified_confirm_20261109_v1';O=P/'report_revision_diagnostics_20261111'/'constant_training'/a.backbone;O.mkdir(parents=True,exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,RS_RAG_MODEL_ROOT=str(P/'models'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=[str(S/'base_code'),str(S/'reference'),str(S/'code')]
import torch
from saqre_phys import SAQREHead
from train_saqre import move_docs,load_captions
from batch_scoring import score_matrix
from evaluate_multik import score_head_matrix,metrics
from reference_kernels import train_one
from study_core import SETTINGS,training_pairs,training_orders,state_sha,head_ranks,sha,write_json,objsha,textsha
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
B=a.backbone;plan=json.loads((S/'protocol.json').read_text());bound=json.loads((S/'INPUTS_BOUND.json').read_text())
trainpath=P/f'formal_experiments_20261103/results/{B}/cache_physics_train.pt';testpath=P/f'formal_experiments_20261103/results/{B}/cache_physics_test.pt'
train=torch.load(trainpath,map_location='cpu',weights_only=True);test=torch.load(testpath,map_location='cpu',weights_only=True)
mask=torch.arange(train['Mc'])[None,:]<train['c_n'][:,None];valid=train['c_tf'][mask].clone();valid[:,3]=0;constant=valid.mean(0)
pairs=training_pairs(train['ids'],load_captions(bound['common']['train_captions']['path']));texts=sorted({t for _,t in pairs})
shared=torch.load(S/f'shared/{B}/text_embeddings.pt',map_location='cpu',weights_only=True);assert objsha([textsha(t) for t in texts])==shared['training_text_sha256']
cap2vec=dict(zip(texts,shared['train']));queries=shared['test'];original=json.loads((S/f'baselines/{B}/test.json').read_text());scene=torch.load(S/f'baselines/{B}/test_scores.pt',map_location='cpu',weights_only=True)['scores']
assert len(pairs)==31945 and len(train['ids'])==9181 and test['ids']==original['gallery_ids']
assert sha(testpath)==bound['evaluation_cache_sha256'][B]['test']
protocol={'study':'20261111 post-hoc constant-input token control','backbone':B,'seeds':[0,1,2,3,4],'settings':{**SETTINGS,'lr':plan['backbones'][B]['lr'],'kappa':plan['backbones'][B]['kappa']},'intervention':'Every valid coarse region uses the same train-only mean descriptor; coordinate 4 is zero. Existing h_psi, U, type_scale, reliability, scene/coarse gate, regions and all parameter shapes remain unchanged. Fine branch excluded exactly as in PARE-S.','constant_descriptor':constant.tolist(),'n_valid_train_regions':len(valid),'train_cache_sha256':sha(trainpath),'test_cache_sha256':sha(testpath),'script_sha256':sha(Path(__file__)),'original_protocol_sha256':sha(S/'protocol.json'),'selection':'No search, seed selection, epoch selection or outcome-triggered extension; evaluate all 25 heads. Exploratory reused benchmark; not a new unseen test.'}
if (O/'protocol.json').exists():assert json.loads((O/'protocol.json').read_text())==protocol
else:write_json(O/'protocol.json',protocol)
class ConstantHead(SAQREHead):
    def _augment_b(self,emb,tf):
        return super()._augment_b(emb,constant.to(tf.device).expand_as(tf))
for seed in range(5):
    folder=O/f'seed_{seed}'
    if (folder/'result.json').exists():continue
    if folder.exists():raise RuntimeError('Partial run retained; manual audit required before retry')
    folder.mkdir();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    h=ConstantHead(train['D']);h.ablate={'cfar_strict'};initial=state_sha(h.state_dict())
    legacy=json.loads((S/f'results/{B}/seed_{seed}/strict/run.json').read_text());assert initial==legacy['initial_state_sha256']
    h=h.cuda();orders=training_orders(seed,len(pairs),10);assert orders==legacy['expected_pair_order_sha256']
    def progress(history):write_json(folder/'training_progress.json',{'history':history})
    history=train_one(h,train,pairs,cap2vec,'cuda',seed,protocol['settings'],orders,progress)
    ck={'state_dict':{k:v.cpu() for k,v in h.state_dict().items()},'constant_descriptor':constant,'backbone':B,'seed':seed,'variant':'strict_constant_descriptor','initial_state_sha256':initial,'protocol_sha256':sha(O/'protocol.json'),'settings':protocol['settings']};torch.save(ck,folder/'head.pt')
    h.eval()
    with torch.no_grad():
        scores=score_head_matrix(h,queries,test,move_docs,score_matrix,'cuda',32,512)
        ranks=head_ranks(scores.cuda(),scene.cuda(),original['gold_gallery_index'])
    old=json.loads((S/f'results/{B}/seed_{seed}/strict/test.json').read_text());m=metrics(ranks);oldm=metrics(old['ranks'])
    write_json(folder/'result.json',{'posthoc_exploratory':True,'backbone':B,'seed':seed,'variant':'strict_constant_descriptor','protocol_sha256':sha(O/'protocol.json'),'checkpoint_sha256':sha(folder/'head.pt'),'initial_state_sha256':initial,'expected_pair_order_sha256':orders,'query_ids':original['query_ids'],'ranks':ranks,'metrics':m,'original_conditional_metrics':oldm,'delta_R10_conditional_minus_constant':oldm['R@10']-m['R@10'],'final_train_loss':history[-1]['mean_batch_loss']})
    print(B,seed,m['R@10'],'conditional_minus_constant',oldm['R@10']-m['R@10'],flush=True)
