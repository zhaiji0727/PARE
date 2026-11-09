"""Synthetic-only tests; no real data, encoding, or scientific experiment runs."""
import argparse
import copy
import importlib.util
import random
from pathlib import Path
import sys
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'base_code'),str(ROOT/'reference'),str(ROOT/'code')]
try:
    import torch
except ImportError:
    torch=None
if torch is not None:
    import torch.nn.functional as F
    if importlib.util.find_spec('open_clip') is None:
        stub=types.ModuleType('open_clip');stub.__spec__=importlib.machinery.ModuleSpec('open_clip',loader=None)
        sys.modules['open_clip']=stub
    from saqre_phys import SAQREHead
    from train_saqre import move_docs
    from batch_scoring import score_matrix
    from reference_kernels import train_one
    from evaluate_multik import score_head_matrix,rank_of_gold,full_gallery_ranks
from study_core import VARIANTS,training_orders,head_ranks,state_sha,same_tensor
DEVICE='cpu'

def fixture(dim=32):
    torch.manual_seed(83);n=4
    cache={'ids':[f'i{x}' for x in range(n)],'D':dim,'Mc':3,'Mf':4,
           'scene':F.normalize(torch.randn(n,dim),dim=-1),'agg':torch.randn(n,dim)*.2,
           'nu':torch.rand(n)*.2}
    for prefix,cap in [('c',3),('f',4)]:
        cache[prefix+'_emb']=F.normalize(torch.randn(n,cap,dim),dim=-1)
        cache[prefix+'_mu']=torch.rand(n,cap)
        cache[prefix+'_t']=torch.rand(n,cap)
        cache[prefix+'_tf']=torch.randn(n,cap,4)
        cache[prefix+'_n']=torch.tensor([0,cap,1,0])
    return cache,move_docs(cache,torch.arange(n),DEVICE),F.normalize(torch.randn(3,dim,device=DEVICE),dim=-1)

@unittest.skipIf(torch is None,'PyTorch is not installed locally; run on the prepared server runtime')
class NumericalTests(unittest.TestCase):
    def test_same_15_parameter_initialization_and_rng_512_768(self):
        for d in [512,768]:
            values=[];rngs=[]
            for variant,flags in VARIANTS.items():
                torch.manual_seed(19);head=SAQREHead(d);head.ablate=set(flags)
                self.assertEqual(len(dict(head.named_parameters())),15)
                values.append(state_sha(head.state_dict()));rngs.append(torch.get_rng_state())
            self.assertEqual(len(set(values)),1)
            self.assertTrue(all(torch.equal(rngs[0],r) for r in rngs))
    def test_strict_removes_all_fine_paths_and_detector_coordinate(self):
        _,docs,q=fixture();before={k:v.clone() for k,v in docs.items()}
        changed=copy.deepcopy(docs)
        for key in list(changed):
            if key.startswith('f_'):
                changed[key]=~changed[key] if changed[key].dtype==torch.bool else torch.randn_like(changed[key])*100
        changed['agg']=torch.randn_like(changed['agg'])*100
        changed['c_tf'][...,3]+=1000
        for variant in ['strict','strict_token_off']:
            torch.manual_seed(7);head=SAQREHead(32).to(DEVICE);head.ablate=set(VARIANTS[variant])
            a=score_matrix(head,q,docs);b=score_matrix(head,q,changed)
            self.assertTrue(torch.equal(a,b),variant)
            self.assertTrue(all(torch.equal(before[k],docs[k]) for k in docs))
    def test_token_off_only_has_no_token_gradients_reliability_remains(self):
        _,docs,q=fixture()
        torch.manual_seed(7);head=SAQREHead(32).to(DEVICE);head.ablate=set(VARIANTS['strict_token_off'])
        F.cross_entropy(score_matrix(head,q,docs)/.03,torch.arange(3,device=DEVICE)).backward()
        for name,p in head.named_parameters():
            if name in ['U','type_scale'] or name.startswith('h_psi.'):
                self.assertIsNone(p.grad,name)
        self.assertIsNotNone(head.log_beta.grad)
        self.assertTrue(torch.isfinite(head.log_beta.grad))
    def test_copied_kernel_matches_original_loop_two_epochs_with_tail(self):
        cache,_,_=fixture();pairs=[(0,'a'),(1,'b'),(2,'c'),(3,'d'),(0,'e')]
        torch.manual_seed(33);vectors={s:F.normalize(torch.randn(32),dim=-1) for _,s in pairs}
        settings=dict(lr=.002,kappa=.03,weight_decay=.0001,epochs=2,batch_size=3,batches_per_epoch=2,tail_batch_pairs=2)
        for variant,flags in VARIANTS.items():
            torch.manual_seed(17);old=SAQREHead(32).to(DEVICE);old.ablate=set(flags)
            torch.manual_seed(17);new=SAQREHead(32).to(DEVICE);new.ablate=set(flags)
            optimizer=torch.optim.AdamW(old.parameters(),lr=.002,weight_decay=.0001)
            permuted=list(pairs);rng=random.Random(4);reference_losses=[]
            for _ in range(2):
                rng.shuffle(permuted);total=0
                for i in range(0,len(permuted),3):
                    batch=permuted[i:i+3];rows=torch.tensor([r for r,_ in batch]);qs=torch.stack([vectors[c] for _,c in batch]).to(DEVICE)
                    scores=score_matrix(old,qs,move_docs(cache,rows,DEVICE))
                    loss=F.cross_entropy(scores/.03,torch.arange(len(batch),device=DEVICE))
                    optimizer.zero_grad();loss.backward();optimizer.step();total+=loss.item()
                reference_losses.append(total/2)
            history=train_one(new,cache,pairs,vectors,DEVICE,4,settings,training_orders(4,5,2),lambda h:None)
            self.assertEqual([x['mean_batch_loss'] for x in history],reference_losses)
            for k,v in old.state_dict().items():self.assertTrue(same_tensor(v,new.state_dict()[k]),(variant,k))
    def test_full_score_kernel_matches_query_loop_and_empty_masks(self):
        cache,docs,q=fixture()
        for variant,flags in VARIANTS.items():
            torch.manual_seed(7);head=SAQREHead(32).to(DEVICE);head.ablate=set(flags)
            with torch.no_grad():
                actual=score_head_matrix(head,q,cache,move_docs,score_matrix,DEVICE,2,2)
                expected=torch.stack([head.fuse_batch(query,docs)@query for query in q]).cpu()
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
            self.assertTrue(torch.isfinite(actual).all())
    def test_ties_keep_legacy_scene_order_then_one_dimensional_sort(self):
        n=20;scores=torch.ones((n,n),device=DEVICE)
        scene=torch.arange(n,device=DEVICE).float().repeat(n,1);gold=list(range(n))
        order=torch.argsort(scene,dim=1,descending=True)
        expected=[rank_of_gold(scores[i,order[i]],order[i].tolist(),g) for i,g in enumerate(gold)]
        actual=head_ranks(scores,scene,gold)
        self.assertEqual(actual,expected)
        self.assertNotEqual(actual,full_gallery_ranks(scores,gold))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args,rest=parser.parse_known_args()
    DEVICE=args.device
    if torch is not None:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    unittest.main(argv=[sys.argv[0]]+rest)
