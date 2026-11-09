"""Synthetic-only intervention, initialization, gradient and optimizer checks."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'base_code'),str(ROOT/'reference'),str(ROOT/'code')]
import torch
import torch.nn.functional as F
# The CPU-only unit environment may omit the text-encoder package. Its import
# is unused by move_docs/the frozen numerical kernel; real execution never
# uses this test-only placeholder, and any encoder call would fail immediately.
if importlib.util.find_spec('open_clip') is None:
    placeholder=types.ModuleType('open_clip')
    placeholder.__spec__=importlib.machinery.ModuleSpec('open_clip',loader=None)
    sys.modules['open_clip']=placeholder
from repair_head import RepairHead,RepairConfig
from token_mechanism_head import TokenMechanismHead,training_descriptor_means,VARIANTS
from batch_scoring import score_matrix
from reference_kernels import train_one
spec=importlib.util.spec_from_file_location('mechanism_runner_tests',ROOT/'scripts/run_mechanism.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
DEVICE='cpu'


def means_fixture():
    return {'branches':{'coarse':{'mean_float32':[.5,1.,1.5,.25]},
                        'fine':{'mean_float32':[1.5,.5,1.,.75]}}}


def data_fixture(dim=32):
    torch.manual_seed(183)
    n=4
    cache={'ids':[f'i{x}' for x in range(n)],'scene':F.normalize(torch.randn(n,dim),dim=-1),
           'agg':torch.randn(n,dim)*.2,'nu':torch.rand(n)*.3,'Mc':3,'Mf':4}
    for prefix,capacity in [('c',3),('f',4)]:
        cache[prefix+'_emb']=F.normalize(torch.randn(n,capacity,dim),dim=-1)
        cache[prefix+'_mu']=torch.rand(n,capacity)
        cache[prefix+'_t']=torch.rand(n,capacity)
        cache[prefix+'_tf']=torch.randn(n,capacity,4)
        cache[prefix+'_n']=torch.tensor([0,capacity,1,0])
    from train_saqre import move_docs
    docs=move_docs(cache,torch.arange(n),DEVICE)
    queries=F.normalize(torch.randn(3,dim,device=DEVICE),dim=-1)
    return cache,docs,queries


def heads(dim,variant):
    torch.manual_seed(71)
    old=RepairHead(dim,RepairConfig(A=1,B=0,C=1,T=int(variant!='token_off'))).to(DEVICE)
    torch.manual_seed(71)
    new=TokenMechanismHead(dim,variant,means_fixture()).to(DEVICE)
    return old,new


class MechanismTests(unittest.TestCase):
    def test_mean_is_training_only_valid_crop_weighted_float64(self):
        cache={'ids':['a','b','c','excluded']}
        for prefix in ['c','f']:
            features=torch.full((4,3,4),float('nan'))
            features[0,0]=torch.tensor([1e8,1.,2.,3.])
            features[0,1]=torch.tensor([1.,3.,4.,5.])
            features[1,0]=torch.tensor([-1e8,5.,6.,7.])
            cache[prefix+'_tf']=features;cache[prefix+'_n']=torch.tensor([2,1,0,3])
        result=training_descriptor_means(cache,['c','a','b'])
        for branch in ['coarse','fine']:
            r=result['branches'][branch]
            self.assertEqual(r['n_valid_crops'],3)
            self.assertEqual(r['mean_float64'],[1/3,3.,4.,5.])
            self.assertEqual(r['mean_float32'],torch.tensor([1/3,3.,4.,5.],dtype=torch.float32).tolist())
        altered=copy.deepcopy(cache)
        altered['c_tf'][3].fill_(99999)
        self.assertEqual(training_descriptor_means(altered,['a','b','c']),result)
        with self.assertRaises(ValueError):training_descriptor_means(cache,['a','a'])
        with self.assertRaises(ValueError):training_descriptor_means(cache,['missing'])
        cache['c_tf'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):training_descriptor_means(cache,['a'])

    def test_variants_have_identical_parameters_and_rng(self):
        for dim in [32,512]:
            states=[]
            for variant in VARIANTS:
                torch.manual_seed(17)
                reference=RepairHead(dim,RepairConfig(A=1,B=0,C=1))
                rng=torch.get_rng_state()
                torch.manual_seed(17)
                candidate=TokenMechanismHead(dim,variant,means_fixture())
                self.assertTrue(torch.equal(torch.get_rng_state(),rng))
                self.assertEqual(set(dict(reference.named_parameters())),set(dict(candidate.named_parameters())))
                for name,value in reference.state_dict().items():
                    self.assertEqual(value.numpy().tobytes(),candidate.state_dict()[name].numpy().tobytes(),name)
                self.assertTrue(all(not x.requires_grad for x in candidate.buffers()))
                states.append(runner.state_sha(candidate.legacy_parameter_state()))
            self.assertEqual(len(set(states)),1)

    def test_each_intervention_matches_direct_reference_and_gradients(self):
        for variant in VARIANTS:
            _,docs,queries=data_fixture(32)
            before={k:v.clone() for k,v in docs.items()}
            old,new=heads(32,variant)
            manual=dict(docs)
            if variant=='constant_descriptor':
                manual['c_tf']=new.constant_coarse_descriptor.expand_as(docs['c_tf'])
                manual['f_tf']=new.constant_fine_descriptor.expand_as(docs['f_tf'])
            a=score_matrix(old,queries,manual);b=score_matrix(new,queries,docs)
            self.assertTrue(torch.equal(a,b),variant)
            loss_a=F.cross_entropy(a/.03,torch.arange(3,device=DEVICE))
            loss_b=F.cross_entropy(b/.03,torch.arange(3,device=DEVICE))
            loss_a.backward();loss_b.backward()
            np=dict(new.named_parameters())
            for name,param in old.named_parameters():
                other=np[name]
                self.assertEqual(param.grad is None,other.grad is None,name)
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all())
                    self.assertTrue(torch.equal(param.grad,other.grad),name)
            self.assertTrue(all(torch.equal(before[k],docs[k]) for k in docs))
            if variant=='token_off':
                for name,param in new.named_parameters():
                    if name in ['U','type_scale'] or name.startswith('h_psi.'):
                        self.assertIsNone(param.grad,name)

    def test_constant_changes_no_reliability_or_gate_input(self):
        _,docs,queries=data_fixture(32)
        old,new=heads(32,'constant_descriptor')
        oldgate=[];newgate=[];oldrho=[];newrho=[]
        handles=[old.gate.register_forward_pre_hook(lambda m,a:oldgate.append(a[0].detach().clone())),
                 new.gate.register_forward_pre_hook(lambda m,a:newgate.append(a[0].detach().clone()))]
        old_attend=old._attend_variant;new_attend=new._attend_variant
        def left(q,e,r,m,b):oldrho.append(r.detach().clone());return old_attend(q,e,r,m,b)
        def right(q,e,r,m,b):newrho.append(r.detach().clone());return new_attend(q,e,r,m,b)
        old._attend_variant=left;new._attend_variant=right
        old.fuse_batch(queries[0],docs);new.fuse_batch(queries[0],docs)
        self.assertTrue(torch.equal(oldgate[0],newgate[0]))
        self.assertEqual(len(oldrho),2)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(oldrho,newrho)))
        for handle in handles:handle.remove()

    def test_two_epoch_optimizer_replays_conditional_exactly(self):
        cache,_,_=data_fixture(32)
        old,new=heads(32,'conditional')
        pairs=[(0,'a'),(1,'b'),(2,'c'),(3,'d'),(0,'e')]
        torch.manual_seed(33)
        vectors={s:F.normalize(torch.randn(32),dim=-1) for _,s in pairs}
        settings=dict(lr=.002,kappa=.03,weight_decay=.0001,epochs=2,batch_size=3,batches_per_epoch=2,tail_batch_pairs=2)
        expected=runner.training_orders(4,len(pairs),2)
        for head in [old,new]:train_one(head,cache,pairs,vectors,DEVICE,4,settings,expected,lambda h:None)
        for name,value in old.state_dict().items():
            self.assertEqual(value.cpu().numpy().tobytes(),new.state_dict()[name].cpu().numpy().tobytes(),name)

    def test_protocol_rejects_scientific_drift(self):
        plan=json.loads((ROOT/'protocol.json').read_text())
        runner.validate_plan(plan)
        for field,value in [('seeds',[0,1,2]),('variants',['conditional','token_off']),('expected_heads',16)]:
            altered=copy.deepcopy(plan);altered[field]=value
            with self.assertRaises(ValueError):runner.validate_plan(altered)
        altered=copy.deepcopy(plan);altered['fixed_model']['B']=1
        with self.assertRaises(ValueError):runner.validate_plan(altered)
        altered=copy.deepcopy(plan);altered['training']['lr']=.001
        with self.assertRaises(ValueError):runner.validate_plan(altered)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args,rest=parser.parse_known_args()
    DEVICE=args.device
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    unittest.main(argv=[sys.argv[0]]+rest)
