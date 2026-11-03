"""Forward/gradient and timing regression against the actual released head."""
import argparse,copy,json,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F

ap=argparse.ArgumentParser();ap.add_argument('--code',required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--out',required=True)
args=ap.parse_args();sys.path.insert(0,args.code)
from saqre_phys import SAQREHead
from batch_scoring import score_matrix
torch.set_num_threads(4)
torch.manual_seed(42)
device=args.device
def make_docs(n,d,empty):
    x={'g':F.normalize(torch.randn(n,d,device=device),dim=-1),'agg':torch.randn(n,d,device=device)*.1,'nu':torch.rand(n,device=device)*.1}
    for pre,m in [('c',6),('f',8)]:
        x[pre+'_emb']=F.normalize(torch.randn(n,m,d,device=device),dim=-1)
        x[pre+'_mu']=torch.rand(n,m,device=device)
        x[pre+'_t']=torch.rand(n,m,device=device)
        x[pre+'_tf']=torch.randn(n,m,4,device=device)
        counts=torch.arange(n,device=device)%(m+1) if empty else torch.full((n,),m,device=device)
        x[pre+'_mask']=torch.arange(m,device=device)[None,:]<counts[:,None]
        for suffix in ['emb','mu','t','tf']:
            v=x[pre+'_'+suffix];mask=x[pre+'_mask']
            while mask.ndim<v.ndim:mask=mask.unsqueeze(-1)
            x[pre+'_'+suffix]=v.masked_fill(~mask,0)
    return x
rows=[]
for dim in [512,768]:
    for empty in [False,True]:
        for ablate in [set(),{'sr','ty'},{'cfar_strict'},{'cfar'}]:
            a=SAQREHead(dim).to(device);a.ablate=ablate;b=copy.deepcopy(a)
            docs=make_docs(7,dim,empty);q=F.normalize(torch.randn(5,dim,device=device),dim=-1)
            ref=torch.stack([a.fuse_batch(qi,docs)@qi for qi in q])
            out=score_matrix(b,q,docs)
            torch.testing.assert_close(out,ref,atol=2e-6,rtol=2e-5)
            target=torch.arange(5,device=device)
            la=F.cross_entropy(ref/.05,target);lb=F.cross_entropy(out/.05,target)
            la.backward();lb.backward()
            gradmax=0
            for (name,pa),(name2,pb) in zip(a.named_parameters(),b.named_parameters()):
                assert name==name2
                if pa.grad is None:assert pb.grad is None;continue
                assert torch.isfinite(pa.grad).all() and torch.isfinite(pb.grad).all(),name
                torch.testing.assert_close(pa.grad,pb.grad,atol=3e-6,rtol=3e-4)
                gradmax=max(gradmax,float((pa.grad-pb.grad).abs().max()))
            rows.append({'D':dim,'empty_rows':empty,'ablation':sorted(ablate),'max_score_error':float((out-ref).abs().max()),'max_gradient_error':gradmax})
if str(device).startswith('cuda'):
    dim=512;n=64;q=F.normalize(torch.randn(n,dim,device=device),dim=-1);docs=make_docs(n,dim,True)
    head=SAQREHead(dim).to(device);head.ablate=set()
    def run(vectorized):
        head.zero_grad(set_to_none=True)
        scores=score_matrix(head,q,docs) if vectorized else torch.stack([head.fuse_batch(qi,docs)@qi for qi in q])
        F.cross_entropy(scores/.05,torch.arange(n,device=device)).backward()
    times={}
    for mode in [False,True]:
        for _ in range(2):run(mode)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(5):run(mode)
        torch.cuda.synchronize();times['vmap' if mode else 'loop']=(time.perf_counter()-start)/5
else:times={}
result={'device':device,'cases':rows,'seconds_per_train_batch':times,'all_passed':True}
Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
