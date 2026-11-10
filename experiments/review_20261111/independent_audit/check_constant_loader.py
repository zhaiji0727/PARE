"""Small CPU portability check against the training script's exact intervention."""
from pathlib import Path
import sys,json,hashlib
P=Path('/home/coder/my_workspace/rs_paper/pare_workspace');S=P/'simplified_confirm_20261109_v1';R=P/'report_revision_diagnostics_20261111'
sys.path[:0]=[str(S/'base_code'),str(R)]
import torch
import torch.nn.functional as F
from saqre_phys import SAQREHead
from batch_scoring import score_matrix
from constant_descriptor_head import load_constant_descriptor_head

torch.set_num_threads(4);torch.manual_seed(125)
checks=[]
for path in sorted((R/'constant_training').glob('*/seed_*/head.pt')):
 ck=torch.load(path,map_location='cpu',weights_only=True);constant=ck['constant_descriptor']
 class ReferenceConstantHead(SAQREHead):
  def _augment_b(self,emb,tf):return super()._augment_b(emb,constant.to(tf.device).expand_as(tf))
 d=ck['state_dict']['U'].shape[1];ref=ReferenceConstantHead(d);ref.load_state_dict(ck['state_dict']);ref.ablate={'cfar_strict'};ref.eval();actual=load_constant_descriptor_head(path)
 n,mc,mf=4,6,8;cs=torch.tensor([6,4,1,0]);fs=torch.tensor([8,3,1,0])
 docs={'g':F.normalize(torch.randn(n,d),dim=-1),'agg':torch.randn(n,d),'nu':torch.rand(n),'c_emb':F.normalize(torch.randn(n,mc,d),dim=-1),'f_emb':F.normalize(torch.randn(n,mf,d),dim=-1),'c_mu':torch.rand(n,mc),'f_mu':torch.rand(n,mf),'c_t':torch.rand(n,mc),'f_t':torch.rand(n,mf),'c_tf':torch.rand(n,mc,4),'f_tf':torch.rand(n,mf,4),'c_mask':torch.arange(mc)[None,:]<cs[:,None],'f_mask':torch.arange(mf)[None,:]<fs[:,None]}
 q=F.normalize(torch.randn(5,d),dim=-1)
 with torch.no_grad():
  expected=score_matrix(ref,q,docs);got=score_matrix(actual,q,docs)
  exact=torch.equal(expected,got);error=float((expected-got).abs().max());assert exact
  per=[]
  for i in range(n):
   doc={'g':docs['g'][i],'agg':docs['agg'][i],'nu':docs['nu'][i]}
   for prefix,count in [('c',cs[i]),('f',fs[i])]:
    for suffix in ['emb','mu','t','tf']:doc[prefix+'_'+suffix]=docs[prefix+'_'+suffix][i,:count]
   per.append(torch.stack([actual(x,doc)@x for x in q]))
  scalar=torch.stack(per,dim=1);scalar_error=float((scalar-got).abs().max());assert scalar_error<1e-5
 checks.append({'backbone':ck['backbone'],'seed':ck['seed'],'head_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'batched_bitwise_equal':exact,'batched_max_abs_error':error,'scalar_vs_batched_max_abs_error':scalar_error})
print(json.dumps({'scope':'CPU small synthetic documents (including zero coarse regions) and all available measured constant-control heads; not a new retrieval benchmark','loader_sha256':hashlib.sha256((R/'constant_descriptor_head.py').read_bytes()).hexdigest(),'n_heads':len(checks),'checks':checks},indent=2))
