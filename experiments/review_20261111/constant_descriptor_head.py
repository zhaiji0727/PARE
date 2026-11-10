"""Load the 2026-11-11 PARE-S constant-descriptor control faithfully.

This is a portability adapter, not the implementation used to train the saved
heads. Requires the same SAQREHead implementation as the published study.
"""
from pathlib import Path
import torch
from saqre_phys import SAQREHead

class ConstantDescriptorHead(SAQREHead):
    def __init__(self,dim,constant_descriptor):
        super().__init__(dim)
        value=torch.as_tensor(constant_descriptor,dtype=torch.float32)
        if value.shape!=(4,) or not torch.isfinite(value).all() or value[3]!=0:
            raise ValueError('Expected a finite four-vector with zero detector coordinate')
        self.register_buffer('constant_descriptor',value.clone(),persistent=False)
        self.ablate={'cfar_strict'}

    def _augment_b(self,emb,tf):
        return super()._augment_b(emb,self.constant_descriptor.to(tf).expand_as(tf))

    def _augment(self,emb,typefeat):
        return super()._augment(emb,self.constant_descriptor.to(typefeat).expand_as(typefeat))

def load_constant_descriptor_head(checkpoint,device='cpu'):
    checkpoint=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
    if checkpoint.get('variant')!='strict_constant_descriptor':
        raise ValueError('This loader is only for the constant-descriptor PARE-S control')
    state=checkpoint['state_dict'];dim=state['U'].shape[1]
    head=ConstantDescriptorHead(dim,checkpoint['constant_descriptor'])
    head.load_state_dict(state,strict=True)
    head=head.to(device).eval()
    for p in head.parameters():p.requires_grad_(False)
    return head
