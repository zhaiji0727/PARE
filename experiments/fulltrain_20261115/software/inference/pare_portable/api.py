"""Portable inference only. No backbone download, data discovery or training."""
import torch
from .spec import ModelSpec
from .io import load_tensor_artifact


def load_model(path,expected_sha256,model_spec,*,expected_backbone,device='cpu'):
    if not isinstance(model_spec,ModelSpec):model_spec=ModelSpec.parse(model_spec)
    if not expected_backbone:raise ValueError('Explicit backbone identity is required')
    checkpoint=load_tensor_artifact(path,expected_sha256)
    if not isinstance(checkpoint,dict) or not {'state_dict','metadata'}<=set(checkpoint):
        raise ValueError('Expected audited-style checkpoint with state_dict and metadata')
    state,meta=checkpoint['state_dict'],checkpoint['metadata']
    if meta.get('backbone')!=expected_backbone:raise ValueError('Checkpoint backbone identity differs')
    d=state['adapter.net.0.weight'].shape[1]
    if any(key in meta and meta[key]!=d for key in ['D','dimension','embedding_dim']):
        raise ValueError('Checkpoint dimension metadata contradicts its tensors')
    if state['physics.U'].shape!=(3,d) or state['coarse_mean'].shape!=(4,) or state['fine_mean'].shape!=(4,):
        raise ValueError('Checkpoint embedding/descriptor dimension differs')
    if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Checkpoint must contain finite FP32 tensors')
    # Construction is CPU-only and does not disturb a caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        model=model_spec.build(d,state['coarse_mean'],state['fine_mean'])
    model_spec.validate_metadata(meta,model)
    for buffer,key in [('coarse_mean','fit_crop_mean_c'),('fine_mean','fit_crop_mean_f')]:
        if key in meta and state[buffer].tolist()!=meta[key]:raise ValueError('Checkpoint mean metadata differs')
    model.load_state_dict(state,strict=True)
    if any(not torch.equal(value,model.state_dict()[name]) for name,value in state.items()):
        raise ValueError('Loaded model tensors differ from checkpoint')
    model.to(device).eval().requires_grad_(False)
    model.portable_spec=model_spec
    model.portable_checkpoint_sha256=expected_sha256
    return model,meta


@torch.inference_mode()
def score_queries(model,queries,documents,*,query_batch_size=8):
    """The same query chunks/vmap scores as the experiment, with no re-normalization."""
    if type(query_batch_size) is not int or query_batch_size<1:raise ValueError('Positive query batch size required')
    if not isinstance(queries,torch.Tensor) or queries.ndim!=2 or len(queries)==0 or queries.dtype!=torch.float32 or not torch.isfinite(queries).all():
        raise ValueError('Nonempty finite FP32 Q x D query tensor required')
    device=next(model.parameters()).device
    queries=queries.to(device)
    if queries.shape[1]!=documents['g'].shape[1]:raise ValueError('Query/cache embedding dimensions differ')
    if any(v.device!=device for v in documents.values()):raise ValueError('Documents and model must use the same device')
    chunks=[model(queries[i:i+query_batch_size],documents).cpu() for i in range(0,len(queries),query_batch_size)]
    result=torch.cat(chunks)
    if not torch.isfinite(result).all():raise FloatingPointError('Non-finite score')
    return result
