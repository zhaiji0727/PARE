"""Extracted existing text loop and tensor identity. Device-only adaptation."""
import json
import torch
import torch.nn.functional as F

def encode_text(strings, model, tokenizer, device):
    chunks = []
    with torch.no_grad():
        for start in range(0,len(strings),256):
            tokens = tokenizer(strings[start:start+256]).to(device)
            chunks.append(F.normalize(model.encode_text(tokens),dim=-1).cpu())
    return torch.cat(chunks)

def state_sha(state):
    import hashlib
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(json.dumps([key,str(value.dtype),list(value.shape)]).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()
