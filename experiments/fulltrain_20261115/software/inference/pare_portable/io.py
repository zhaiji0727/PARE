"""Path-independent, hash-bound inputs; nothing is loaded at import time."""
import hashlib
import json
from pathlib import Path
import re
import torch


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def require_file_hash(path,expected):
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
        raise ValueError('An explicit SHA256 is required for every loaded artifact')
    if file_sha256(path)!=expected:raise ValueError('Artifact SHA256 differs: '+str(path))


def load_tensor_artifact(path,expected_sha256):
    require_file_hash(path,expected_sha256)
    # Historical metadata may contain this inert str subclass. Never use
    # unrestricted pickle loading or enable arbitrary caller-supplied globals.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        value=torch.load(path,map_location='cpu',weights_only=True)
    require_file_hash(path,expected_sha256)
    return value


def load_json_artifact(path,expected_sha256):
    require_file_hash(path,expected_sha256)
    value=json.loads(Path(path).read_text())
    require_file_hash(path,expected_sha256)
    return value


def validate_input_contract(contract,contract_sha256,cache,queries,*,backbone,checkpoint_sha256,cache_sha256):
    """Validate an explicitly reviewed binding, not a guessed same-D identity.

    Queries carry this contract hash; it binds checkpoint, cache and the full
    parent encoder signature. It is provenance metadata, not proof that an
    arbitrary external producer actually generated an embedding correctly.
    """
    keys={'schema','backbone','dimension','checkpoint_sha256','cache_sha256',
          'cache_schema_version','cache_signature_sha256','embedding_signature_path','embedding_signature'}
    if set(contract)!=keys or contract['schema']!='pare.portable-input-binding.v1':
        raise ValueError('Expected an explicit portable input binding')
    if (contract['backbone'],contract['checkpoint_sha256'],contract['cache_sha256'])!=(backbone,checkpoint_sha256,cache_sha256):
        raise ValueError('Input binding does not match backbone/checkpoint/cache')
    if not isinstance(queries,dict) or set(queries)!={'embeddings','query_ids','input_contract_sha256'} or queries['input_contract_sha256']!=contract_sha256:
        raise ValueError('Query artifact must be explicitly exported against the same input binding')
    provenance=cache.get('provenance',{})
    signature=provenance.get('signature')
    if not isinstance(signature,dict):raise ValueError('Cache has no provenance.signature')
    signature_sha=object_sha256(signature)
    if provenance.get('schema_version')!=contract['cache_schema_version'] or provenance.get('signature_sha256')!=signature_sha or contract['cache_signature_sha256']!=signature_sha:
        raise ValueError('Cache schema/signature binding differs')
    path=contract['embedding_signature_path']
    if not isinstance(path,list) or not all(isinstance(key,str) for key in path):raise ValueError('Explicit encoder signature path required')
    encoder=signature
    for key in path:
        if not isinstance(encoder,dict) or key not in encoder:raise ValueError('Encoder signature path missing from cache')
        encoder=encoder[key]
    if not isinstance(encoder,dict) or encoder!=contract['embedding_signature'] or not {'model','pretrained','pretrained_weight_sha256','external_checkpoint_sha256'}<=set(encoder):
        raise ValueError('Encoder model/pretrained/weight identity differs')
    weight_hashes=[encoder['pretrained_weight_sha256'],encoder['external_checkpoint_sha256']]
    if not isinstance(encoder['model'],str) or not encoder['model'] or not any(weight_hashes) or any(value is not None and (not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value)) for value in weight_hashes):
        raise ValueError('Encoder identity requires a model name and explicit weight hash')
    if type(contract['dimension']) is not int or contract['dimension']!=cache['D'] or queries['embeddings'].ndim!=2 or queries['embeddings'].shape[1]!=cache['D']:
        raise ValueError('Bound embedding dimension differs')
    return {'schema_version':provenance['schema_version'],'signature':signature,'signature_sha256':signature_sha,'encoder_signature':encoder,'input_contract_sha256':contract_sha256}


def documents_from_cache(cache,gallery_ids,device='cpu'):
    """Same tensor indexing/count masks as original move_docs, in explicit ID order."""
    ids=cache['ids'];gallery_ids=list(gallery_ids)
    if not gallery_ids or len(set(ids))!=len(ids) or len(set(gallery_ids))!=len(gallery_ids):
        raise ValueError('Require nonempty unique gallery and unique cache IDs')
    index={name:i for i,name in enumerate(ids)}
    if any(name not in index for name in gallery_ids):raise ValueError('Gallery ID missing from cache')
    rows=torch.tensor([index[name] for name in gallery_ids],dtype=torch.long)
    n,d=len(ids),cache['D']
    if type(d) is not int or d<1 or cache['scene'].shape!=(n,d) or cache['agg'].shape!=(n,d) or cache['nu'].shape!=(n,):
        raise ValueError('Cache scene/aggregate/dimension shape differs')
    result={k:cache[source][rows].to(device) for k,source in [('g','scene'),('agg','agg'),('nu','nu')]}
    for prefix,capacity in [('c',cache['Mc']),('f',cache['Mf'])]:
        counts=cache[prefix+'_n']
        if type(capacity) is not int or capacity<1 or counts.shape!=(n,) or counts.dtype!=torch.int64 or torch.any((counts<0)|(counts>capacity)):
            raise ValueError('Invalid region capacity/counts')
        for suffix,shape in [('emb',(n,capacity,d)),('mu',(n,capacity)),('t',(n,capacity)),('tf',(n,capacity,4))]:
            tensor=cache[prefix+'_'+suffix]
            if not isinstance(tensor,torch.Tensor) or tuple(tensor.shape)!=shape:raise ValueError('Invalid region tensor shape')
            result[prefix+'_'+suffix]=tensor[rows].to(device)
        # This is the original count-mask expression, including region order.
        result[prefix+'_mask']=(torch.arange(capacity).unsqueeze(0)<counts[rows].unsqueeze(1)).to(device)
    for key,tensor in result.items():
        if tensor.dtype!=torch.bool and (tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all()):
            raise ValueError('Expected finite FP32 input: '+key)
    return result
