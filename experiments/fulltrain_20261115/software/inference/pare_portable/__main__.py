"""Explicit tensor scoring CLI; no dataset/backbone discovery or default model."""
import argparse
import os
from pathlib import Path
import torch
from .api import load_model,score_queries
from .io import documents_from_cache,load_tensor_artifact,load_json_artifact,file_sha256,object_sha256,validate_input_contract
from .spec import ModelSpec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,help='Explicit path:100:full or joint:joint_full, etc.')
    for name in ['checkpoint','cache','queries','gallery']:
        ap.add_argument('--'+name,type=Path,required=True)
        ap.add_argument('--'+name+'-sha256',required=True)
    ap.add_argument('--backbone',required=True)
    ap.add_argument('--input-contract',type=Path,required=True)
    ap.add_argument('--input-contract-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    ap.add_argument('--query-batch-size',type=int,default=8)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Refuse to overwrite score output')
    spec=ModelSpec.parse(args.model)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cache=load_tensor_artifact(args.cache,args.cache_sha256)
    queries=load_tensor_artifact(args.queries,args.queries_sha256)
    gallery=load_json_artifact(args.gallery,args.gallery_sha256)
    contract=load_json_artifact(args.input_contract,args.input_contract_sha256)
    cache_identity=validate_input_contract(contract,args.input_contract_sha256,cache,queries,backbone=args.backbone,checkpoint_sha256=args.checkpoint_sha256,cache_sha256=args.cache_sha256)
    if not isinstance(gallery,list) or not all(isinstance(x,str) for x in gallery):
        raise ValueError('Gallery JSON must be an explicit ordered list of IDs')
    ids=queries['query_ids']
    if not isinstance(ids,list) or not all(isinstance(x,str) for x in ids) or len(set(ids))!=len(ids) or len(ids)!=len(queries['embeddings']):
        raise ValueError('Query IDs must be unique and match the embedding row count')
    model,metadata=load_model(args.checkpoint,args.checkpoint_sha256,spec,expected_backbone=args.backbone,device=args.device)
    docs=documents_from_cache(cache,gallery,args.device)
    scores=score_queries(model,queries['embeddings'],docs,query_batch_size=args.query_batch_size)
    package=Path(__file__).parent
    provenance={'model_spec':spec.text(),'backbone':args.backbone,'input_sha256':{name:getattr(args,name+'_sha256') for name in ['checkpoint','cache','queries','gallery']},
                'query_ids_sha256':object_sha256(ids),'gallery_ids_sha256':object_sha256(gallery),
                'input_contract':contract,'cache_identity':cache_identity,'checkpoint_metadata':metadata,
                'package_source_sha256':{p.name:file_sha256(p) for p in sorted(package.glob('*.py'))},
                'runtime':{'torch':str(torch.__version__),'device':args.device,'tf32_matmul':False,'tf32_cudnn':False,'deterministic_algorithms':True,'query_batch_size':args.query_batch_size},
                'validation_scope':'Portable scoring operation only; package real-development parity and retraining replay require separate validation.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:torch.save({'scores':scores,'query_ids':ids,'gallery_ids':gallery,'provenance':provenance},stream)
    print(f'Saved explicit {scores.shape[0]} x {scores.shape[1]} scores to {args.output}')


if __name__=='__main__':main()
