"""Full training backend, gated by an unbound protocol; no evaluation operations.

The public runner always initializes fresh. The internal resume argument exists
for exact continuity tests and later separately authorized recovery, not a CLI.
"""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
import time
from cache_common import BoundReader,require,sha
from data_contract import load_captions,training_index

ROOT=Path(__file__).resolve().parent


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def state_sha(state):
    import hashlib
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(json.dumps([key,str(value.dtype),list(value.shape)]).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def documents_from_cache(cache,device):
    import torch
    result={k:cache[source].to(device) for k,source in [('g','scene'),('agg','agg'),('nu','nu')]}
    for b,slots in [('c',cache['Mc']),('f',cache['Mf'])]:
        for suffix in ('emb','mu','t','tf'):result[b+'_'+suffix]=cache[b+'_'+suffix].to(device)
        result[b+'_mask']=(torch.arange(slots).unsqueeze(0)<cache[b+'_n'].unsqueeze(1)).to(device)
    return result


def full_region_means(documents):
    """Uniform over ALL valid regions, never a fit-only or test-derived mean."""
    import torch
    means=[];counts=[]
    for b in ('c','f'):
        values=documents[b+'_tf'][documents[b+'_mask']]
        require(len(values)>0,'No valid training regions in branch '+b)
        means.append(values.mean(0));counts.append(len(values))
    require(all(torch.isfinite(mean).all() for mean in means),'Nonfinite full-training region mean')
    return means,counts


def train_epochs(model,optimizer,documents,text_vectors,pairs,*,seed,epochs,kappa,metadata,
                 checkpoint_epochs,checkpoint_callback=None,progress_callback=None,resume_checkpoint=None):
    """Same batch/loss/update ordering as reviewed fixed80 runner; arbitrary pair count."""
    import torch
    from torch.nn import functional as F
    from resume_state import capture,restore,order_sha,assert_same
    require(pairs and type(epochs) is int and epochs>0 and math.isfinite(kappa) and kappa>0,'Training arguments')
    require(checkpoint_epochs==sorted(set(checkpoint_epochs)) and all(1<=e<=epochs for e in checkpoint_epochs),'Checkpoint schedule')
    device=text_vectors.device
    pairrows=torch.tensor([row for row,_ in pairs],device=device)
    pairtexts=torch.tensor([text for _,text in pairs],device=device)
    if resume_checkpoint is None:
        order=list(range(len(pairs)));rng=random.Random(seed);history=[]
    else:
        assert_same(resume_checkpoint['metadata'],metadata,'Resume metadata')
        order,rng,history=restore(resume_checkpoint,model,optimizer,seed=seed,n_pairs=len(pairs),
                                  epoch=len(resume_checkpoint['history']),device=device)
    start=time.monotonic()
    for epoch in range(len(history)+1,epochs+1):
        rng.shuffle(order);loss_sum=0.;nb=0;model.train()
        for off in range(0,len(order),64):
            b=torch.tensor(order[off:off+64],device=device);ri=pairrows[b]
            scores=model(text_vectors[pairtexts[b]],{key:value[ri] for key,value in documents.items()})
            loss=F.cross_entropy(scores/kappa,torch.arange(len(b),device=device))
            require(torch.isfinite(loss),'Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if epoch==1 and nb==0:
                require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite first gradients')
            optimizer.step();loss_sum+=loss.item();nb+=1
        require(nb==math.ceil(len(pairs)/64),'Batch count mismatch')
        history.append({'epoch':epoch,'loss':loss_sum/nb,'batches':nb,'pair_order_sha256':order_sha(order),
                        'elapsed_seconds':time.monotonic()-start})
        if progress_callback:progress_callback(history)
        if epoch in checkpoint_epochs:
            require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite checkpoint')
            checkpoint=capture(model,optimizer,metadata,history,order,rng,device)
            if checkpoint_callback:checkpoint_callback(epoch,checkpoint)
    return capture(model,optimizer,metadata,history,order,rng,device)


def execute_training(protocol,args):
    """Called only after the stdlib gate; no old checkpoint input is accepted."""
    from protocol_gate import validate_values
    validate_values(protocol,'train')
    require(args.backbone in protocol['backbones'] and args.model in protocol['models'] and args.seed in protocol['seeds'],'Run absent from frozen matrix')
    require(args.gpu is not None and args.gpu.isdigit() and int(args.gpu) in range(8),'Explicit GPU0..7 required')
    out=ROOT/'runs'/args.backbone/f'seed_{args.seed}'/args.model
    require(not out.exists(),'Existing run cannot be retried or overwritten')
    reader=BoundReader();reader.bind({'path':str(args.protocol.resolve()),'sha256':args.protocol_sha256})
    lock=reader.json(protocol['source_lock']);require(lock['status']=='FROZEN_FINAL_TRAINING_SOURCE','Final source lock status')
    required={'run_final.py','protocol_gate.py','data_contract.py','train_backend.py','training_caches.py','cache_common.py',
              'v9_train_cache.py','joint_model.py','model_factory.py','model_spec.py','matched_model_factory.py',
              'path_control_factory.py','path_isolation.py','resume_state.py','saqre_phys.py','scene_adapter.py','batch_scoring.py',
              'reference/v2_source.py','reference/old_retrieval_queries.py'}
    require(required<=set(lock['files']),'Final source coverage incomplete')
    for name,digest in lock['files'].items():
        path=ROOT/name;require(not Path(name).is_absolute() and ROOT in path.resolve().parents,'Unsafe source path')
        reader.bind({'path':str(path),'sha256':digest})
    decision=reader.json(protocol['decision_record']);require(decision['status']=='frozen_for_final_training','Final decision not frozen')
    for key in ('backbones','models','seeds','epochs','hyperparameters'):require(decision[key]==protocol[key],'Decision/protocol mismatch '+key)
    catalog=reader.json(protocol['input_catalog']);training=catalog['training']
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
    from training_caches import load_training_cache
    cache,cache_receipt=load_training_cache(training,args.backbone,reader)
    # Only CPU cache reading above. Evaluation catalog/IDs/texts are never accessed.
    import torch
    from matched_model_factory import build_matched_model
    from model_spec import parse_model_id
    torch.set_num_threads(3)
    require(str(torch.__version__)==protocol['expected_torch_version'],'Torch version differs')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One visible CUDA device required')
    require(torch.cuda.get_device_name()==protocol['expected_gpu_name'],'GPU model differs')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    row=training['backbones'][args.backbone]
    text_manifest=reader.json(row['shared_text_manifest']);text_path=reader.bind(row['shared_text'])
    require(text_manifest['artifact_sha256']==row['shared_text']['sha256'],'Shared text manifest binding')
    data=training_index(cache['ids'],load_captions(training['common']['train_captions']['path']))
    text=torch.load(text_path,map_location='cpu',weights_only=True)
    for key,value in text_manifest['identity'].items():require(text[key]==value,'Shared text identity '+key)
    require(text['backbone']==args.backbone and text['training_text_sha256']==data['training_text_sha256'],'Training text order')
    qmatrix=text['train']
    require(qmatrix.shape==(18461,cache['D']) and qmatrix.dtype==torch.float32 and torch.isfinite(qmatrix).all(),'Training text shape/dtype')
    require(state_sha({'train':qmatrix})==text_manifest['train_tensor_sha256'],'Training text tensor hash')
    # The historical dictionary also contains an unused first-caption tensor;
    # no test field is indexed for training, and no test score is computed.
    qmatrix=qmatrix.cuda();del text
    docs=documents_from_cache(cache,'cuda');means,counts=full_region_means(docs)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model=build_matched_model(cache['D'],args.model,*means).cuda();hp=protocol['hyperparameters'][args.backbone]
    optimizer=torch.optim.AdamW(model.active_parameters(),lr=hp['learning_rate'],weight_decay=.0001)
    metadata={'schema':'pare.final-full-train.run.v1','backbone':args.backbone,'dimension':cache['D'],'variant':args.model,'seed':args.seed,
              'epochs':protocol['epochs'],'checkpoint_epochs':protocol['checkpoint_epochs'],'evaluation_enabled':False,
              'train_images':9181,'train_pairs':31945,'training_texts':18461,'batches_per_epoch':500,'last_batch_size':9,
              'initialization':'fresh_from_seed','descriptor_mean_population':'all_valid_regions_of_all_9181_training_images',
              'full_training_crop_mean_c':means[0].tolist(),'full_training_crop_mean_f':means[1].tolist(),'fit_crop_mean_c':means[0].tolist(),'fit_crop_mean_f':means[1].tolist(),'descriptor_mean_counts':counts,
              'initial_adapter_sha256':state_sha(model.adapter.state_dict()),'initial_physics_sha256':state_sha(model.physics.state_dict()),
              'training_pair_order_sha256':data['pair_order_sha256'],'fit_pair_order_sha256':data['pair_order_sha256'],'training_text_sha256':data['training_text_sha256'],'train_ids_sha256':data['training_ids_sha256'],
              'model_spec':parse_model_id(args.model),'paths_enabled':getattr(model,'paths_enabled',None),'ablate':sorted(model.physics.ablate),'underlying_joint_variant':model.variant,
              'model_parameter_order':[n for n,_ in model.named_parameters()],
              'active_parameter_names':[n for n,p in model.named_parameters() if id(p) in {id(v) for v in model.active_parameters()}],
              'stored_parameters':sum(p.numel() for p in model.parameters()),'optimizer_parameters':sum(p.numel() for p in model.active_parameters()),
              'optimizer':{'name':'AdamW','lr':hp['learning_rate'],'weight_decay':.0001,'betas':[.9,.999],'eps':1e-8},'kappa':hp['kappa'],
              'protocol':protocol,'protocol_sha256':args.protocol_sha256,'source_lock':protocol['source_lock'],'decision_record':protocol['decision_record'],
              'input_catalog':protocol['input_catalog'],'input_sha256':dict(reader.files),'cache_validation':cache_receipt,
              'torch':str(torch.__version__),'gpu_name':torch.cuda.get_device_name(),'tf32_matmul':False,'tf32_cudnn':False,'deterministic_algorithms':True,
              'started_utc':datetime.now(timezone.utc).isoformat()}
    out.mkdir(parents=True,exist_ok=False)
    try:
        write(out/'INPUTS_VERIFIED.json',{'files':reader.files,'cache_validation':cache_receipt});write(out/'run.json',metadata)
        del cache
        def save_checkpoint(epoch,checkpoint):
            with (out/f'epoch_{epoch}.pt').open('xb') as stream:torch.save(checkpoint,stream)
        def save_progress(history):
            write(out/'progress.json',{'history':history});print(json.dumps({'backbone':args.backbone,'variant':args.model,'seed':args.seed,**history[-1]}),flush=True)
        train_epochs(model,optimizer,docs,qmatrix,data['pairs'],seed=args.seed,epochs=protocol['epochs'],kappa=hp['kappa'],metadata=metadata,
                     checkpoint_epochs=protocol['checkpoint_epochs'],checkpoint_callback=save_checkpoint,progress_callback=save_progress)
        reader.finish()
        write(out/'complete.json',{'status':'completed','backbone':args.backbone,'variant':args.model,'seed':args.seed,'epochs':protocol['epochs'],
                                  'checkpoint_epochs':protocol['checkpoint_epochs'],'evaluation_enabled':False,
                                  'artifact_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'ended_utc':datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        write(out/'failed.json',{'error_type':type(error).__name__,'message':str(error),'utc':datetime.now(timezone.utc).isoformat()});raise
