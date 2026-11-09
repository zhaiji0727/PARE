#!/usr/bin/env python3
"""Fixed 75-head simplified-method study; no adaptive selection or automatic retries.

preflight -> freeze -> anchor (each tag) -> train-all (each tag)
 -> evaluate (each tag) -> audit. Ten anchors are included in the 75 heads.
A process lock per backbone permits independent backbone jobs. Stage advancement
requires completed anchors for all five backbones before any remaining training.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
P = Path('<PRIVATE_PATH>')
sys.path[:0] = [str(ROOT/'base_code'), str(ROOT/'reference'), str(ROOT/'code')]
from study_core import (SEEDS,VARIANTS,BACKBONES,SETTINGS,sha,objsha,textsha,now,
                        write_json,state_sha,same_tensor,validate_plan,training_pairs,
                        training_orders,rank_metrics,head_ranks,evaluation_labels)


def source_files():
    paths = [p for folder in ('base_code','reference','code','scripts','tests','analysis_scripts')
             for p in (ROOT/folder).glob('*.py')]
    paths += [ROOT/'REFERENCE_COPY_PROVENANCE.json']
    paths += [p for p in (ROOT/'analysis_inputs').glob('*') if p.is_file()]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def setup(args):
    if args.gpu is not None:
        if args.gpu not in range(6):
            raise ValueError('GPU must identify one of the six existing devices')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.update(RS_RAG_MODEL_ROOT=str(args.workspace/'models'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def environment(torch):
    import importlib.metadata
    return {'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('torch','open_clip_torch','numpy','Pillow','scipy')},
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'tf32':False,'deterministic_algorithms':True}


def locate_inputs(args,plan,tag):
    from artifacts import resolve_pretrained_path
    p = args.workspace; b = plan['backbones'][tag]
    paths = {'train_cache':p/f'formal_experiments_20261103/results/{tag}/cache_physics_train.pt',
             'test_cache':p/f'formal_experiments_20261103/results/{tag}/cache_physics_test.pt',
             'pretrained_weight':resolve_pretrained_path(b['model'],b['pretrained']),
             'legacy_queries':p/f'evaluation_results_20261103/{tag}/queries.json',
             'legacy_text':p/f'evaluation_results_20261103/{tag}/query_embeddings.npy',
             'legacy_scene':p/f'evaluation_results_20261103/{tag}/scene_scores_physics.npy'}
    for variant,oldname in [('full','full'),('strict','cfar_strict')]:
        paths['legacy_head_'+variant] = Path(b['legacy_heads'][oldname]['path'])
        paths['legacy_scores_'+variant] = p/f'evaluation_results_20261103/{tag}/scores_{oldname}.npy'
        paths['legacy_result_'+variant] = p/f'evaluation_results_20261103/{tag}/eval_{oldname}_k3688.json'
    if tag in ('remoteclip','georsclip'):
        paths['external_checkpoint'] = p/('models/external_checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt' if tag == 'remoteclip' else 'models/external_checkpoints/georsclip/ckpt/RS5M_ViT-B-32.pt')
    if tag == 'vitb32':
        for label,factor in [('spacing_1p5','1.5'),('spacing_2','2')]:
            paths[label+'_cache'] = p/f'gsd_revision_20261103/results/vitb32/cache_spacing_{factor}.pt'
            paths[label+'_manifest'] = p/f'gsd_revision_20261103/results/vitb32/cache_spacing_{factor}.pt.manifest.json'
    return paths


def query_identity(records,ids):
    return {'query_ids':[r['query_id'] for r in records], 'query_image_ids':[r['image_id'] for r in records],
            'query_caption_sha256':[r['caption_sha256'] for r in records], 'gallery_ids':ids,
            'gold_gallery_index':[r['gold_gallery_row'] for r in records]}


def inspect_tag(torch,plan,tag,paths,common):
    from artifacts import validate_cache,validate_head
    from train_saqre import load_captions
    from retrieval_queries import select_queries
    from evaluate_multik import validate_tensor_layout
    from stress_cache import validate_scale_cache
    b = plan['backbones'][tag]; ext = paths.get('external_checkpoint')
    train = torch.load(paths['train_cache'],map_location='cpu',weights_only=True)
    test = torch.load(paths['test_cache'],map_location='cpu',weights_only=True)
    for label,cache,n in [('train',train,9181),('test',test,3688)]:
        signature = validate_cache(cache,b['model'],b['pretrained'],ext)
        if signature != b['cache_signature'] or len(cache['ids']) != n:
            raise ValueError('Cache signature or population differs: '+tag+'/'+label)
        validate_tensor_layout(cache)
        for name in ('L','c_star'):
            if tuple(cache[name].shape) != (n,) or cache[name].dtype != torch.float32 or not torch.isfinite(cache[name]).all():
                raise ValueError('Invalid reliability tensor: '+name)
        if cache['ids'] != json.loads(Path(common[label+'_ids']).read_text()):
            raise ValueError('Cache IDs/order differ from original split')
    if set(train['ids']) & set(test['ids']):
        raise ValueError('Training and test populations overlap')
    pairs = training_pairs(train['ids'],load_captions(common['train_captions']))
    records,_ = select_queries(load_captions(common['test_captions']),test['ids'])
    if len(pairs) != 31945 or len(records) != 3688 or len({r for r,_ in pairs}) != 9181:
        raise ValueError('Caption populations differ')
    previous = json.loads(Path(paths['legacy_queries']).read_text())
    if previous['gallery_ids'] != test['ids'] or previous['queries'] != records:
        raise ValueError('Legacy query identities differ')
    for variant,oldname in [('full','full'),('strict','cfar_strict')]:
        hp = paths['legacy_head_'+variant]
        if sha(hp) != b['legacy_heads'][oldname]['sha256']:
            raise ValueError('Legacy checkpoint bytes differ')
        head = torch.load(hp,map_location='cpu',weights_only=True)
        validate_head(head,train)
        c = head['config']
        if (head['n_images'],head['n_pairs'],c['epochs'],c['seed'],c['lr'],c['kappa'],c['bs'],c['text_bs'],head['ablate']) != (9181,31945,10,0,b['lr'],b['kappa'],64,256,VARIANTS[variant]):
            raise ValueError('Legacy training settings differ')
        if head['provenance']['train_cache_sha256'] != sha(paths['train_cache']) or head['provenance']['captions_sha256'] != sha(common['train_captions']):
            raise ValueError('Fresh and legacy training inputs differ')
    if tag == 'vitb32':
        for label,factor in [('spacing_1p5','1.5'),('spacing_2','2')]:
            scaled = torch.load(paths[label+'_cache'],map_location='cpu',weights_only=True)
            validate_scale_cache(scaled,test,sha(paths['test_cache']),factor,b['model'],b['pretrained'],ext)
            validate_tensor_layout(scaled)
    return {'D':train['D'],'n_train_images':9181,'n_train_pairs':len(pairs),
            'initial_pairs_sha256':objsha([[train['ids'][r],textsha(c)] for r,c in pairs]),
            'query_identity':query_identity(records,test['ids'])}


def validate_r2(plan,bound):
    r2 = plan['r2_bindings']
    groupfile = ROOT/r2['groups_file']
    if sha(groupfile) != r2['groups_sha256']:
        raise ValueError('Fixed R2 grouping file changed')
    provenance = json.loads(Path(bound['common']['raster_spacing_audit_provenance']['path']).read_text())
    if provenance['audit_sha256'] != bound['common']['raster_spacing_audit']['sha256']:
        raise ValueError('Raster audit provenance differs from its recorded bytes')
    groups = json.loads(groupfile.read_text())
    original = json.loads(Path(bound['common']['spacing_groups']['path']).read_text())
    for key in ('test_ids','test_ids_sha256','n_gallery_retained_for_all_subgroups','rounding',
                'bin_edges_m_per_pixel_after_rounding','groups','nominal_spacing_m_by_id','counts'):
        if groups[key] != original[key]:
            raise ValueError('Fixed R2 group definition differs from original: '+key)
    if groups['counts'] != r2['counts']:
        raise ValueError('Fixed R2 group counts differ')
    for localkey,binding in [('source_audit_sha256','raster_spacing_audit'),('source_grouping_code_sha256','spacing_groups_source')]:
        if groups[localkey] != bound['common'][binding]['sha256']:
            raise ValueError('R2 group source binding differs')
    if groups['source_queries_sha256'] != bound['backbones']['vitb32']['files']['legacy_queries']['sha256']:
        raise ValueError('R2 grouping uses different query provenance')
    for rel,digest in r2['scale_manifests'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('R2 scale manifest changed')
        key = 'spacing_1p5_manifest' if '1.5' in rel else 'spacing_2_manifest'
        if digest != bound['backbones']['vitb32']['files'][key]['sha256']:
            raise ValueError('Local and original scale manifests differ')


def preflight(args,plan,torch):
    if (ROOT/'SOURCE_FROZEN.json').exists() or (ROOT/'results').exists():
        raise FileExistsError('Preparation is not allowed after freezing or producing results')
    p = args.workspace
    common = {'train_captions':p/'data/ret_clean_20261007/Caption_train.csv',
              'test_captions':p/'data/ret_clean_20261007/Caption_test.csv',
              'train_ids':p/'data/ret_clean_20261007/train_ids.json',
              'test_ids':p/'data/ret_clean_20261007/test_ids.json',
              'spacing_groups':p/'gsd_revision_20261103/test_spacing_groups.json',
              'spacing_groups_source':p/'gsd_revision_20261103/spacing_groups.py',
              'raster_spacing_audit':ROOT/'analysis_inputs/raster_spacing_audit.json',
              'raster_spacing_audit_provenance':ROOT/'analysis_inputs/raster_spacing_audit_provenance.json'}
    bound = {'schema':'pare.simplified-inputs.v1','protocol_sha256':sha(ROOT/'protocol.json'),
             'protocol_markdown_sha256':sha(ROOT/'PROTOCOL.md'),'source_sha256':source_files(),
             'common':{k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in common.items()},'backbones':{}}
    for tag in BACKBONES:
        paths = locate_inputs(args,plan,tag)
        info = inspect_tag(torch,plan,tag,paths,common)
        bound['backbones'][tag] = {'files':{k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in paths.items()},**info}
        print(json.dumps({'preflight_tag_passed':tag,'D':info['D']}),flush=True)
    bound['evaluation_cache_sha256'] = {tag:{label:bound['backbones'][tag]['files'][('test' if label=='test' else label)+'_cache']['sha256'] for label in evaluation_labels(tag)} for tag in BACKBONES}
    validate_r2(plan,bound)
    write_json(ROOT/'INPUTS_BOUND.json',bound)


def check_frozen(plan,tags=(),rehash=True):
    validate_plan(plan)
    frozen = json.loads((ROOT/'SOURCE_FROZEN.json').read_text())
    expected = {'protocol_sha256':sha(ROOT/'protocol.json'),'protocol_markdown_sha256':sha(ROOT/'PROTOCOL.md'),
                'source_sha256':source_files(),'inputs_bound_sha256':sha(ROOT/'INPUTS_BOUND.json')}
    if any(frozen.get(k) != v for k,v in expected.items()):
        raise ValueError('Source, protocol or input bindings changed after freezing')
    bound = json.loads((ROOT/'INPUTS_BOUND.json').read_text())
    if any(bound[k] != expected[k] for k in ('protocol_sha256','protocol_markdown_sha256','source_sha256')):
        raise ValueError('Preflight is stale')
    if rehash:
        files = list(bound['common'].values())
        for tag in tags:
            files += list(bound['backbones'][tag]['files'].values())
        for entry in files:
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('Input bytes changed: '+entry['path'])
    validate_r2(plan,bound)
    return bound


def identity(tag,seed,variant):
    return {'backbone':tag,'seed':seed,'variant':variant,'protocol_sha256':sha(ROOT/'protocol.json'),
            'freeze_sha256':sha(ROOT/'SOURCE_FROZEN.json')}


def paths_from_bound(bound,tag):
    return ({k:Path(v['path']) for k,v in bound['backbones'][tag]['files'].items()},
            {k:Path(v['path']) for k,v in bound['common'].items()})


def shared_text(torch,plan,tag,bound):
    """Train: sorted unique TRAIN only; test: original records separately, bs256."""
    import numpy as np
    from train_saqre import load_captions
    from retrieval_queries import select_queries
    folder = ROOT/'shared'/tag
    paths,common = paths_from_bound(bound,tag)
    train = torch.load(paths['train_cache'],map_location='cpu',weights_only=True)
    test = torch.load(paths['test_cache'],map_location='cpu',weights_only=True)
    pairs = training_pairs(train['ids'],load_captions(common['train_captions']))
    records,_ = select_queries(load_captions(common['test_captions']),test['ids'])
    texts = sorted({caption for _,caption in pairs})
    binding = {'backbone':tag,'protocol_sha256':sha(ROOT/'protocol.json'),'freeze_sha256':sha(ROOT/'SOURCE_FROZEN.json'),
               'training_text_sha256':objsha([textsha(t) for t in texts]),
               'test_text_sha256':objsha([r['caption_sha256'] for r in records])}
    if (folder/'manifest.json').exists():
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['identity'] != binding or manifest['artifact_sha256'] != sha(folder/'text_embeddings.pt'):
            raise ValueError('Shared text artifact changed')
        data = torch.load(folder/'text_embeddings.pt',map_location='cpu',weights_only=True)
    else:
        if folder.exists():
            raise FileExistsError('Incomplete shared text encoding retained; inspect before retry')
        folder.mkdir(parents=True)
        import open_clip
        import torch.nn.functional as F
        from runtime import create_backbone
        b = plan['backbones'][tag]
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        model,_,_ = create_backbone(b['model'],b['pretrained'],paths.get('external_checkpoint'))
        model = model.cuda().eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(b['model'])
        def encode(strings):
            chunks = []
            with torch.no_grad():
                for start in range(0,len(strings),256):
                    tokens = tokenizer(strings[start:start+256]).cuda()
                    chunks.append(F.normalize(model.encode_text(tokens),dim=-1).cpu())
            return torch.cat(chunks)
        data = {**binding,'train':encode(texts),'test':encode([r['caption'] for r in records])}
        del model;torch.cuda.empty_cache()
        if not same_tensor(data['test'],torch.from_numpy(np.load(paths['legacy_text'],allow_pickle=False))):
            raise ValueError('Fresh test text vectors differ from 07 exact reference; inspect before training')
        if any(data[k].dtype != torch.float32 or not torch.isfinite(data[k]).all() for k in ('train','test')):
            raise ValueError('Non-finite or non-float32 text embeddings')
        torch.save(data,folder/'text_embeddings.pt')
        manifest = {'identity':binding,'artifact_sha256':sha(folder/'text_embeddings.pt'),
                    'train_tensor_sha256':state_sha({'train':data['train']}),
                    'test_tensor_sha256':state_sha({'test':data['test']}),'environment':environment(torch)}
        write_json(folder/'manifest.json',manifest)
    if any(data[k] != v for k,v in binding.items()):
        raise ValueError('Shared text inner identity differs')
    return train,test,pairs,records,dict(zip(texts,data['train'])),data['test'],manifest


def verify_run(tag,seed,variant):
    folder = ROOT/'results'/tag/f'seed_{seed}'/variant
    run = json.loads((folder/'run.json').read_text())
    if run['identity'] != identity(tag,seed,variant) or run['training_status'] != 'complete':
        raise ValueError('Missing, incomplete or mismatched training')
    for file,digest in run['training_artifacts_sha256'].items():
        if sha(folder/file) != digest:
            raise ValueError('Training artifact changed: '+str(folder/file))
    return folder,run


def train_variant(torch,plan,tag,bound,cache,pairs,cap2vec,tm,seed,variant):
    from saqre_phys import SAQREHead
    from reference_kernels import train_one
    folder = ROOT/'results'/tag/f'seed_{seed}'/variant
    if (folder/'run.json').exists():
        verify_run(tag,seed,variant);return
    if folder.exists():
        raise FileExistsError('Incomplete training retained; no automatic overwrite/retry')
    folder.mkdir(parents=True)
    settings = {**SETTINGS,'lr':plan['backbones'][tag]['lr'],'kappa':plan['backbones'][tag]['kappa']}
    run = {'schema':'pare.simplified-run.v1','identity':identity(tag,seed,variant),
           'training_status':'running','started_utc':now(),'evaluations':{}}
    write_json(folder/'run.json',run)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head = SAQREHead(cache['D']);head.ablate = set(VARIANTS[variant])
    initial = state_sha(head.state_dict());head = head.cuda()
    orders = training_orders(seed,len(pairs),10);states = []
    def progress(history):
        states.append({'epoch':history[-1]['epoch'],'parameter_state_sha256':state_sha(head.state_dict())})
        write_json(folder/'training_progress.json',{'history':history,'epoch_parameter_states':states,'updated_utc':now()})
    try:
        history = train_one(head,cache,pairs,cap2vec,'cuda',seed,settings,orders,progress)
        check_frozen(plan,rehash=False)
        provenance = {**identity(tag,seed,variant),'input_files':bound['backbones'][tag]['files'],
                      'common_input_files':bound['common'],'cache_signature':plan['backbones'][tag]['cache_signature'],
                      'train_ids':cache['ids'],'initial_state_sha256':initial,
                      'initial_pairs_sha256':bound['backbones'][tag]['initial_pairs_sha256'],
                      'expected_pair_order_sha256':orders,'text_embeddings_sha256':tm['train_tensor_sha256'],
                      'shared_text_artifact_sha256':tm['artifact_sha256'],'environment':environment(torch)}
        checkpoint = {'schema':'pare.simplified-head.v1','state_dict':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                      'D':cache['D'],'ablate':VARIANTS[variant],'seed':seed,'variant':variant,'backbone':tag,
                      'n_images':9181,'n_pairs':31945,'settings':settings,'history':history,'provenance':provenance}
        torch.save(checkpoint,folder/'head.pt')
        run.update(training_status='complete',completed_utc=now(),initial_state_sha256=initial,
                   final_parameter_sha256=state_sha(head.state_dict()),expected_pair_order_sha256=orders,
                   text_embeddings_sha256=tm['train_tensor_sha256'],
                   training_artifacts_sha256={f:sha(folder/f) for f in ('head.pt','training_progress.json')})
        write_json(folder/'run.json',run)
        print(json.dumps({'training_complete':tag,'seed':seed,'variant':variant}),flush=True)
    except BaseException as error:
        run.update(training_status='failed',failure={'time':now(),'type':type(error).__name__,'message':str(error)})
        write_json(folder/'run.json',run);raise
    finally:
        del head;torch.cuda.empty_cache()


def scene_result(torch,tag,label,cache,records,vectors,tm,bound):
    import torch.nn.functional as F
    from evaluate_multik import full_gallery_ranks
    folder = ROOT/'baselines'/tag
    folder.mkdir(parents=True,exist_ok=True)
    scores = (vectors.cuda() @ F.normalize(cache['scene'].cuda(),dim=-1).T).cpu()
    ids = query_identity(records,cache['ids'])
    common = {**identity(tag,None,'scene'),**ids,'condition':label,'evaluation_label':label,
              'evaluation_cache_sha256':bound['evaluation_cache_sha256'][tag][label],
              'text_embeddings_sha256':tm['test_tensor_sha256'],
              'n_queries':3688,'n_gallery':3688,'training':'none; frozen static baseline','posthoc_exploratory':True}
    file = folder/(label+'.json'); matrixfile = folder/(label+'_scores.pt')
    ranks = full_gallery_ranks(scores.cuda(),ids['gold_gallery_index'])
    if file.exists():
        old = json.loads(file.read_text())
        if any(old[k] != v for k,v in common.items()) or old['scores_artifact_sha256'] != sha(matrixfile):
            raise ValueError('Scene baseline identity changed')
        oldmatrix = torch.load(matrixfile,map_location='cpu',weights_only=True)
        if not same_tensor(oldmatrix['scores'],scores) or old['ranks'] != ranks:
            raise ValueError('Scene baseline recomputation differs')
    else:
        if matrixfile.exists():raise FileExistsError('Partial scene result retained')
        torch.save({**common,'scores':scores},matrixfile)
        write_json(file,{**common,'ranks':ranks,'metrics_percent':rank_metrics(ranks),'scores_artifact_sha256':sha(matrixfile)})
    return scores


def evaluate_variant(torch,plan,tag,label,cache,records,vectors,tm,scene,seed,variant,bound):
    from saqre_phys import SAQREHead
    from evaluate_multik import score_head_matrix
    from train_saqre import move_docs
    from batch_scoring import score_matrix
    folder,run = verify_run(tag,seed,variant)
    if label in run['evaluations']:
        for file,digest in run['evaluations'][label]['artifacts_sha256'].items():
            if sha(folder/file) != digest:raise ValueError('Completed evaluation changed')
        return
    if any((folder/(label+s)).exists() for s in ('.json','_scores.pt')):
        raise FileExistsError('Partial evaluation retained; inspect before retry')
    ck = torch.load(folder/'head.pt',map_location='cpu',weights_only=True)
    if (ck['backbone'],ck['seed'],ck['variant'],ck['ablate'],ck['n_images'],ck['n_pairs']) != (tag,seed,variant,VARIANTS[variant],9181,31945):
        raise ValueError('Wrong checkpoint identity')
    head = SAQREHead(cache['D']).cuda();head.load_state_dict(ck['state_dict'],strict=True)
    head.ablate = set(VARIANTS[variant]);head.eval()
    for parameter in head.parameters():parameter.requires_grad_(False)
    with torch.no_grad():
        matrix = score_head_matrix(head,vectors,cache,move_docs,score_matrix,'cuda',32,512)
        ranks = head_ranks(matrix.cuda(),scene.cuda(),[r['gold_gallery_row'] for r in records])
    common = {**identity(tag,seed,variant),**query_identity(records,cache['ids']),
              'condition':label,'evaluation_label':label,
              'evaluation_cache_sha256':bound['evaluation_cache_sha256'][tag][label],
              'checkpoint_sha256':sha(folder/'head.pt'),'text_embeddings_sha256':tm['test_tensor_sha256'],
              'posthoc_exploratory':True,'original_test_previously_seen':True}
    torch.save({**common,'scores':matrix},folder/(label+'_scores.pt'))
    write_json(folder/(label+'.json'),{**common,'n_queries':3688,'n_gallery':3688,'ranks':ranks,'metrics_percent':rank_metrics(ranks),
               'ranking':'07 K=N convention: CUDA 2D scene argsort, then per-query 1D head argsort in scene candidate order',
               'scores_artifact_sha256':sha(folder/(label+'_scores.pt'))})
    run['evaluations'][label] = {'artifacts_sha256':{f:sha(folder/f) for f in (label+'.json',label+'_scores.pt')}}
    write_json(folder/'run.json',run)
    print(json.dumps({'evaluation_complete':tag,'seed':seed,'variant':variant,'condition':label}),flush=True)
    del head,matrix;torch.cuda.empty_cache()


def verify_anchor(torch,tag,bound,write=False):
    import numpy as np
    paths,_ = paths_from_bound(bound,tag)
    checks = {};digests = {}
    for variant in ('full','strict'):
        folder,run = verify_run(tag,0,variant)
        new = torch.load(folder/'head.pt',map_location='cpu',weights_only=True)
        old = torch.load(paths['legacy_head_'+variant],map_location='cpu',weights_only=True)
        keys = set(new['state_dict']) == set(old['state_dict'])
        different = [k for k in new['state_dict'] if k not in old['state_dict'] or not same_tensor(new['state_dict'][k],old['state_dict'][k])]
        report = json.loads((folder/'test.json').read_text())
        legacy_report = json.loads(paths['legacy_result_'+variant].read_text())
        matrix = torch.load(folder/'test_scores.pt',map_location='cpu',weights_only=True)['scores']
        target = torch.from_numpy(np.load(paths['legacy_scores_'+variant],allow_pickle=False))
        exact_scores = same_tensor(matrix,target)
        oldranks = [r['natural_PARE_rank'] for r in legacy_report['queries']]
        check = {'same_parameter_keys':keys,'different_parameters':different,'parameter_count':len(new['state_dict']),
                 'max_absolute_parameter_difference':max(float((new['state_dict'][k]-old['state_dict'][k]).abs().max()) for k in old['state_dict']),
                 'epoch_losses_exact':[r['mean_batch_loss'] for r in new['history']] == [r['mean_batch_loss'] for r in old['history']],
                 'scores_exact':exact_scores,'max_absolute_score_difference':float((matrix-target).abs().max()),
                 'ranks_exact':report['ranks']==oldranks}
        check['passed'] = keys and not different and check['epoch_losses_exact'] and exact_scores and check['ranks_exact']
        checks[variant] = check
        digests[variant] = {'head.pt':sha(folder/'head.pt'),'test.json':sha(folder/'test.json'),'test_scores.pt':sha(folder/'test_scores.pt')}
    passed = all(v['passed'] for v in checks.values())
    report = {'status':'pass' if passed else 'fail','backbone':tag,'protocol_sha256':sha(ROOT/'protocol.json'),
              'freeze_sha256':sha(ROOT/'SOURCE_FROZEN.json'),'checks':checks,'new_artifacts_sha256':digests,
              'reference_files':{k:v for k,v in bound['backbones'][tag]['files'].items() if k.startswith('legacy_')}}
    if write:write_json(ROOT/'validation'/f'anchor_{tag}.json',report)
    if not passed:raise ValueError('Exact 07 anchor failed for '+tag+'; no further heads may run')
    return report


def require_all_anchors(torch,bound):
    for tag in BACKBONES:
        stored = json.loads((ROOT/'validation'/f'anchor_{tag}.json').read_text())
        if stored != verify_anchor(torch,tag,bound):raise ValueError('Anchor report differs: '+tag)


def audit(torch,plan,bound):
    require_all_anchors(torch,bound)
    checked = [];artifact_hashes = {}
    for tag in BACKBONES:
        paths,_ = paths_from_bound(bound,tag)
        qids = bound['backbones'][tag]['query_identity']
        for seed in SEEDS:
            initials=set();orders=set();texts=set()
            for variant in VARIANTS:
                folder,run=verify_run(tag,seed,variant)
                initials.add(run['initial_state_sha256']);orders.add(tuple(run['expected_pair_order_sha256']));texts.add(run['text_embeddings_sha256'])
                if set(run['evaluations']) != set(evaluation_labels(tag)):
                    raise ValueError('Missing or extra condition evaluations')
                for label in evaluation_labels(tag):
                    result=json.loads((folder/(label+'.json')).read_text())
                    matrix=torch.load(folder/(label+'_scores.pt'),map_location='cpu',weights_only=True)
                    for k,v in {**identity(tag,seed,variant),**qids,'condition':label,'evaluation_label':label,'evaluation_cache_sha256':bound['evaluation_cache_sha256'][tag][label],'checkpoint_sha256':sha(folder/'head.pt')}.items():
                        if result[k] != v or matrix[k] != v:raise ValueError('Result/matrix identity differs')
                    for file,digest in run['evaluations'][label]['artifacts_sha256'].items():
                        if sha(folder/file) != digest:raise ValueError('Evaluation artifact changed')
                    scores=matrix['scores']
                    if tuple(scores.shape)!=(3688,3688) or scores.dtype!=torch.float32 or not torch.isfinite(scores).all():
                        raise ValueError('Invalid score matrix')
                    scene_file=ROOT/'baselines'/tag/(label+'_scores.pt')
                    scene=torch.load(scene_file,map_location='cpu',weights_only=True)['scores']
                    ranks=head_ranks(scores.cuda(),scene.cuda(),qids['gold_gallery_index'])
                    if ranks!=result['ranks'] or rank_metrics(ranks)!=result['metrics_percent']:
                        raise ValueError('Ranks or metrics fail replay')
                    checked.append({'backbone':tag,'seed':seed,'variant':variant,'condition':label})
                for file in folder.iterdir():
                    if file.is_file():artifact_hashes[str(file.relative_to(ROOT))]=sha(file)
            if len(initials)!=1 or len(orders)!=1 or len(texts)!=1 or next(iter(orders)) != tuple(training_orders(seed,31945,10)):
                raise ValueError('Within-seed pairing failed')
    if len(checked)!=105:raise ValueError('Expected exactly 105 model evaluations')
    check_frozen(plan,BACKBONES)
    write_json(ROOT/'validation/completed_verification.json',{'status':'pass','heads':75,'evaluations':checked,
               'artifact_sha256':artifact_hashes,'protocol_sha256':sha(ROOT/'protocol.json'),
               'freeze_sha256':sha(ROOT/'SOURCE_FROZEN.json'),'completed_utc':now(),'original_test_previously_seen':True})
    print(json.dumps({'audit_passed':True,'heads':75,'evaluations':105}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','freeze','anchor','train-all','evaluate','audit'))
    parser.add_argument('--workspace',type=Path,default=P)
    parser.add_argument('--tag',choices=BACKBONES)
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.command in ('anchor','train-all','evaluate') and (args.tag is None or args.gpu is None):
        parser.error('This stage requires explicit --tag and --gpu')
    torch=setup(args);plan=json.loads((ROOT/'protocol.json').read_text());validate_plan(plan)
    (ROOT/'validation').mkdir(exist_ok=True)
    lockname=args.tag if args.command in ('anchor','train-all','evaluate') else 'global'
    with (ROOT/f'.{lockname}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='preflight':preflight(args,plan,torch);return
        if args.command=='freeze':
            bound=json.loads((ROOT/'INPUTS_BOUND.json').read_text())
            expected={'protocol_sha256':sha(ROOT/'protocol.json'),'protocol_markdown_sha256':sha(ROOT/'PROTOCOL.md'),
                      'source_sha256':source_files()}
            if any(bound[k]!=v for k,v in expected.items()):raise ValueError('Stale preflight')
            with (ROOT/'SOURCE_FROZEN.json').open('x') as stream:
                json.dump({**expected,'inputs_bound_sha256':sha(ROOT/'INPUTS_BOUND.json'),'created_utc':now()},stream,indent=2)
                stream.write('\n')
            check_frozen(plan,BACKBONES);return
        bound=check_frozen(plan,BACKBONES if args.command=='audit' else [args.tag])
        if args.command=='audit':audit(torch,plan,bound);return
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for exact numerical replication')
        if args.command=='train-all':require_all_anchors(torch,bound)
        train,test,pairs,records,cap2vec,vectors,tm=shared_text(torch,plan,args.tag,bound)
        if args.command in ('anchor','train-all'):
            grid=[(0,v) for v in ('full','strict')] if args.command=='anchor' else [(s,v) for s in SEEDS for v in VARIANTS]
            for seed,variant in grid:train_variant(torch,plan,args.tag,bound,train,pairs,cap2vec,tm,seed,variant)
            if args.command=='train-all':
                write_json(ROOT/'validation'/f'training_complete_{args.tag}.json',{'status':'complete','backbone':args.tag,'heads':15,
                           'protocol_sha256':sha(ROOT/'protocol.json'),'freeze_sha256':sha(ROOT/'SOURCE_FROZEN.json')})
        if args.command in ('anchor','evaluate'):
            if args.command=='evaluate':
                for tag in BACKBONES:
                    for seed in SEEDS:
                        for variant in VARIANTS:verify_run(tag,seed,variant)
            paths,_=paths_from_bound(bound,args.tag)
            labels=['test'] if args.command=='anchor' else evaluation_labels(args.tag)
            for label in labels:
                cache=test if label=='test' else torch.load(paths[label+'_cache'],map_location='cpu',weights_only=True)
                scene=scene_result(torch,args.tag,label,cache,records,vectors,tm,bound)
                grid=[(0,v) for v in ('full','strict')] if args.command=='anchor' else [(s,v) for s in SEEDS for v in VARIANTS]
                for seed,variant in grid:evaluate_variant(torch,plan,args.tag,label,cache,records,vectors,tm,scene,seed,variant,bound)
            if args.command=='anchor':verify_anchor(torch,args.tag,bound,write=True)
        check_frozen(plan,[args.tag])

if __name__=='__main__':main()
