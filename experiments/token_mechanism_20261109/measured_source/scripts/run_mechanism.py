#!/usr/bin/env python3
"""Fixed 15-head token-conditioning study. Never accepts official-test inputs.

Stages: preflight, freeze, anchor, train-all, evaluate, audit.
The fresh conditional seed-0 anchor is part of the 15 heads. There is no
configuration selection and no success-based early stopping.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
P = Path('<PRIVATE_PATH>')
sys.path[:0] = [str(ROOT/'base_code'), str(ROOT/'reference'), str(ROOT/'code')]
spec = importlib.util.spec_from_file_location('mechanism_legacy_io', ROOT/'reference/repair_runner_original.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
sha, objsha, state_sha, write_json = legacy.sha, legacy.objsha, legacy.state_sha, legacy.write_json
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ['conditional', 'constant_descriptor', 'token_off']
FIXED_MODEL = dict(A=1, B=0, C=1, fine_crop_pixels=24, fusion='original', reliability='original', token_mode='original')
TRAINING = dict(lr=.002, kappa=.03, epochs=10, batch_size=64, weight_decay=.0001,
                optimizer='AdamW', precision='float32', n_train_images=6610,
                n_train_pairs=23016, batches_per_epoch=360, tail_batch_pairs=40)
ANCHOR_NAME = 'ABC101_T1_Roriginal_Tokoriginal'


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_plan(plan):
    if plan.get('schema') != 'pare.token-mechanism-study.v1':
        raise ValueError('Unexpected scientific protocol')
    if plan.get('seeds') != SEEDS or plan.get('variants') != VARIANTS or plan.get('expected_heads') != 15:
        raise ValueError('Exactly three variants and five fixed seeds required')
    if plan.get('fixed_model') != FIXED_MODEL:
        raise ValueError('Only token conditioning may change')
    if any(plan['training'].get(k) != v for k,v in TRAINING.items()):
        raise ValueError('Training recipe differs')
    if any(plan['training'].get(k, False) for k in ['amp','early_stopping','new_hyperparameter_search','warm_start']):
        raise ValueError('No AMP, selection, early stopping or warm start')
    if plan['backbone']['model'] != 'ViT-B-32' or plan['backbone']['pretrained'] != 'laion2b_s34b_b79k':
        raise ValueError('Wrong frozen backbone')
    for name,digest in plan['input_bindings']['reference_source_sha256'].items():
        if sha(ROOT/'base_code'/name) != digest:
            raise ValueError('Copied baseline source differs: '+name)


def source_files():
    files = [p for folder in ['base_code','reference','code','scripts','tests'] for p in (ROOT/folder).glob('*.py')]
    files += [ROOT/'REFERENCE_COPY_PROVENANCE.json', ROOT/'reference/REFERENCE_KERNEL_PROVENANCE.json']
    # A root-owned analysis script can be frozen before training too, without
    # making this runner responsible for inferential statistics.
    files += list((ROOT/'analysis_scripts').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def anchor_paths(args):
    return {'head':args.legacy_root/'results/seed_0'/ANCHOR_NAME/'head.pt',
            'development':args.legacy_root/'results/seed_0'/ANCHOR_NAME/'development.json',
            'development_scores':args.legacy_root/'results/seed_0'/ANCHOR_NAME/'development_scores.pt',
            'internal_check':args.legacy_root/'results/seed_0/internal_check'/ANCHOR_NAME/'internal_check.json',
            'internal_check_scores':args.legacy_root/'results/seed_0/internal_check'/ANCHOR_NAME/'internal_check_scores.pt'}


def exclusive_json(path, payload):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def preflight(args, plan, torch):
    from token_mechanism_head import training_descriptor_means
    from train_saqre import load_captions
    if args.freeze.exists() or args.out.exists():
        raise ValueError('Preflight preparation requires no source freeze or results yet')
    cache, signature, populations, paths, hashes = legacy.load_inputs(args, plan)
    means = training_descriptor_means(cache, populations['train'])
    means.update(training_ids_sha256=objsha(sorted(populations['train'])), training_cache_sha256=hashes['training_cache'])
    pairs, dev, gallery = legacy.pairs_for(cache, populations, load_captions(args.captions), 'train')
    _, internal, igallery = legacy.pairs_for(cache, populations, load_captions(args.captions), 'check')
    if (len(pairs),len(dev),len(gallery),len(internal),len(igallery)) != (23016,918,918,1653,1653):
        raise ValueError('Unexpected training/evaluation populations')
    references=anchor_paths(args)
    old=torch.load(references['head'],map_location='cpu',weights_only=True)
    if old['seed'] != 0 or old['n_images'] != 6610 or old['n_pairs'] != 23016:
        raise ValueError('Wrong reference anchor population')
    expected_config=dict(A=1,B=0,C=1,T=1,fusion='original',reliability='original',token_mode='original')
    if old['config'] != expected_config:
        raise ValueError('Wrong reference anchor configuration')
    if old['provenance']['input_sha256'] != hashes:
        raise ValueError('Reference anchor and new input hashes differ')
    for label, expected in [('development',populations['development']),('internal_check',populations['internal_check'])]:
        result=json.loads(references[label].read_text())
        if result['seed'] != 0 or set(result['query_ids']) != set(expected) or set(result['gallery_ids']) != set(expected):
            raise ValueError('Reference anchor result identities differ')
        if result['checkpoint_sha256'] != sha(references['head']):
            raise ValueError('Reference result/checkpoint binding differs')
    write_json(ROOT/'training_descriptor_means.json',means)
    bound={'schema':'pare.token-mechanism-inputs.v1','protocol_sha256':sha(args.protocol),
           'protocol_markdown_sha256':sha(ROOT/'PROTOCOL.md'),'source_sha256':source_files(),
           'paths':{k:str(v.resolve()) for k,v in paths.items()},'input_sha256':hashes,
           'means_sha256':sha(ROOT/'training_descriptor_means.json'),
           'anchor_paths':{k:str(v.resolve()) for k,v in references.items()},
           'anchor_sha256':{k:sha(v) for k,v in references.items()},
           'pair_count':len(pairs),'development_count':len(dev),'internal_check_count':len(internal),
           'official_test_accessed':False}
    write_json(ROOT/'INPUTS_BOUND.json',bound)
    print(json.dumps({'preflight_passed':True,'training_pairs':len(pairs),'mean_crop_counts':{k:v['n_valid_crops'] for k,v in means['branches'].items()}}),flush=True)


def check_frozen(args, plan):
    validate_plan(plan)
    frozen=json.loads(args.freeze.read_text())
    expected={'protocol_sha256':sha(args.protocol),'protocol_markdown_sha256':sha(ROOT/'PROTOCOL.md'),
              'source_sha256':source_files(),'inputs_bound_sha256':sha(ROOT/'INPUTS_BOUND.json'),
              'means_sha256':sha(ROOT/'training_descriptor_means.json')}
    if any(frozen.get(k) != v for k,v in expected.items()):
        raise ValueError('Reviewed source, protocol or bound inputs changed')
    bound=json.loads((ROOT/'INPUTS_BOUND.json').read_text())
    if bound['protocol_sha256'] != expected['protocol_sha256'] or bound['source_sha256'] != expected['source_sha256']:
        raise ValueError('Preflight is stale')
    for k,path in bound['paths'].items():
        if sha(path) != bound['input_sha256'][k]:
            raise ValueError('Input file changed: '+k)
    for k,path in bound['anchor_paths'].items():
        if sha(path) != bound['anchor_sha256'][k]:
            raise ValueError('Reference anchor changed: '+k)
    return frozen,bound


def load_bound(args, plan):
    from token_mechanism_head import training_descriptor_means
    frozen,bound=check_frozen(args,plan)
    cache,signature,populations,paths,hashes=legacy.load_inputs(args,plan)
    if hashes != bound['input_sha256']:
        raise ValueError('Loaded input hashes differ')
    means=json.loads((ROOT/'training_descriptor_means.json').read_text())
    computed=training_descriptor_means(cache,populations['train'])
    computed.update(training_ids_sha256=objsha(sorted(populations['train'])),training_cache_sha256=hashes['training_cache'])
    if means != computed:
        raise ValueError('Training-only descriptor means failed recomputation')
    return cache,signature,populations,means,frozen,bound


def shared_data(cache,populations,args):
    from train_saqre import load_captions
    caps=load_captions(args.captions)
    pairs,dev,dgallery=legacy.pairs_for(cache,populations,caps,'train')
    _,internal,igallery=legacy.pairs_for(cache,populations,caps,'check')
    return pairs,{'development':(dev,dgallery),'internal_check':(internal,igallery)}


def identity(args,seed,variant):
    return {'seed':seed,'variant':variant,'protocol_sha256':sha(args.protocol),
            'freeze_sha256':sha(args.freeze),'means_sha256':sha(ROOT/'training_descriptor_means.json')}


def verify_run(args,seed,variant):
    folder=args.out/f'seed_{seed}'/variant
    report=json.loads((folder/'run.json').read_text())
    if report['identity'] != identity(args,seed,variant) or report['training_status'] != 'complete':
        raise ValueError('Incomplete or mismatched training run')
    for file,digest in report['training_artifacts_sha256'].items():
        if sha(folder/file) != digest:
            raise ValueError('Training artifact changed: '+str(folder/file))
    return folder,report


def training_orders(seed,n,epochs):
    order=list(range(n)); rng=random.Random(seed); hashes=[]
    for _ in range(epochs):
        rng.shuffle(order);hashes.append(objsha(order))
    return hashes


def train_one_variant(args,plan,torch,cache,signature,populations,means,pairs,cap2vec,textsha,seed,variant):
    from saqre_phys import SAQREHead
    from token_mechanism_head import TokenMechanismHead
    from reference_kernels import train_one
    folder=args.out/f'seed_{seed}'/variant
    if (folder/'run.json').exists():
        try:
            verify_run(args,seed,variant)
            return
        except (ValueError,FileNotFoundError,KeyError):
            if not args.retry_incomplete:raise
    if folder.exists():
        if not args.retry_incomplete:
            raise FileExistsError('Partial run exists; explicit --retry-incomplete preserves and retries it')
        archive=args.out/'failed_attempts'/f'seed_{seed}_{variant}_{time.time_ns()}'
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(folder),archive)
    folder.mkdir(parents=True)
    report={'schema':'pare.token-mechanism-run.v1','identity':identity(args,seed,variant),
            'training_status':'running','started_utc':now(),'evaluations':{}}
    write_json(folder/'run.json',report)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    base=SAQREHead(512)
    base_state={k:v.detach().clone() for k,v in base.state_dict().items()};del base
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=TokenMechanismHead(512,variant,means)
    head.load_base_state_dict(base_state)
    initial_base=state_sha(base_state);initial_parameters=state_sha(head.legacy_parameter_state())
    head=head.cuda()
    expected=training_orders(seed,len(pairs),plan['training']['epochs'])
    epochs=[]
    def progress(history):
        epochs.append({'epoch':history[-1]['epoch'],'parameter_state_sha256':state_sha(head.legacy_parameter_state())})
        write_json(folder/'training_progress.json',{'history':history,'epoch_parameter_states':epochs,'updated_utc':now()})
    try:
        history=train_one(head,cache,pairs,cap2vec,'cuda',seed,plan['training'],expected,progress)
        check_frozen(args,plan)
        provenance={**identity(args,seed,variant),'input_sha256':json.loads((ROOT/'INPUTS_BOUND.json').read_text())['input_sha256'],
                    'train_ids':sorted(populations['train']),'development_ids':sorted(populations['development']),
                    'excluded_internal_check_ids':sorted(populations['internal_check']),'cache_signature':signature,
                    'base_initial_state_sha256':initial_base,'legacy_initial_parameter_sha256':initial_parameters,
                    'text_embeddings_sha256':textsha,
                    'initial_pairs_sha256':objsha([[cache['ids'][row],hashlib.sha256(text.encode()).hexdigest()] for row,text in pairs]),
                    'expected_pair_order_sha256':expected,'environment':legacy.environment(torch)}
        saved={'schema':'pare.token-mechanism-head.v1','state_dict':{k:v.detach().cpu() for k,v in head.state_dict().items()},
               'D':512,'variant':variant,'config':head.config.to_dict(),'seed':seed,'n_images':6610,'n_pairs':len(pairs),
               'settings':plan['training'],'history':history,'descriptor_means':means,'provenance':provenance}
        torch.save(saved,folder/'head.pt')
        report.update(training_status='complete',completed_utc=now(),base_initial_state_sha256=initial_base,
                      legacy_initial_parameter_sha256=initial_parameters,final_parameter_sha256=state_sha(head.legacy_parameter_state()),
                      text_embeddings_sha256=textsha,expected_pair_order_sha256=expected,
                      training_artifacts_sha256={name:sha(folder/name) for name in ['head.pt','training_progress.json']})
        write_json(folder/'run.json',report)
        print(json.dumps({'training_complete':variant,'seed':seed}),flush=True)
    except BaseException as error:
        report.update(training_status='failed',failure={'time':now(),'type':type(error).__name__,'message':str(error)})
        write_json(folder/'run.json',report)
        raise
    finally:
        del head
        torch.cuda.empty_cache()


def evaluate_variant(args,plan,torch,cache,means,evaldata,embeddings,seed,variant):
    from token_mechanism_head import TokenMechanismHead
    from reference_kernels import validate_one
    folder,report=verify_run(args,seed,variant)
    checkpoint=torch.load(folder/'head.pt',map_location='cpu',weights_only=True)
    if (checkpoint.get('seed') != seed or checkpoint.get('variant') != variant
            or checkpoint.get('n_images') != 6610 or checkpoint.get('n_pairs') != 23016
            or checkpoint.get('descriptor_means') != means
            or any(checkpoint.get('provenance',{}).get(k) != v for k,v in identity(args,seed,variant).items())):
        raise ValueError('Checkpoint identity, training population or fixed means differ')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=TokenMechanismHead(512,variant,means).cuda()
    head.load_state_dict(checkpoint['state_dict'],strict=True)
    for label,(queries,gallery) in evaldata.items():
        if label in report['evaluations']:
            for file,digest in report['evaluations'][label]['artifacts_sha256'].items():
                if sha(folder/file) != digest:raise ValueError('Completed evaluation changed')
            continue
        if (folder/(label+'.json')).exists() or (folder/(label+'_scores.pt')).exists():
            raise FileExistsError('Partial evaluation retained; inspect before retry')
        vectors,textsha=embeddings[label]
        matrix,ranks,metrics,gold=validate_one(head,cache,gallery,queries,vectors,'cuda')
        ids={'query_ids':[cache['ids'][row] for row,_ in queries],
             'query_caption_sha256':[hashlib.sha256(text.encode()).hexdigest() for _,text in queries],
             'gallery_ids':[cache['ids'][row] for row in gallery],'gold_gallery_index':gold}
        common={**identity(args,seed,variant),**ids,'checkpoint_sha256':sha(folder/'head.pt'),
                'text_embeddings_sha256':textsha,'posthoc_exploratory':True,'official_test_accessed':False}
        torch.save({**common,'scores':matrix},folder/(label+'_scores.pt'))
        write_json(folder/(label+'.json'),{**common,'ranks':ranks,'metrics_percent':metrics,
                   'n_queries':len(queries),'n_gallery':len(gallery),'ranking':'torch.argsort descending CUDA; full within-split gallery'})
        report['evaluations'][label]={'metrics_percent':metrics,'artifacts_sha256':{name:sha(folder/name) for name in [label+'.json',label+'_scores.pt']}}
        write_json(folder/'run.json',report)
        print(json.dumps({'evaluation_complete':variant,'seed':seed,'split':label,'metrics_percent':metrics}),flush=True)
        del matrix
    del head
    torch.cuda.empty_cache()


def verify_anchor(args,plan,torch):
    from token_mechanism_head import TokenMechanismHead
    folder,_=verify_run(args,0,'conditional');paths=anchor_paths(args)
    old=torch.load(paths['head'],map_location='cpu',weights_only=True)
    new=torch.load(folder/'head.pt',map_location='cpu',weights_only=True)
    current={k:v for k,v in new['state_dict'].items() if k not in TokenMechanismHead.MEAN_BUFFERS}
    same_keys=set(current)==set(old['state_dict'])
    different=[];maxdelta=0.
    if same_keys:
        for key,value in current.items():
            target=old['state_dict'][key]
            if value.dtype != target.dtype or value.shape != target.shape or value.contiguous().numpy().tobytes() != target.contiguous().numpy().tobytes():
                different.append(key)
            if value.shape==target.shape:maxdelta=max(maxdelta,float((value-target).abs().max()))
    split_checks={};score_differences={}
    new_artifacts={'head.pt':sha(folder/'head.pt')}
    for label in ['development','internal_check']:
        a=json.loads((folder/(label+'.json')).read_text());b=json.loads(paths[label].read_text())
        fields=['query_ids','query_caption_sha256','gallery_ids','gold_gallery_index','ranks','metrics_percent','text_embeddings_sha256']
        # Legacy evaluation JSON does not always repeat the text embedding hash.
        compared=[key for key in fields if key in b]
        split_checks[label]={key:a[key]==b[key] for key in compared}
        old_matrix=torch.load(paths[label+'_scores'],map_location='cpu',weights_only=True)['scores']
        new_matrix=torch.load(folder/(label+'_scores.pt'),map_location='cpu',weights_only=True)['scores']
        matrix_shape_dtype=old_matrix.shape==new_matrix.shape and old_matrix.dtype==new_matrix.dtype
        split_checks[label]['score_shape_dtype']=matrix_shape_dtype
        split_checks[label]['score_bytes_identical']=matrix_shape_dtype and old_matrix.contiguous().numpy().tobytes()==new_matrix.contiguous().numpy().tobytes()
        score_differences[label]=float((old_matrix-new_matrix).abs().max()) if matrix_shape_dtype else None
        new_artifacts[label+'.json']=sha(folder/(label+'.json'))
        new_artifacts[label+'_scores.pt']=sha(folder/(label+'_scores.pt'))
    pairs_match=new['provenance']['initial_pairs_sha256']==old['provenance']['initial_pairs_sha256']
    initial_match=new['provenance']['base_initial_state_sha256']==old['provenance']['base_initial_state_sha256']
    text_match=new['provenance']['text_embeddings_sha256']==old['provenance']['text_embeddings_sha256']
    order_match=[x['pair_order_sha256'] for x in new['history']]==[x['pair_order_sha256'] for x in old['history']]
    passed=same_keys and not different and pairs_match and initial_match and text_match and order_match and all(all(x.values()) for x in split_checks.values())
    report={'status':'pass' if passed else 'fail','protocol_sha256':sha(args.protocol),'freeze_sha256':sha(args.freeze),
            'exact_legacy_parameter_bytes':same_keys and not different,'parameter_count':len(current),'different_parameters':different,
            'max_absolute_parameter_difference':maxdelta,'initial_state_match':initial_match,'pair_order_match':order_match,
            'training_pair_identity_match':pairs_match,'training_text_embeddings_match':text_match,'split_checks':split_checks,
            'max_absolute_score_difference':score_differences,
            'new_artifacts_sha256':new_artifacts,'reference_sha256':{k:sha(v) for k,v in paths.items()}}
    write_json(ROOT/'validation/anchor_verification.json',report)
    if not passed:raise ValueError('Anchor failed exact replay; report retained. No further heads may run.')
    return report


def require_anchor(args,plan,torch):
    report=verify_anchor(args,plan,torch)
    if report['status'] != 'pass':raise ValueError('No complete successful anchor')


def audit(args,plan,torch):
    check_frozen(args,plan);require_anchor(args,plan,torch)
    expected={'development':918,'internal_check':1653};checked=[]
    for seed in SEEDS:
        initial=set();orders=set()
        for variant in VARIANTS:
            folder,report=verify_run(args,seed,variant)
            initial.add(report['legacy_initial_parameter_sha256']);orders.add(tuple(report['expected_pair_order_sha256']))
            if set(report['evaluations']) != set(expected):raise ValueError('Incomplete evaluations')
            for label,n in expected.items():
                result=json.loads((folder/(label+'.json')).read_text())
                matrix=torch.load(folder/(label+'_scores.pt'),map_location='cpu',weights_only=True)
                for name,digest in report['evaluations'][label]['artifacts_sha256'].items():
                    if sha(folder/name) != digest:raise ValueError('Evaluation artifact changed')
                if result['checkpoint_sha256'] != sha(folder/'head.pt') or any(result[k] != v for k,v in identity(args,seed,variant).items()):
                    raise ValueError('Evaluation checkpoint/source identity differs')
                for key in ['query_ids','query_caption_sha256','gallery_ids','gold_gallery_index']:
                    if matrix[key] != result[key]:raise ValueError('Matrix identity differs')
                scores=matrix['scores']
                if tuple(scores.shape)!=(n,n) or not bool(torch.isfinite(scores).all()):raise ValueError('Invalid score matrix')
                order=torch.argsort(scores.cuda(),dim=1,descending=True)
                gold=torch.tensor(result['gold_gallery_index'],device='cuda')[:,None]
                ranks=(order==gold).nonzero(as_tuple=True)[1].add(1).cpu().tolist()
                if ranks != result['ranks'] or legacy.rank_metrics(ranks) != result['metrics_percent']:raise ValueError('Rank/metric replay failed')
                checked.append({'seed':seed,'variant':variant,'split':label,'queries':n})
                del scores,matrix,order,gold
        if len(initial)!=1 or len(orders)!=1:raise ValueError('Within-seed initial parameters/order not matched')
    report={'status':'pass','heads':15,'evaluations':checked,'no_selection':True,'official_test_accessed':False,
            'protocol_sha256':sha(args.protocol),'freeze_sha256':sha(args.freeze),'completed_utc':now()}
    write_json(ROOT/'validation/completed_verification.json',report)
    print(json.dumps({'audit_passed':True,'heads':15,'evaluations':len(checked)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','freeze','anchor','train-all','evaluate','audit'])
    parser.add_argument('--protocol',type=Path,default=ROOT/'protocol.json')
    parser.add_argument('--freeze',type=Path,default=ROOT/'SOURCE_FROZEN.json')
    parser.add_argument('--split-dir',type=Path,default=ROOT/'plans/splits')
    parser.add_argument('--original-split-dir',type=Path,default=P/'data/ret_clean_20261007')
    parser.add_argument('--cache',type=Path,default=P/'formal_experiments_20261103/results/vitb32/cache_physics_train.pt')
    parser.add_argument('--captions',type=Path,default=P/'data/ret_clean_20261007/Caption_train.csv')
    parser.add_argument('--models',type=Path,default=P/'models')
    parser.add_argument('--legacy-root',type=Path,default=P/'repair_experiments_20261103_v1')
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--retry-incomplete',action='store_true')
    parser.add_argument('--acknowledge-reviewed-source',action='store_true')
    args=parser.parse_args();plan=json.loads(args.protocol.read_text());validate_plan(plan)
    (ROOT/'validation').mkdir(exist_ok=True)
    lock=(ROOT/'.execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.command=='freeze':
        if not args.acknowledge_reviewed_source or args.freeze.exists() or args.out.exists():raise ValueError('Freeze requires reviewed source and no prior results')
        bound=json.loads((ROOT/'INPUTS_BOUND.json').read_text())
        if bound['protocol_sha256']!=sha(args.protocol) or bound['source_sha256']!=source_files():raise ValueError('Run preflight again after source/protocol changes')
        exclusive_json(args.freeze,{'schema':'pare.token-mechanism-source-freeze.v1','frozen_utc':now(),
                       'protocol_sha256':sha(args.protocol),'protocol_markdown_sha256':sha(ROOT/'PROTOCOL.md'),
                       'source_sha256':source_files(),'inputs_bound_sha256':sha(ROOT/'INPUTS_BOUND.json'),
                       'means_sha256':sha(ROOT/'training_descriptor_means.json')})
        print(json.dumps({'source_frozen':str(args.freeze),'sha256':sha(args.freeze)}));return
    torch=legacy.setup_runtime(args)
    # The original setup prepends base/code; the frozen reference folder remains
    # on sys.path and supplies the unchanged RepairHead and training kernels.
    if not torch.cuda.is_available():raise ValueError('Execution stages require CUDA')
    if args.command=='preflight':preflight(args,plan,torch);return
    cache,signature,populations,means,frozen,bound=load_bound(args,plan)
    if args.command=='audit':audit(args,plan,torch);return
    pairs,evaldata=shared_data(cache,populations,args)
    if args.command in ['train-all','evaluate']:require_anchor(args,plan,torch)
    if args.command=='evaluate':
        for seed in SEEDS:
            for variant in VARIANTS:verify_run(args,seed,variant)
    random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    dev_vectors,dev_sha=legacy.embed(plan,pairs,evaldata['development'][0],torch)
    if args.command in ['anchor','train-all']:
        seeds=[0] if args.command=='anchor' else SEEDS
        variants=['conditional'] if args.command=='anchor' else VARIANTS
        for seed in seeds:
            for variant in variants:
                train_one_variant(args,plan,torch,cache,signature,populations,means,pairs,dev_vectors,dev_sha,seed,variant)
    if args.command in ['anchor','evaluate']:
        internal_vectors,internal_sha=legacy.embed(plan,[],evaldata['internal_check'][0],torch)
        embeddings={'development':(dev_vectors,dev_sha),'internal_check':(internal_vectors,internal_sha)}
        for seed in ([0] if args.command=='anchor' else SEEDS):
            for variant in (['conditional'] if args.command=='anchor' else VARIANTS):
                evaluate_variant(args,plan,torch,cache,means,evaldata,embeddings,seed,variant)
        check_frozen(args,plan)
        if args.command=='anchor':verify_anchor(args,plan,torch)
    if args.command=='train-all':
        write_json(ROOT/'validation/training_complete.json',{'status':'complete','heads':15,'protocol_sha256':sha(args.protocol),
                   'freeze_sha256':sha(args.freeze),'completed_utc':now(),
                   'head_sha256':{f'{s}/{v}':sha(verify_run(args,s,v)[0]/'head.pt') for s in SEEDS for v in VARIANTS}})


if __name__=='__main__':main()
