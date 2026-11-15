"""Independent CPU audit of saved final scores. No models/checkpoints loaded."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
from independent_core import require,objsha,identities,summarize,compare_metrics
from summary_tables import build

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


class Reader:
    def __init__(self):self.files={}
    def bind(self,row):
        require(isinstance(row,dict) and isinstance(row.get('path'),str) and Path(row['path']).is_absolute()
                and isinstance(row.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Unbound artifact')
        path=Path(row['path']).resolve()
        if str(path) not in self.files:require(sha(path)==row['sha256'],'Artifact hash differs: '+str(path))
        else:require(self.files[str(path)]==row['sha256'],'Conflicting artifact binding')
        self.files[str(path)]=row['sha256'];return path
    def json(self,row):return json.loads(self.bind(row).read_text())
    def finish(self):
        for path,digest in self.files.items():require(sha(path)==digest,'Artifact changed during audit: '+path)


def write(path,data):
    with path.open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')


def validate_plan(plan):
    require(plan.get('schema')=='pare.formal-score-audit.plan.v1','Unknown audit plan')
    require(plan.get('status')=='FROZEN_FOR_FORMAL_SCORE_AUDIT','UNBOUND_DRAFT: no score files or Torch loaded')
    require(plan.get('ranking_rule')=='one_plus_greater_plus_exact_ties_at_earlier_gallery_rows','Unknown rank rule')
    for key in ('score_schedule','schedule_completion','source_lock','result_slots'):
        require(isinstance(plan.get(key),dict),'Unbound '+key)


def prepare(plan,reader):
    lock=reader.json(plan['source_lock'])
    require(lock['status']=='FROZEN_FORMAL_SCORE_AUDIT_SOURCE','Audit source not frozen')
    for name,digest in lock['files'].items():
        path=(ROOT/name).resolve();require(ROOT in path.parents and not Path(name).is_absolute(),'Unsafe audit source')
        reader.bind({'path':str(path),'sha256':digest})
    require({'audit_scores.py','independent_core.py','summary_tables.py'}<=set(lock['files']),'Audit source coverage missing')
    schedule=reader.json(plan['score_schedule'])
    require(schedule['schema']=='pare.final-score-schedule.v1' and schedule['status']=='FROZEN_FOR_FINAL_SCORE_SCHEDULE','Score schedule not frozen')
    completed=reader.json(plan['schedule_completion'])
    require(completed['status']=='completed_pending_result_review' and completed['failed']==[]
            and completed['not_launched']==[] and completed['downloaded_scores'] is False,'Score schedule incomplete')
    receipts={row['entry']['id']:row for row in completed['completed']}
    require(len(receipts)==len(completed['completed'])==315,'Incomplete completion inventory')
    slots=reader.json(plan['result_slots'])
    require(slots['schema']=='pare.manuscript-result-slots.v1','Unknown result slot registry')
    jobs={};proofs={};common=None
    for c in schedule['cohorts']:
        name=c['id'];require(name in ('main','b32_diagnostics') and name not in proofs,'Duplicate/unknown cohort')
        train=reader.json(c['training_protocol']);ev=reader.json(c['evaluation_protocol'])
        require(train['status']=='frozen_for_final_run' and ev['status']=='FROZEN_FOR_FINAL_EVALUATION','Unfrozen input protocol')
        require(slots['protocol_sha256'][name]==c['training_protocol']['sha256'],'Manuscript slots bind another training protocol')
        require(train['seeds']==[0,1,2,3,4] and train['checkpoint_epochs']==[20,50],'Seed/checkpoint coverage differs')
        require((len(train['backbones']),len(train['models']))==((5,5) if name=='main' else (1,4)),'Training matrix dimensions differ')
        expected={(b,m,s,e,'1') for b in train['backbones'] for m in train['models'] for s in train['seeds'] for e in (20,50)}
        if name=='main':
            expected|={(b,'frozen_scene',None,None,'1') for b in train['backbones']}
            expected|={('vitb32',m,s,50,f) for m in ('adapter','path_100__full') for s in train['seeds'] for f in ('1.5','2')}
        observed=[tuple(j[k] for k in ('backbone','variant','seed','epoch','factor')) for j in ev['jobs']]
        require(len(observed)==len(set(observed)) and set(observed)==expected,'Incomplete formal score matrix')
        audit=reader.json(ev['training_audit'])
        require(audit['schema']=='pare.final-full-training-audit.v1' and audit['status']=='audit_passed'
                and audit['errors']==[] and audit['training_protocol']['sha256']==c['training_protocol']['sha256'],
                'Wrong/incomplete training audit')
        require(Path(audit['training_protocol']['path']).resolve()==(Path(c['training_root'])/'PROTOCOL.json').resolve(),'Training audit root differs')
        expected_checkpoints={key[:-1] for key in expected if key[-1]=='1' and key[1]!='frozen_scene'}
        cps={tuple(x[k] for k in ('backbone','variant','seed','epoch')):x for x in audit['checkpoints']}
        require(len(cps)==len(audit['checkpoints']) and set(cps)==expected_checkpoints,'Training audit checkpoint matrix incomplete')
        eval_auth=reader.json(c['evaluation_authorization'])
        require(eval_auth['status']=='AUTHORIZED_FINAL_EVALUATION'
                and eval_auth['protocol_sha256']==c['evaluation_protocol']['sha256']
                and set(eval_auth['allowed_job_ids'])=={j['id'] for j in ev['jobs']},'Evaluation authorization differs')
        source=reader.json(ev['source_lock']);evalroot=Path(schedule['evaluator']['path']).resolve().parent
        require(source['status']=='FROZEN_EVALUATION_SOURCE','Evaluator source not frozen')
        for file,digest in source['files'].items():
            path=(evalroot/file).resolve();require(evalroot in path.parents and not Path(file).is_absolute(),'Unsafe evaluator source')
            reader.bind({'path':str(path),'sha256':digest})
        catalog=reader.json(ev['input_catalog']);fresh=reader.json(ev['fresh_extra_verification']);grouping=reader.json(ev['gsd_groups'])
        require(fresh['status']=='input_verification_passed' and fresh['errors']==[],'Fresh text proof not passed')
        if name=='main':
            scale=reader.json(ev['scale_input_catalog'])
            require(scale['status']=='BOUND_FOR_FINAL_SCALE_INPUTS' and {'1.5','2'}<=set(scale['scales']),
                    'Scale input catalog incomplete')
        shared=tuple(ev[k]['sha256'] for k in ('source_lock','input_catalog','fresh_extra_verification','gsd_groups'))
        if common is None:common=(shared,catalog,grouping)
        else:require(common[0]==shared,'Cohort inputs differ')
        for j in ev['jobs']:
            jid=j['id'];require(re.fullmatch('[a-zA-Z0-9_.-]+',jid) and jid not in jobs,'Invalid duplicate job ID')
            if j['variant']=='frozen_scene':require(j['checkpoint'] is None,'Frozen baseline checkpoint')
            else:
                cp=cps[(j['backbone'],j['variant'],j['seed'],j['epoch'])]
                require(cp['checkpoint']==j['checkpoint'],'Score uses unaudited checkpoint')
                reader.bind(j['checkpoint'])  # Stream SHA only; never load checkpoint tensors.
                expected_path=Path(c['training_root'])/'runs'/j['backbone']/f"seed_{j['seed']}"/j['variant']/f"epoch_{j['epoch']}.pt"
                require(Path(j['checkpoint']['path']).resolve()==expected_path.resolve(),'Checkpoint path differs')
            if j['factor']!='1':
                original=[o for o in ev['jobs'] if o['factor']=='1' and all(o[k]==j[k] for k in ('backbone','variant','seed','epoch','checkpoint'))]
                require(len(original)==1,'Scale and original checkpoint differ')
            jobs[jid]=(c,ev,j,cps)
        proofs[name]=audit
    require(set(proofs)=={'main','b32_diagnostics'} and set(jobs)==set(receipts),'Score completion job coverage differs')
    require({(x['cohort'],x['id']) for x in schedule['jobs']}=={(c['id'],jid) for jid,(c,_,_,_) in jobs.items()}
            and len(schedule['jobs'])==315,'Schedule queue differs')
    output=Path(schedule['output_root']).resolve()
    require(Path(plan['schedule_completion']['path']).resolve()==output/'SCORE_SCHEDULE_RESULT.json','Schedule completion root differs')
    for name in proofs:
        require({p.name for p in (output/name).iterdir() if p.is_dir()}=={jid for jid,(c,_,_,_) in jobs.items() if c['id']==name},
                'Missing/extra job directories')
    _,catalog,grouping=common
    gallery=reader.json(catalog['common']['test_ids']);captions=reader.bind(catalog['common']['test_captions'])
    records,first=identities(gallery,captions)
    require(grouping['test_ids']==gallery and grouping['test_ids_sha256']==objsha(gallery),'GSD/gallery identity mismatch')
    return schedule,receipts,jobs,slots,gallery,records,first,grouping


def audit(plan,reader,out):
    schedule,receipts,jobs,slots,gallery,records,first,grouping=prepare(plan,reader)
    # Saved prediction tensors only. No model code or checkpoint loader imported.
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
    import torch
    torch.set_num_threads(3)
    def deny_cuda(*args,**kwargs):raise RuntimeError('CUDA forbidden in independent score audit')
    torch.cuda._lazy_init=deny_cuda
    out.mkdir(parents=True,exist_ok=False);(out/'ranks').mkdir()
    observations=[];rank_files=[];gold=[r['gold'] for r in records];query_ids=[r['query_id'] for r in records]
    for jid,(cohort,ev,job,cps) in jobs.items():
        receipt=receipts[jid];folder=Path(schedule['output_root'])/cohort['id']/jid
        require(receipt['entry']=={'id':jid,'cohort':cohort['id']} and receipt['returncode']==0,'Completion receipt differs')
        require(Path(receipt['completion']['path']).resolve()==(folder/'complete.json').resolve(),'Job completion path differs')
        complete=reader.json(receipt['completion'])
        require(complete['status']=='completed' and complete['job']==job
                and complete['protocol_sha256']==cohort['evaluation_protocol']['sha256'],'Completion identity differs')
        require(set(complete['artifacts'])=={'scores.pt','RESULT.json'} and {p.name for p in folder.iterdir()}=={'scores.pt','RESULT.json','complete.json'},
                'Missing/extra job artifacts')
        result=reader.json({'path':str(folder/'RESULT.json'),'sha256':complete['artifacts']['RESULT.json']})
        score_path=reader.bind({'path':str(folder/'scores.pt'),'sha256':complete['artifacts']['scores.pt']})
        require(result['job']==job and result['protocol_sha256']==cohort['evaluation_protocol']['sha256']
                and result['authorization_sha256']==cohort['evaluation_authorization']['sha256']
                and result['source_lock_sha256']==ev['source_lock']['sha256']
                and result['score_sha256']==complete['artifacts']['scores.pt'],'Result provenance differs')
        require(result['gallery_size']==3688 and result['official_population'] is True
                and result['tie_rule']=='descending_float32_score_then_ascending_gallery_row; CPU stable argsort',
                'Reported gallery/rank protocol differs')
        if job['variant']!='frozen_scene':require(result['query_batch_size']==8,'Query chunk size differs')
        mandatory=[ev[k] for k in ('source_lock','input_catalog','training_audit','fresh_extra_verification','gsd_groups')]
        mandatory.extend([cohort['evaluation_protocol'],cohort['evaluation_authorization']])
        if job['checkpoint']:mandatory.append(job['checkpoint'])
        if job['factor']!='1':mandatory.append(ev['scale_input_catalog'])
        require(all(result['source_and_input_sha256'].get(str(Path(v['path']).resolve()))==v['sha256'] for v in mandatory),'Result source/input bindings missing')
        for path,digest in result['source_and_input_sha256'].items():
            reader.bind({'path':path,'sha256':digest})
        require(result['input_receipt']['current_extra_reencoded_and_compared'] is True
                and result['input_receipt']['final_scoring_ready'] is True
                and result['input_receipt']['accepted_fresh_extra_verification']==ev['fresh_extra_verification'],
                'Input validation receipt incomplete')
        if job['variant']=='frozen_scene':require(result['checkpoint_metadata'] is None,'Baseline metadata borrows trained checkpoint')
        else:
            proof=cps[(job['backbone'],job['variant'],job['seed'],job['epoch'])]
            metadata=reader.json(proof['run_json'])
            require(result['checkpoint_metadata']==metadata,'Result checkpoint metadata differs from audited training run')
        saved=torch.load(score_path,map_location='cpu',weights_only=True)
        require(saved['job']==job and saved['protocol_sha256']==cohort['evaluation_protocol']['sha256']
                and saved['gallery_ids']==gallery and saved['query_ids']==query_ids and saved['gold']==gold,'Saved score identities differ')
        scores=saved['scores'];require(tuple(scores.shape)==(6190,3688),'Saved score shape differs')
        independent=summarize(scores,gold,records,first,grouping)
        for mode in ('first-caption','all-captions'):
            ranks=independent['first_ranks'] if mode=='first-caption' else independent['ranks']
            ids=[query_ids[i] for i in first] if mode=='first-caption' else query_ids
            require(result[mode]['ranks']==ranks and result[mode]['query_ids']==ids and result[mode]['n_queries']==len(ranks),
                    'Independent ranks/query population disagree')
            compare_metrics(result[mode]['metrics_percent'],independent['query_modes'][mode])
        require(result['first-caption']['source_all_caption_rows']==first and result['all-captions']['gold_score_tie_group_sizes']==independent['ties'],
                'First/all mapping or exact tie counts differ')
        require(result['gsd_subgroups']['gallery_size']==3688 and result['gsd_subgroups']['groups_sha256']==objsha(grouping),'GSD binding differs')
        require(set(result['gsd_subgroups']['subgroups'])==set(independent['gsd']),'GSD subgroup set differs')
        for name,g in independent['gsd'].items():
            reported=result['gsd_subgroups']['subgroups'][name]
            require(reported['n_queries']==g['n_queries'],'GSD count differs')
            compare_metrics(reported['metrics']['evaluated_model'],g['metrics'])
        rankpath=out/'ranks'/(jid+'.json')
        write(rankpath,{'job_id':jid,'ranks':independent['ranks'],'first_ranks':independent['first_ranks'],'exact_gold_tie_sizes':independent['ties']})
        rank_files.append({'path':str(rankpath.resolve()),'sha256':sha(rankpath)})
        observations.append({'job_id':jid,'cohort':cohort['id'],'backbone':job['backbone'],'model_id':job['variant'],
            'seed':job['seed'],'epoch':job['epoch'],'factor':job['factor'],'checkpoint':job['checkpoint'],
            'score_sha256':complete['artifacts']['scores.pt'],'query_modes':independent['query_modes'],'gsd':independent['gsd']})
        del saved,scores,independent
    summary=build(observations,slots)
    require(not torch.cuda.is_initialized(),'CUDA initialized during score audit')
    reader.finish()
    summary.update(result_slots_sha256=plan['result_slots']['sha256'],score_schedule_sha256=plan['score_schedule']['sha256'])
    write(out/'SUMMARY.json',summary)
    report={'schema':'pare.formal-score-audit.v1','status':'audit_passed','errors':[],
            'jobs':315,'unique_trained_checkpoints':290,'summary':{'path':str((out/'SUMMARY.json').resolve()),'sha256':sha(out/'SUMMARY.json')},
            'result_slots':plan['result_slots'],'score_schedule':plan['score_schedule'],'schedule_completion':plan['schedule_completion'],
            'source_lock':plan['source_lock'],'artifact_sha256':reader.files,'independent_rank_files':rank_files,
            'ranking_rule':plan['ranking_rule'],'models_loaded':False,'checkpoint_tensors_loaded':False,
            'recorded_input_file_bytes_rechecked':True,'cuda_initialized':False,
            'completed_utc':datetime.now(timezone.utc).isoformat()}
    write(out/'AUDIT.json',report)
    write(out/'complete.json',{'status':'completed','artifacts':{n:sha(out/n) for n in ('AUDIT.json','SUMMARY.json')}})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    reader=Reader();plan=reader.json({'path':str(args.plan.resolve()),'sha256':args.plan_sha256})
    validate_plan(plan);require(not args.out.exists(),'Refuse to overwrite audit output')
    report=audit(plan,reader,args.out.resolve())
    print(json.dumps({'status':report['status'],'jobs':report['jobs'],'summary':report['summary']}))


if __name__=='__main__':main()
