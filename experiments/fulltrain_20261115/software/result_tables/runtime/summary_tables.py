"""Paired seed summaries; five backbones are averaged before seed statistics."""
import math
import statistics
from independent_core import require

BONES=('vitb32','vitb16','remoteclip','siglip','georsclip')
METRICS=('R@1','R@5','R@10','MRR@10')
GROUPS={'primary_n4':[1,2,3,4],'secondary_n5':[0,1,2,3,4],'frozen':[None]}
KEYS=('backbone','model_id','epoch','factor','query_mode','metric','seed_group','stat','subgroup',
      'comparator_model_id','comparator_epoch','comparator_factor')


def key(row):return tuple(row.get(k) for k in KEYS)


def observation_index(observations):
    rows={}
    for row in observations:
        k=tuple(row[x] for x in ('backbone','model_id','epoch','factor','seed'))
        require(k not in rows,'Duplicate audited observation');rows[k]=row
    return rows


def aggregate(index,request):
    bones=BONES if request['backbone']=='macro_equal_five_backbones' else [request['backbone']]
    seeds=GROUPS[request['seed_group']]
    delta=request['stat'].startswith('paired_')
    def value(b,model,epoch,factor,seed):
        if model=='frozen_scene':seed=None;epoch=None
        row=index[(b,model,epoch,factor,seed)]
        subgroup=request.get('subgroup')
        if subgroup is not None:
            require(request['query_mode']=='first-caption','GSD uses first captions only')
            metrics=row['gsd'][subgroup]['metrics']
        else:metrics=row['query_modes'][request['query_mode']]
        result=metrics[request['metric']]
        require(math.isfinite(result),'Missing/nonfinite audited metric')
        return result
    values=[]
    for seed in seeds:
        per_bone=[]
        for b in bones:
            x=value(b,request['model_id'],request['epoch'],request['factor'],seed)
            if delta:x-=value(b,request['comparator_model_id'],request['comparator_epoch'],request['comparator_factor'],seed)
            per_bone.append(x)
        values.append(math.fsum(per_bone)/len(per_bone))
    stat=request['stat']
    if stat in ('mean','paired_mean_delta'):number=math.fsum(values)/len(values)
    elif stat in ('sample_sd','paired_sample_sd'):number=statistics.stdev(values) if len(values)>1 else None
    else:raise ValueError('Unknown statistic '+stat)
    return {**{k:request.get(k) for k in KEYS},'value':number,'n_seeds':len(seeds),
            'seeds':seeds,'seed_values':values,'n_backbones_per_seed':len(bones),
            'units':'percentage_points' if delta else 'percent',
            'statistical_unit':'paired seed after equal-backbone averaging' if len(bones)>1 else 'seed'}


def build(observations,registry):
    index=observation_index(observations);requests={}
    def add(request):requests[key(request)]=request
    for row in observations:
        seedgroups=['frozen'] if row['model_id']=='frozen_scene' else ['primary_n4','secondary_n5']
        contexts=[(mode,None) for mode in ('first-caption','all-captions')]
        contexts += [('first-caption',name) for name in row['gsd']]
        for mode,subgroup in contexts:
            for metric in METRICS:
                for sg in seedgroups:
                    for stat in (['mean'] if sg=='frozen' else ['mean','sample_sd']):
                        add({k:row[k] for k in ('backbone','model_id','epoch','factor')} |
                            dict(query_mode=mode,subgroup=subgroup,metric=metric,seed_group=sg,stat=stat))
    for slot in registry['slots']:
        if slot['kind']!='measurement':continue
        request={k:slot.get(k) for k in KEYS};add(request)
        if request['seed_group']=='primary_n4':add({**request,'seed_group':'secondary_n5'})
        if request['stat']=='paired_mean_delta':
            add({**request,'stat':'paired_sample_sd'})
            if request['seed_group']=='primary_n4':add({**request,'seed_group':'secondary_n5','stat':'paired_sample_sd'})
    # Prespecified core contrasts are reported even if the manuscript currently
    # has no slot for them. Never take absolute values or drop negative seeds.
    for request in list(requests.values()):
        if request['stat']!='mean' or request['seed_group']=='frozen':continue
        comparisons=[]
        if request['model_id']=='path_100__full' and request['factor']=='1':
            for comparator in ('frozen_scene','adapter','original_full','path_000__full','joint_full',
                               'path_100__constant','path_100__no_token','path_100__uniform','path_100__stats_off'):
                comparisons.append((comparator,None if comparator=='frozen_scene' else request['epoch'],'1'))
        if request['epoch']==50 and request['factor']=='1':
            comparisons.append((request['model_id'],20,'1'))
        if request['factor']!='1':
            comparisons.append((request['model_id'],request['epoch'],'1'))
        for model,epoch,factor in comparisons:
            paired={**request,'stat':'paired_mean_delta','comparator_model_id':model,
                    'comparator_epoch':epoch,'comparator_factor':factor}
            try:aggregate(index,paired)
            except KeyError:continue  # For example, B32-only controls have no other-backbone result.
            add(paired);add({**paired,'stat':'paired_sample_sd'})
    # Every eligible model/contrast macro uses exactly five backbones, not a
    # pooled sample of 5*seed runs. B32-only diagnostics/scales have no such macro.
    measurements=[aggregate(index,r) for r in requests.values()]
    macro_requests={}
    for request in requests.values():
        if request['factor']!='1':continue
        macro={**request,'backbone':'macro_equal_five_backbones'}
        try:measurement=aggregate(index,macro)
        except KeyError:continue
        macro_requests[key(macro)]=measurement
    existing={key(r) for r in measurements}
    measurements.extend(value for k,value in macro_requests.items() if k not in existing)
    return {'schema':'pare.formal-score-summary.v1','status':'derived_from_independent_score_audit',
            'seed_groups':GROUPS,'measurements':measurements,
            'observations':observations,
            'limits':['SD is across seeds, not a confidence interval.',
                      'Five-backbone macro averages backbones within each seed before mean/SD.',
                      'All signed paired differences are retained; no p-values or significance claims.',
                      'Official benchmark was historically reused; this does not make it a new untouched holdout.']}
