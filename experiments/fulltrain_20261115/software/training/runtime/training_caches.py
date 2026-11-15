"""Full-train v2/v9 cache routes. No test input readers or automatic execution."""
from pathlib import Path
from cache_common import BoundReader,DIMENSIONS,objsha,require


def load_training_cache(training,backbone,reader):
    require(training['status']=='BOUND_FOR_FINAL_TRAINING','Training input catalog unbound')
    require(backbone in DIMENSIONS,'Unknown backbone')
    common=training['common'];row=training['backbones'][backbone]
    for name in ('train_ids','fit_ids','val_ids','train_captions'):reader.bind(common[name])
    ids=reader.json(common['train_ids']);fit=reader.json(common['fit_ids']);val=reader.json(common['val_ids'])
    require((len(ids),len(set(ids)),len(fit),len(set(fit)),len(val),len(set(val)))==(9181,9181,8263,8263,918,918),'Training identity counts')
    require(set(fit).isdisjoint(val) and set(fit)|set(val)==set(ids),'Historical fit/val identity partition')
    require(row['D']==DIMENSIONS[backbone],'Backbone dimension')
    import torch
    from v9_train_cache import validate_layout
    if backbone=='vitb32':
        require(row['route']=='B32_v2_paired_legacy_train','B32 route must be v2 paired legacy')
        cache,receipt=load_v2_training(torch,row['v2_inputs'],ids,reader)
    else:
        require(row['route']=='v9_legacy_train','Other backbone route must be v9 legacy')
        cache,receipt=load_v9_training(training,backbone,ids,reader)
    validate_layout(torch,cache,DIMENSIONS[backbone],ids,'full training input')
    # Historical fit/val files validate identity only: no subset is selected.
    receipt.update(training_population='all_9181_images',selected_training_rows=list(range(9181)),
                   historical_fit_val_used_only_for_input_identity=True)
    return cache,receipt


def load_v2_training(torch,bindings,ids,reader):
    from reference.v2_source import checked_pair
    for key in ('paired_audit','parent_cache','cache','numerical_acceptance'):reader.bind(bindings[key])
    audit=reader.json(bindings['paired_audit']);acceptance=reader.json(bindings['numerical_acceptance'])
    require(acceptance['full_audit_sha256']==bindings['paired_audit']['sha256'],'Numerical history applies to another cache audit')
    # This old note documents a numerical finding. It does NOT authorize this
    # final experiment; only the separate future final decision/protocol does.
    require(acceptance['actions']['cache_mutations'] is False and acceptance['actions']['excluded_images']==0,'Input numerical history changed')
    parent=bindings['parent_cache'];require(audit['contract']['parent_cache_sha256']==parent['sha256'],'v2 parent binding')
    for entry in audit['contract']['population_id_files'].values():reader.bind(entry)
    for entry in audit['outputs'].values():
        reader.bind({'path':str((Path(bindings['paired_audit']['path']).resolve().parent/entry['path']).resolve()),'sha256':entry['sha256']})
    cache,receipt,populations=checked_pair(torch,Path(bindings['paired_audit']['path']),bindings['paired_audit']['sha256'],
                                         Path(bindings['cache']['path']),'legacy_raster',Path(parent['path']))
    require(populations['train']==ids and cache['ids']==ids,'v2 final train order')
    require(receipt['caches']['legacy_raster']['sha256']==bindings['cache']['sha256'],'v2 selected cache binding')
    receipt.update(route='B32_v2_paired_legacy_train',numerical_acceptance=bindings['numerical_acceptance'])
    return cache,receipt


def load_v9_training(training,backbone,ids,reader):
    from v9_train_cache import load_checked_cache
    row=training['backbones'][backbone]['v9_inputs']
    for key in ('parent_cache','cache','build_audit'):reader.bind(row[key])
    audit=reader.json(training['v9_accepted_full_input_audit'])
    require(audit['status']=='audit_passed' and audit['errors']==[] and audit['bound_files_unchanged_through_audit'] is True,'Unaccepted v9 inputs')
    require(audit['source_lock_sha256']==training['v9_builder_source_lock_sha256'],'v9 accepted source binding')
    observations=[r for r in audit['observations'] if r['task']==backbone+'_train'];require(len(observations)==1,'v9 train observation missing/duplicate')
    observed=observations[0]
    require((observed['population'],observed['images'],observed['output_path'],observed['output_sha256'],observed['generation_audit_sha256'])==
            ('train',9181,row['cache']['path'],row['cache']['sha256'],row['build_audit']['sha256']),'v9 accepted input binding')
    subcatalog={'common':training['common'],'backbones':{backbone:row},'v9_builder_source_lock_sha256':training['v9_builder_source_lock_sha256']}
    cache,receipt=load_checked_cache(subcatalog,backbone)
    require(cache['ids']==ids,'v9 final train order');receipt.update(route='v9_legacy_train')
    return cache,receipt
