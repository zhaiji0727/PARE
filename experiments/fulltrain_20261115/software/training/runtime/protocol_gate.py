"""Standard-library-only gate. An unbound draft cannot read inputs or initialize GPU."""
import hashlib
import json
import math
from model_spec import VARIANTS
from pathlib import Path

DIMENSIONS={'vitb32':512,'vitb16':512,'remoteclip':512,'siglip':768,'georsclip':512}

def require(condition,message):
    if not condition:raise ValueError(message)

def is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def validate_binding(row,label):
    require(isinstance(row,dict) and isinstance(row.get('path'),str) and Path(row['path']).is_absolute() and is_sha(row.get('sha256')),label+' not bound')

def validate_values(p,stage):
    require(p.get('schema')=='pare.final-evaluation-draft.v1','Unknown schema')
    require(p.get('status')=='frozen_for_final_run','UNBOUND_DRAFT: no input access or GPU work allowed')
    require(stage in ('train','evaluate'),'Unknown stage')
    require(p.get(stage+'_backend_reviewed_and_ready') is True,'Backend implementation/review not complete')
    for key in ('source_lock','decision_record','input_catalog'):validate_binding(p.get(key),key)
    require(p.get('backbones') and set(p['backbones'])<=set(DIMENSIONS),'Backbone matrix unbound')
    require(isinstance(p.get('models'),list) and p['models'] and len(p['models'])==len(set(p['models'])) and set(p['models'])<=set(VARIANTS),'Model matrix unbound')
    require(isinstance(p.get('seeds'),list) and p['seeds'] and all(type(s) is int and s>=0 for s in p['seeds']) and len(p['seeds'])==len(set(p['seeds'])),'Seeds unbound')
    require(type(p.get('epochs')) is int and p['epochs']>0,'Epoch budget unbound')
    points=p.get('checkpoint_epochs')
    require(isinstance(points,list) and points and points==sorted(set(points)) and all(type(x) is int and 1<=x<=p['epochs'] for x in points) and p['epochs'] in points,'Checkpoint schedule unbound')
    require(p.get('batch_size')==64 and p.get('weight_decay')==.0001,'Inherited optimizer recipe changed')
    require(isinstance(p.get('expected_torch_version'),str) and p['expected_torch_version'] and isinstance(p.get('expected_gpu_name'),str) and p['expected_gpu_name'],'Runtime binding absent')
    hp=p.get('hyperparameters');require(isinstance(hp,dict) and set(hp)==set(p['backbones']),'Hyperparameters unbound')
    for tag,row in hp.items():
        require(set(row)=={'learning_rate','kappa'} and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in row.values()),'Hyperparameter values unbound: '+tag)
    require(p.get('training_population')=='all_9181_images_31945_caption_image_pairs' and p.get('descriptor_mean_population')=='all_valid_regions_of_all_9181_training_images','Full training population required')
    require(p.get('initialization')=='fresh_from_seed' and p.get('warm_start') is False,'No checkpoint warm-start')
    require(p.get('automatic_selection') is False,'No automatic selection')
    for b in p['backbones']:
        route=p.get('train_cache_routes',{}).get(b)
        require(route==('B32_v2_paired_legacy_train' if b=='vitb32' else 'v9_legacy_train'),'Wrong training cache schema route')
    ev=p.get('evaluation');require(isinstance(ev,dict),'Evaluation unbound')
    require(ev.get('query_modes')==['first-caption','all-captions'] and ev.get('query_counts')==[3688,6190] and ev.get('gallery_size')==3688 and ev.get('max_queries')==0,'Complete official query protocol required')
    if stage=='evaluate':require(ev.get('tie_policy') in ('legacy_scene_reorder_exact','stable_gallery_order'),'Tie policy must be explicitly chosen and documented')
    require(ev.get('reuse_historical_scores') is False,'New heads require fresh scores')
    if stage=='evaluate':validate_binding(ev.get('query_identity_manifest'),'official query identity')
    if stage=='evaluate':require(type(ev.get('query_batch_size')) is int and ev['query_batch_size']>0 and type(ev.get('gallery_batch_size')) is int and ev['gallery_batch_size']>0,'Scoring chunks unbound')
    require(ev.get('test_previously_seen') is True,'Historical reuse disclosure required')
    if stage=='evaluate':
        validate_binding(ev.get('final_training_audit'),'complete final training audit')
        validate_binding(ev.get('evaluation_authorization'),'explicit final evaluation release')
    return p

def load_protocol(path,expected_sha,stage):
    # The only file touched before rejection is the explicitly supplied protocol itself.
    data=Path(path).read_bytes();require(is_sha(expected_sha) and hashlib.sha256(data).hexdigest()==expected_sha,'Protocol hash differs')
    return validate_values(json.loads(data),stage)
