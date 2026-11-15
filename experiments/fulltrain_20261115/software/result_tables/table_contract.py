"""Public saved-rank contracts. Standard library only; no models or scores."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'runtime'))
from independent_core import require, objsha, metrics
from summary_tables import build, key, GROUPS

BONES = ('vitb32', 'vitb16', 'remoteclip', 'siglip', 'georsclip')
MAIN = ('adapter', 'original_full', 'path_000__full', 'joint_full', 'path_100__full')
CONTROLS = ('path_100__constant', 'path_100__no_token', 'path_100__uniform', 'path_100__stats_off')
IDENTITY = ('cohort', 'backbone', 'model_id', 'seed', 'epoch', 'factor')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def matrix():
    rows = {('main', *r, '1') for r in itertools.product(BONES, MAIN, range(5), (20, 50))}
    rows |= {('b32_diagnostics', 'vitb32', *r, '1') for r in itertools.product(CONTROLS, range(5), (20, 50))}
    rows |= {('main', b, 'frozen_scene', None, None, '1') for b in BONES}
    rows |= {('main', 'vitb32', *r) for r in itertools.product(('adapter', 'path_100__full'), range(5), (50,), ('1.5', '2'))}
    return rows


def validate_matrix(observations):
    keys = [tuple(row[k] for k in IDENTITY) for row in observations]
    require(len(keys) == len(set(keys)) == 315 and set(keys) == matrix(), 'Incomplete/unknown 315-evaluation matrix')
    jobids = [row['job_id'] for row in observations]
    require(len(jobids) == len(set(jobids)) and all(isinstance(j,str) and re.fullmatch('[A-Za-z0-9_.-]+',j) for j in jobids),
            'Unsafe or duplicate job ID')
    for row in observations:
        if row['model_id'] == 'frozen_scene':
            require(row['seed'] is None and row['epoch'] is None, 'Frozen baseline must not have synthetic seed/budget')
        else:
            require(type(row['seed']) is int and type(row['epoch']) is int, 'Noninteger seed/budget')


def same(left, right, where):
    """Same frozen arithmetic must reproduce each finite numeric value exactly."""
    if isinstance(left, dict):
        require(isinstance(right,dict) and set(left) == set(right), 'Keys differ at '+where)
        for k,v in left.items():
            same(v,right[k],where+'/'+str(k))
    elif isinstance(left, list):
        require(isinstance(right,list) and len(left) == len(right), 'List differs at '+where)
        for i,(a,b) in enumerate(zip(left,right)):
            same(a,b,where+'/'+str(i))
    elif type(left) in (int,float):
        require(type(right) in (int,float) and math.isfinite(left) and math.isfinite(right) and left == right,
                'Numeric value differs at '+where)
    else:
        require(type(left) is type(right) and left == right, 'Value differs at '+where)


def query_contract(identities, grouping, fixed=None):
    require(identities['schema'] == 'pare.public-query-identities.v1'
            and identities['raw_caption_text_included'] is False, 'Unrecognized public query identities')
    gallery = identities['gallery_ids']; rows = identities['queries']
    require(len(gallery) == len(set(gallery)) and all(isinstance(i,str) for i in gallery), 'Invalid gallery IDs')
    require(gallery == grouping['test_ids'] and objsha(gallery) == grouping['test_ids_sha256'], 'Query/grouping gallery differs')
    require(grouping['n_gallery_retained_for_all_subgroups'] == len(gallery), 'GSD must retain the full gallery')
    first = []; seen_images = set(); seen_queries = set(); previous = None; caption_number = 0
    for index,row in enumerate(rows):
        require(set(row) == {'query_index','query_id','gold_gallery_index','caption_sha256'}, 'Unexpected query field/full text')
        gold = row['gold_gallery_index']
        require(type(gold) is int and 0 <= gold < len(gallery) and row['query_index'] == index, 'Query index/gold differs')
        image = gallery[gold]
        if image != previous:
            require(image not in seen_images, 'Caption rows must be grouped in first-image encounter order')
            first.append(index); seen_images.add(image); caption_number = 1; previous = image
        else:
            caption_number += 1
        require(row['query_id'] == f'{image}#caption={caption_number}' and row['query_id'] not in seen_queries,
                'Query ID/order mismatch')
        seen_queries.add(row['query_id'])
        require(isinstance(row['caption_sha256'],str) and re.fullmatch('[0-9a-f]{64}',row['caption_sha256']), 'Invalid caption hash')
    require(len(first) == len(gallery) and first == identities['first_caption_all_query_indices'], 'First-caption coverage/order differs')
    groups = grouping['groups']; flattened = [i for values in groups.values() for i in values]
    require(len(flattened) == len(set(flattened)) and set(flattened) == set(gallery), 'GSD groups must partition first captions')
    require({k:len(v) for k,v in groups.items()} == grouping['counts'], 'GSD subgroup counts differ')
    if fixed is not None:
        require((len(gallery),len(rows),len(first)) == (3688,6190,3688), 'Full public benchmark required')
        require(grouping['counts'] == {'lt_0.30':760,'0.30_to_lt_0.45':238,'ge_0.45':188,'unavailable':2502}, 'Fixed GSD groups differ')
        first_set = set(first); extra = [r for i,r in enumerate(rows) if i not in first_set]
        streams = {'first':[rows[i] for i in first], 'extra':extra, 'all':rows}
        digests = fixed['digests']
        for name,records in streams.items():
            require(objsha([r['caption_sha256'] for r in records]) == digests[name+'_text_sha256'], 'Fixed caption identity differs: '+name)
            require(objsha([r['query_id'] for r in records]) == digests[name+'_query_ids_sha256'], 'Fixed query identity differs: '+name)
    return gallery, rows, first


def from_ranks(rank, gallery, queries, first, grouping):
    require(set(rank) == {'job_id','ranks','first_ranks','exact_gold_tie_sizes'}, 'Unexpected saved-rank fields')
    require(len(rank['ranks']) == len(queries), 'Rank query count differs')
    full = metrics(rank['ranks'],len(gallery))
    first_ranks = [rank['ranks'][i] for i in first]
    require(rank['first_ranks'] == first_ranks, 'Stored first-caption ranks differ from identity mapping')
    ties = rank['exact_gold_tie_sizes']
    require(len(ties) == len(queries) and all(type(t) is int and 1 <= t <= len(gallery) for t in ties), 'Invalid stored tie counts')
    by_image = {gallery[queries[i]['gold_gallery_index']]:rank['ranks'][i] for i in first}
    return {'query_modes':{'first-caption':metrics(first_ranks,len(gallery)), 'all-captions':full},
            'gsd':{name:{'n_queries':len(images),'metrics':metrics([by_image[i] for i in images],len(gallery))}
                   for name,images in grouping['groups'].items()}}


def compare_measurements(recomputed, published):
    expected = {key(r):r for r in published}; actual = {key(r):r for r in recomputed}
    require(len(expected) == len(published) and len(actual) == len(recomputed), 'Duplicate summary measurement')
    require(set(actual) == set(expected), 'Complete measurement inventory differs')
    for k,row in actual.items():
        same(row,expected[k],'measurement/'+repr(k))


def recompute(summary, identities, grouping, registry, rank_loader, fixed=None):
    validate_matrix(summary['observations'])
    require(summary['schema'] == 'pare.formal-score-summary.v1'
            and summary['status'] == 'derived_from_independent_score_audit', 'Not a published formal summary')
    same(summary['seed_groups'],GROUPS,'seed_groups')
    require(registry['schema'] == 'pare.manuscript-result-slots.v1', 'Wrong slot registry')
    kinds = [s['kind'] for s in registry['slots']]
    require(len(kinds) == 333 and kinds.count('measurement') == 327 and kinds.count('manual_text') == 6, 'Fixed 327+6 slots required')
    require(all(s.get('value') is None for s in registry['slots']), 'Prefilled result slot')
    gallery, queries, first = query_contract(identities,grouping,fixed)
    observations = []
    for published in summary['observations']:
        rank = rank_loader(published)
        require(rank['job_id'] == published['job_id'], 'Rank job differs')
        calculated = from_ranks(rank,gallery,queries,first,grouping)
        for field in ('query_modes','gsd'):
            same(calculated[field],published[field],published['job_id']+'/'+field)
        row = {k:copy.deepcopy(v) for k,v in published.items() if k not in ('query_modes','gsd')}
        row.update(calculated); observations.append(row)
    result = build(observations,registry)
    compare_measurements(result['measurements'],summary['measurements'])
    return result


def compare_original(public, original):
    validate_matrix(original['observations'])
    expected = {r['job_id']:r for r in original['observations']}
    require(set(expected) == {r['job_id'] for r in public['observations']}, 'Original/public job IDs differ')
    for row in public['observations']:
        source = expected[row['job_id']]
        same({k:row[k] for k in (*IDENTITY,'score_sha256','query_modes','gsd')},
             {k:source[k] for k in (*IDENTITY,'score_sha256','query_modes','gsd')},'original/'+row['job_id'])
        if row['model_id'] == 'frozen_scene':
            require(row['checkpoint'] is None and source['checkpoint'] is None, 'Frozen checkpoint differs')
        else:
            require(row['source_checkpoint_sha256'] == source['checkpoint']['sha256'], 'Original checkpoint identity differs')
    same(public['seed_groups'],original['seed_groups'],'original/seed_groups')
    compare_measurements(public['measurements'],original['measurements'])
