"""Complete 315-case synthetic rank fixture; no scores, weights, Torch or GPU."""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from table_contract import (ROOT, GROUPS, matrix, query_contract, from_ranks, recompute,
    validate_matrix, compare_measurements, objsha, sha)
from summary_tables import build, key
from bind_results import render
from recompute_tables import ReleaseReader
import recompute_tables


def fixture():
    registry = json.loads((ROOT/'metadata/RESULT_SLOTS.json').read_text())
    gallery = ['img'+str(i) for i in range(8)]
    queries = []; first = []
    # Non-gallery CSV encounter order; repeated caption text hashes are kept.
    for i,image in enumerate(reversed(gallery)):
        first.append(len(queries))
        for c in range(2 if i%2 == 0 else 1):
            queries.append({'query_index':len(queries),'query_id':f'{image}#caption={c+1}',
                            'gold_gallery_index':gallery.index(image),
                            'caption_sha256':hashlib.sha256(('same text' if c else image).encode()).hexdigest()})
    identities = {'schema':'pare.public-query-identities.v1','raw_caption_text_included':False,
                  'gallery_ids':gallery,'queries':queries,'first_caption_all_query_indices':first}
    names = ['lt_0.30','0.30_to_lt_0.45','ge_0.45','unavailable']
    grouping = {'test_ids':gallery,'test_ids_sha256':objsha(gallery),'n_gallery_retained_for_all_subgroups':8,
                'groups':{name:gallery[i*2:i*2+2] for i,name in enumerate(names)},'counts':dict.fromkeys(names,2)}
    observations = []; ranks = {}
    for i,k in enumerate(sorted(matrix(),key=repr)):
        cohort,bone,model,seed,epoch,factor = k
        jid = 'synthetic_'+str(i)
        # Ensure both signs occur, without depending on randomized Python hash.
        offset = int(hashlib.sha256(repr(k).encode()).hexdigest()[:8],16)%8
        rank_values = [1+(offset+q)%8 for q in range(len(queries))]
        if model == 'adapter':
            rank_values = [1+(q+(seed or 0))%3 for q in range(len(queries))]
        elif model == 'path_100__full':
            rank_values = [4+(q+(seed or 0))%5 for q in range(len(queries))]
        rank = {'job_id':jid,'ranks':rank_values,'first_ranks':[rank_values[n] for n in first],
                'exact_gold_tie_sizes':[1]*len(queries)}
        observed = dict(zip(('cohort','backbone','model_id','seed','epoch','factor'),k),job_id=jid,
                        checkpoint=None if model == 'frozen_scene' else {'path':'weights/synthetic.pt','sha256':'a'*64},
                        score_sha256='b'*64)
        observed.update(from_ranks(rank,gallery,queries,first,grouping))
        observations.append(observed); ranks[jid] = rank
    summary = build(observations,registry)
    summary['result_slots_sha256'] = sha(ROOT/'metadata/RESULT_SLOTS.json')
    return registry,identities,grouping,ranks,summary


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry,cls.identities,cls.grouping,cls.ranks,cls.summary = fixture()

    def test_all_315_and_all_measurements_and_327_slots(self):
        result = recompute(self.summary,self.identities,self.grouping,self.registry,
                           lambda row:self.ranks[row['job_id']])
        self.assertEqual(len(result['observations']),315)
        self.assertEqual(result['measurements'],self.summary['measurements'])
        self.assertGreater(len(result['measurements']),327)
        summary_binding = {'path':'synthetic_fixture/SUMMARY.json','sha256':'a'*64}
        slots_binding = {'path':'synthetic_fixture/RESULT_SLOTS.json','sha256':sha(ROOT/'metadata/RESULT_SLOTS.json')}
        audit_fixture = {'schema':'pare.formal-score-audit.v1','status':'audit_passed','errors':[],
                         'jobs':315,'summary':summary_binding,'result_slots':slots_binding}
        text,values,manual = render(audit_fixture,self.summary,self.registry,summary_binding,slots_binding)
        self.assertEqual((len(values),len(manual)),(327,6))
        self.assertEqual(text.count('\\expandafter\\def'),327)
        self.assertTrue(any(r['value']<0 for r in result['measurements'] if r['stat']=='paired_mean_delta'))
        self.assertEqual(self.identities['first_caption_all_query_indices'],[0,2,3,5,6,8,9,11])

    def test_macro_mean_and_sample_sd_use_seed_then_backbone(self):
        rows = [r for r in self.summary['observations'] if r['model_id']=='original_full'
                and r['epoch']==50 and r['factor']=='1' and r['seed'] in (1,2,3,4)]
        values = [math.fsum(r['query_modes']['first-caption']['R@1'] for r in rows if r['seed']==s)/5 for s in (1,2,3,4)]
        chosen = [r for r in self.summary['measurements'] if r['backbone']=='macro_equal_five_backbones'
                  and r['model_id']=='original_full' and r['epoch']==50 and r['factor']=='1'
                  and r['query_mode']=='first-caption' and r['metric']=='R@1'
                  and r['seed_group']=='primary_n4' and r['subgroup'] is None]
        by_stat = {r['stat']:r for r in chosen}
        self.assertEqual(by_stat['mean']['value'],math.fsum(values)/4)
        self.assertEqual(by_stat['sample_sd']['value'],statistics.stdev(values))
        self.assertEqual(by_stat['mean']['seed_values'],values)

    def test_signed_pairs_and_frozen_single_observation(self):
        negatives = [r for r in self.summary['measurements'] if r['model_id']=='path_100__full'
                     and r['comparator_model_id']=='adapter' and r['metric']=='R@1'
                     and r['stat']=='paired_mean_delta']
        self.assertTrue(negatives and all(r['value']<0 for r in negatives))
        frozen = [r for r in self.summary['measurements'] if r['model_id']=='frozen_scene']
        self.assertTrue(all(r['n_seeds']==1 and r['seeds']==[None] and r['stat']=='mean' for r in frozen))

    def test_hidden_non_slot_measurement_corruption_is_rejected(self):
        slot_keys = {key(r) for r in self.registry['slots'] if r['kind']=='measurement'}
        bad = copy.deepcopy(self.summary)
        row = next(r for r in bad['measurements'] if key(r) not in slot_keys and r['value'] is not None)
        row['value'] += 0.01
        with self.assertRaisesRegex(ValueError,'Numeric value differs'):
            recompute(bad,self.identities,self.grouping,self.registry,lambda r:self.ranks[r['job_id']])

    def test_corrupt_ranks_are_rejected(self):
        ranks = copy.deepcopy(self.ranks); jid = self.summary['observations'][0]['job_id']
        ranks[jid]['ranks'][0] = 9
        with self.assertRaisesRegex(ValueError,'Invalid ranks'):
            recompute(self.summary,self.identities,self.grouping,self.registry,lambda r:ranks[r['job_id']])
        ranks = copy.deepcopy(self.ranks); ranks[jid]['first_ranks'] = ranks[jid]['ranks'][:8]
        with self.assertRaisesRegex(ValueError,'Stored first-caption ranks'):
            recompute(self.summary,self.identities,self.grouping,self.registry,lambda r:ranks[r['job_id']])

    def test_missing_duplicate_unknown_scale_matrix_rejected(self):
        for kind in ('missing','duplicate','unknown','scale'):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(self.summary['observations'])
                if kind=='missing': rows.pop()
                elif kind=='duplicate': rows[-1] = rows[0]
                elif kind=='unknown': rows[0]['model_id'] = 'unapproved'
                else: rows[0]['factor'] = '3'
                with self.assertRaisesRegex(ValueError,'matrix'):
                    validate_matrix(rows)

    def test_groups_and_identity_map_rejected_when_inconsistent(self):
        bad = copy.deepcopy(self.grouping); bad['groups']['unavailable'].pop()
        with self.assertRaisesRegex(ValueError,'partition'):
            query_contract(self.identities,bad)
        bad = copy.deepcopy(self.identities); bad['first_caption_all_query_indices'] = list(range(8))
        with self.assertRaisesRegex(ValueError,'First-caption coverage'):
            query_contract(bad,self.grouping)
        bad = copy.deepcopy(self.identities); bad['queries'][0]['caption'] = 'must not be published'
        with self.assertRaisesRegex(ValueError,'Unexpected query field'):
            query_contract(bad,self.grouping)
        with self.assertRaisesRegex(ValueError,'Full public benchmark'):
            query_contract(self.identities,self.grouping,{'digests':{}})

    def test_release_reader_hash_path_mutation_and_failure_markers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'x.json').write_text('{}')
            manifest = {'schema':'pare.public-fulltrain-release.v1','status':'exported_from_complete_independent_audits',
                        'formal':{'trajectories':145,'checkpoints':290,'score_jobs':315,'rank_files':315},
                        'files':{'x.json':{'sha256':sha(root/'x.json')}}}
            (root/'MANIFEST.json').write_text(json.dumps(manifest))
            reader = ReleaseReader(root,sha(root/'MANIFEST.json')); self.assertEqual(reader.read('x.json'),{})
            with self.assertRaisesRegex(ValueError,'Nonportable'):
                reader.bind('../private')
            (root/'x.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed during'):
                reader.finish()
            (root/'EXPORT_FAILED.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'failure marker'):
                ReleaseReader(root,sha(root/'MANIFEST.json'))

    def test_public_export_bridge_genuine_binding_shape(self):
        # All records here are synthetic fixtures, not a public or real audit.
        # Only official query population-size enforcement is bypassed; the
        # production release/audit/rank/hash/renderer bridges run unchanged.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); package = root/'companion'; release = root/'synthetic_release'
            (package/'metadata').mkdir(parents=True); release.mkdir()
            def save(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value,indent=2)+'\n')
            for name,value in [('RESULT_SLOTS.json',self.registry),('GSD_GROUPS.json',self.grouping),('FIXED_QUERY_PROTOCOL.json',{})]:
                if name=='RESULT_SLOTS.json':
                    (package/'metadata'/name).write_bytes((ROOT/'metadata'/name).read_bytes())
                else:
                    save(package/'metadata'/name,value)
            original = copy.deepcopy(self.summary); public = copy.deepcopy(original)
            original_rank_bindings = []
            for row in public['observations']:
                rank_path = 'results/ranks/'+row['job_id']+'.json'
                save(release/rank_path,self.ranks[row['job_id']])
                row['ranks'] = {'path':rank_path,'sha256':sha(release/rank_path)}
                if row['checkpoint']:
                    row['source_checkpoint_sha256'] = row['checkpoint']['sha256']
                    row['checkpoint'] = {'path':'weights/projected.pt','sha256':'c'*64}
                original_rank_bindings.append({'path':'/synthetic/audit/'+row['job_id']+'.json','sha256':sha(release/rank_path)})
            save(release/'results/SUMMARY.json',public)
            save(release/'results/QUERY_IDENTITIES.json',self.identities)
            save(release/'provenance/formal_score_summary.original.json',original)
            audit = {'schema':'pare.formal-score-audit.v1','status':'audit_passed','errors':[],
                     'jobs':315,'unique_trained_checkpoints':290,
                     'summary':{'path':'/synthetic/audit/SUMMARY.json','sha256':sha(release/'provenance/formal_score_summary.original.json')},
                     'result_slots':{'path':'/synthetic/slots.json','sha256':sha(package/'metadata/RESULT_SLOTS.json')},
                     'artifact_sha256':{'/synthetic/GSD_GROUPS.json':sha(package/'metadata/GSD_GROUPS.json')},
                     'independent_rank_files':original_rank_bindings}
            save(release/'provenance/formal_score_audit.original.json',audit)
            files = {str(p.relative_to(release)):{'sha256':sha(p)} for p in release.rglob('*') if p.is_file()}
            manifest = {'schema':'pare.public-fulltrain-release.v1','status':'exported_from_complete_independent_audits',
                        'formal':{'trajectories':145,'checkpoints':290,'score_jobs':315,'rank_files':315,
                                  'source_score_audit_sha256':sha(release/'provenance/formal_score_audit.original.json')},'files':files}
            save(release/'MANIFEST.json',manifest)
            reader = ReleaseReader(release,sha(release/'MANIFEST.json'))
            def small_fixture(summary,identities,grouping,registry,loader,fixed):
                return recompute(summary,identities,grouping,registry,loader,fixed=None)
            with patch.object(recompute_tables,'ROOT',package), patch.object(recompute_tables,'recompute',small_fixture):
                result,tex,values,manual = recompute_tables.prepare(reader)
                self.assertEqual((len(result['observations']),len(values),len(manual)),(315,327,6))
                reader.finish()
                manifest['files']['results/ranks/extra.json'] = {'sha256':'d'*64}
                save(release/'MANIFEST.json',manifest)
                with self.assertRaisesRegex(ValueError,'extra public rank'):
                    recompute_tables.prepare(ReleaseReader(release,sha(release/'MANIFEST.json')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
