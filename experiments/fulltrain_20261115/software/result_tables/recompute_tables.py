"""Recompute public tables from saved ranks; never loads models, weights or scores."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

from table_contract import ROOT, sha, require, recompute, compare_original
from bind_results import render
from summary_tables import KEYS


def write(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


class ReleaseReader:
    def __init__(self, root, manifest_sha):
        self.root = root.resolve(); self.files = {}
        require(not (self.root/'EXPORT_FAILED.json').exists(),'Release has an export-failure marker')
        require(sha(self.root/'MANIFEST.json') == manifest_sha,'Release manifest SHA differs')
        self.manifest = json.loads((self.root/'MANIFEST.json').read_text())
        require(self.manifest['schema'] == 'pare.public-fulltrain-release.v1'
                and self.manifest['status'] == 'exported_from_complete_independent_audits', 'Incomplete release export')
        formal = self.manifest['formal']
        require((formal['trajectories'],formal['checkpoints'],formal['score_jobs'],formal['rank_files']) == (145,290,315,315),
                'Release matrix differs')
        self.files['MANIFEST.json'] = manifest_sha

    def bind(self, name):
        require(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts,'Nonportable release path')
        require(name in self.manifest['files'],'File absent from release manifest: '+name)
        row = self.manifest['files'][name]
        require(isinstance(row['sha256'],str) and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Invalid release SHA')
        path = (self.root/name).resolve()
        require(self.root in path.parents and path.is_file(), 'Release path escapes root')
        require(sha(path) == row['sha256'], 'Release file SHA differs: '+name)
        self.files[name] = row['sha256']; return path

    def read(self, name):
        return json.loads(self.bind(name).read_text())

    def finish(self):
        for name,digest in self.files.items():
            require(sha(self.root/name) == digest,'Release changed during table reconstruction: '+name)


def sources():
    path = ROOT/'SOURCE_LOCK.json'; lock = json.loads(path.read_text())
    require(lock['schema'] == 'pare.public-rank-tables-source-lock.v1','Wrong source lock')
    required = {'recompute_tables.py','table_contract.py','runtime/independent_core.py',
                'runtime/summary_tables.py','runtime/bind_results.py','runtime/audit_scores.py',
                'metadata/GSD_GROUPS.json','metadata/RESULT_SLOTS.json','metadata/FIXED_QUERY_PROTOCOL.json'}
    require(required <= set(lock['files']),'Incomplete source lock')
    for name,digest in lock['files'].items():
        source = (ROOT/name).resolve()
        require(ROOT in source.parents and not Path(name).is_absolute() and sha(source) == digest,'Source changed: '+name)
    return sha(path),lock


def prepare(reader):
    summary = reader.read('results/SUMMARY.json')
    identities = reader.read('results/QUERY_IDENTITIES.json')
    audit_name = 'provenance/formal_score_audit.original.json'
    audit = reader.read(audit_name)
    require(reader.files[audit_name] == reader.manifest['formal']['source_score_audit_sha256'],'Wrong original audit')
    require(audit['schema'] == 'pare.formal-score-audit.v1' and audit['status'] == 'audit_passed'
            and audit['errors'] == [] and audit['jobs'] == 315 and audit['unique_trained_checkpoints'] == 290,'Incomplete original audit')
    original_name = 'provenance/formal_score_summary.original.json'
    original = reader.read(original_name)
    require(reader.files[original_name] == audit['summary']['sha256'],'Original summary does not belong to audit')
    compare_original(summary,original)
    slots = json.loads((ROOT/'metadata/RESULT_SLOTS.json').read_text())
    slots_sha = sha(ROOT/'metadata/RESULT_SLOTS.json')
    require(slots_sha == audit['result_slots']['sha256'] == original['result_slots_sha256'] == summary['result_slots_sha256'],
            'Result slots differ from the actual formal audit')
    grouping = json.loads((ROOT/'metadata/GSD_GROUPS.json').read_text())
    require(sha(ROOT/'metadata/GSD_GROUPS.json') in audit['artifact_sha256'].values(),'GSD groups were not bound by the actual audit')
    fixed = json.loads((ROOT/'metadata/FIXED_QUERY_PROTOCOL.json').read_text())
    ranks = {Path(r['path']).stem:r for r in audit['independent_rank_files']}
    require(len(ranks) == len(audit['independent_rank_files']) == 315,'Incomplete original rank inventory')
    require(set(ranks) == {r['job_id'] for r in summary['observations']},'Original rank job identities differ')
    require({name for name in reader.manifest['files'] if name.startswith('results/ranks/')}
            == {f'results/ranks/{jid}.json' for jid in ranks}, 'Missing/extra public rank files')
    def load_rank(row):
        name = row['ranks']['path']
        require(name == f"results/ranks/{row['job_id']}.json",'Unexpected public rank path')
        value = reader.read(name)
        require(reader.files[name] == row['ranks']['sha256'] == ranks[row['job_id']]['sha256'],
                'Public rank bytes differ from independently audited original ranks')
        return value
    rebuilt = recompute(summary,identities,grouping,slots,load_rank,fixed)
    # No synthetic audit is constructed. The unchanged renderer consumes the
    # genuine hash-bound source audit/summary after every value matches ranks.
    tex,bound,manual = render(audit,original,slots,audit['summary'],audit['result_slots'])
    require(len(bound) == 327 and len(manual) == 6,'Incomplete numeric/manual slot coverage')
    return rebuilt,tex,bound,manual


def csv_tables(output, result):
    with (output/'PER_SEED_METRICS.csv').open('x',newline='') as stream:
        fields = ['job_id','cohort','backbone','model_id','seed','epoch','factor','query_mode','subgroup','n_queries',
                  'R@1','R@5','R@10','MRR@10']
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in result['observations']:
            base = {k:row[k] for k in fields[:7]}
            for mode,values in row['query_modes'].items():
                writer.writerow(base|{'query_mode':mode,'subgroup':'','n_queries':3688 if mode == 'first-caption' else 6190}|values)
            for name,group in row['gsd'].items():
                writer.writerow(base|{'query_mode':'first-caption','subgroup':name,'n_queries':group['n_queries']}|group['metrics'])
    with (output/'MEASUREMENTS.csv').open('x',newline='') as stream:
        fields = list(KEYS)+['value','n_seeds','seeds','seed_values','n_backbones_per_seed','units','statistical_unit']
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in result['measurements']:
            writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})


def run(release, manifest_sha, output):
    require(not output.exists(),'Output must be new; no overwrite or resume')
    lock_sha,lock = sources()
    reader = ReleaseReader(release,manifest_sha)
    result,tex,bound,manual = prepare(reader)
    require('torch' not in sys.modules,'Saved-rank table reconstruction unexpectedly imported Torch')
    reader.finish(); output.mkdir(parents=True)
    try:
        result.update(schema='pare.public-rank-recomputed-summary.v1',status='matched_exported_formal_summary',
                      scope='Recalculation from saved audited ranks; no score ranking or model evaluation performed.')
        write(output/'RECOMPUTED_SUMMARY.json',result)
        (output/'RESULT_VALUES.tex').write_text('% Checked against every public saved-rank result. No model evaluation performed.\n'+tex)
        write(output/'RESULT_VALUES.binding.json',{'status':'327_numeric_slots_verified_from_public_ranks',
            'bound_measurements':bound,'manual_slots_not_filled':manual,'source_manifest_sha256':manifest_sha,
            'result_slots_sha256':sha(ROOT/'metadata/RESULT_SLOTS.json'),'tex_sha256':sha(output/'RESULT_VALUES.tex')})
        csv_tables(output,result)
        reader.finish(); require(sources() == (lock_sha,lock),'Table source changed during reconstruction')
        files = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
        report = {'schema':'pare.public-rank-table-verification.v1','status':'recomputed_and_exactly_matched',
            'source_lock_sha256':lock_sha,'release_manifest_sha256':manifest_sha,'source_files':reader.files,
            'evaluations':315,'measurements':len(result['measurements']),'numeric_slots':len(bound),
            'manual_slots_not_filled':manual,'all_reported_measurements_compared':True,
            'comparison':'exact finite numeric equality using the unchanged formal-summary arithmetic',
            'scope':{'checkpoint_tensors_read':False,'score_tensors_read':False,'models_evaluated':False,
                     'rank_order_recomputed_from_scores':False,'new_formal_measurements':False},'files':files}
        write(output/'VERIFICATION.json',report)
        write(output/'complete.json',{'schema':'pare.public-rank-table-completion.v1','status':'completed',
             'verification_sha256':sha(output/'VERIFICATION.json'),'source_manifest_sha256':manifest_sha})
        return report
    except BaseException as error:
        write(output/'FAILED.json',{'status':'failed','error':str(error)}); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True,type=Path)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    result = run(args.release,args.manifest_sha256,args.out.resolve())
    print(json.dumps({k:result[k] for k in ('status','evaluations','measurements','numeric_slots')}))


if __name__ == '__main__':
    main()
