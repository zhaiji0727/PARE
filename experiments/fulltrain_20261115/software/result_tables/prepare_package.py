"""Freeze this companion's files and emit a future-integration inventory only."""
import json
from pathlib import Path
from table_contract import ROOT, sha


def main():
    names = ['table_contract.py','recompute_tables.py','test_public_tables.py','prepare_package.py',
             'README.md','SOURCE_PROVENANCE.json']
    names += [str(p.relative_to(ROOT)) for directory in ('runtime','metadata') for p in sorted((ROOT/directory).glob('*')) if p.is_file()]
    files = {name:sha(ROOT/name) for name in names}
    lock = {'schema':'pare.public-rank-tables-source-lock.v1','files':files,
            'scope':'Saved-rank reconstruction only; no score/checkpoint/model reads.'}
    (ROOT/'SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
    assets = files|{'SOURCE_LOCK.json':sha(ROOT/'SOURCE_LOCK.json')}
    plan = {'schema':'pare.public-rank-tables-integration.v1','status':'prepared_not_integrated',
            'assets':[{'source':name,'sha256':digest,'destination':'software/result_tables/'+name}
                      for name,digest in sorted(assets.items())],
            'command':'python software/result_tables/recompute_tables.py --release . --manifest-sha256 EXACT_MANIFEST_SHA --out ./reader_tables',
            'existing_frozen_release_modified':False,'actual_formal_results_read':False,'published':False}
    (ROOT/'INTEGRATION.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'source_lock_sha256':sha(ROOT/'SOURCE_LOCK.json'),'integration_sha256':sha(ROOT/'INTEGRATION.json'),
                      'assets':len(assets)}))


if __name__ == '__main__':
    main()
