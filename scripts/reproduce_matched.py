#!/usr/bin/env python3
"""Run the fixed matched study against its byte-identical experimental source.

The repository's current documentation may change without invalidating the
archived experiment contract. This launcher creates a detached source worktree;
it never edits the current checkout or relaxes SOURCE_REQUIREMENTS.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PIN = 'b174e52fa987085f09b2c75ca173fa0ca042c484'

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hf-home', type=Path)
    p.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'])
    p.add_argument('--stage', required=True, choices=['plan','preflight','cache','train','evaluate','scale-cache','evaluate-scales'])
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    a.out = a.out.resolve()
    source = a.out.parent / (a.out.name + '.source-b174e52')
    # A normal clone includes this ancestor. Never substitute an arbitrary HEAD.
    if git(repo, 'rev-parse', PIN + '^{commit}') != PIN:
        raise RuntimeError('The exact experimental commit is unavailable')
    if source.exists():
        if git(source, 'rev-parse', 'HEAD') != PIN:
            raise RuntimeError('Existing source worktree has a different commit')
        if git(source, 'status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError('Experimental source has tracked modifications')
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(source),PIN],check=True)
    script = source / 'experiments/simplified_20261109/portable/run.py'
    cmd = [sys.executable, str(script), '--repo', str(source),
           '--data', str(a.data.resolve()), '--models', str(a.models.resolve()),
           '--out', str(a.out), '--device', a.device, '--stage', a.stage]
    if a.hf_home:
        cmd += ['--hf-home', str(a.hf_home.resolve())]
    a.out.mkdir(parents=True, exist_ok=True)
    provenance = {'schema':'pare.pinned-launcher.v1', 'experimental_commit':PIN,
                  'launcher_commit':git(repo,'rev-parse','HEAD'),
                  'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'argv':cmd, 'source_worktree':str(source),
                  'contract':'Unmodified experimental SOURCE_REQUIREMENTS is enforced by the child entry.'}
    record = a.out / ('launcher_' + a.stage + '.json')
    if record.exists():
        raise FileExistsError('Prior launcher record exists; use a fresh output or inspect it before retrying: ' + str(record))
    record.write_text(json.dumps(provenance, indent=2) + '\n')
    result = subprocess.run(cmd)
    provenance['returncode'] = result.returncode
    record.write_text(json.dumps(provenance, indent=2) + '\n')
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
