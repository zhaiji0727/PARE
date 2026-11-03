#!/usr/bin/env python3
"""Wait for this backbone's frozen caches, then import, select, and train.

Stops before evaluation. This controller neither edits sources nor retries a
failed scientific run; every command and transition is written to a state file.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run', type=Path, required=True)
    ap.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.workspace.resolve()
    source_run = args.source_run.resolve()
    tag = source_run.name
    state_path = root / 'logs' / f'training_{tag}_state.json'
    if state_path.exists():
        raise FileExistsError(f'Refusing to overwrite controller state: {state_path}')
    frozen_path = root / 'TRAINING_SOURCE_FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    frozen_hash = sha256(frozen_path)
    state = {'status': 'waiting_for_verified_caches', 'started_utc': now(),
             'source_run': str(source_run), 'workspace': str(root), 'tag': tag,
             'controller_pid': os.getpid(), 'frozen_manifest_sha256': frozen_hash,
             'evaluation_allowed': False, 'events': []}
    save(state_path, state)

    def verify_sources():
        if sha256(frozen_path) != frozen_hash:
            raise RuntimeError('Frozen training manifest changed')
        for name, expected in frozen['file_sha256'].items():
            if sha256(root / name) != expected:
                raise RuntimeError(f'Frozen training source changed: {name}')

    def execute(stage, command):
        verify_sources()
        print(f'{now()} Starting {stage}: {command}', flush=True)
        child = subprocess.Popen(command, cwd=root, env=os.environ.copy())
        event = {'stage': stage, 'started_utc': now(), 'command': command, 'child_pid': child.pid}
        state['events'].append(event)
        state.update(status='running_' + stage, current_child_pid=child.pid, updated_utc=now())
        save(state_path, state)
        returncode = child.wait()
        event.update(returncode=returncode, finished_utc=now())
        save(state_path, state)
        if returncode:
            raise RuntimeError(f'{stage} failed with exit {returncode}; see controller and stage logs')

    try:
        while True:
            source_manifest = source_run / 'run_manifest.json'
            source = json.loads(source_manifest.read_text())
            if source['status'] == 'failed':
                raise RuntimeError(f'Original cache run failed: {source.get("error")}')
            cache_steps = [s for s in source['steps'] if s['stage'] == 'cache']
            if len(cache_steps) == 4 and all(s['status'] == 'complete' for s in cache_steps) and not (source_run / '.runner.lock').exists():
                break
            time.sleep(15)
        verify_sources()
        original = source['config']
        if original['smoke']:
            raise ValueError('Smoke caches are not accepted for formal training')
        python = str(root / '.venv/bin/python')
        execute('cache_import', [python, str(root / 'scripts/import_verified_caches.py'),
                                '--workspace', str(root), '--source-run', str(source_run),
                                '--review-report', str(root / 'validation/equivalence_cuda.json')])
        command = [python, '-u', str(root / 'scripts/run_repaired_experiment.py'),
                   '--workspace', str(root), '--out', str(root / 'results' / tag),
                   '--tag', original['tag'], '--device', original['device'],
                   '--model', original['model'], '--pretrained', original['pretrained']]
        if original.get('remoteclip'):
            command += ['--remoteclip', original['remoteclip']]
        execute('sweep', command + ['--stage', 'sweep'])
        execute('train', command + ['--stage', 'train'])
        state.update(status='complete_sweep_and_train', ready_for_evaluation=True,
                     updated_utc=now(), current_child_pid=None)
        save(state_path, state)
        print(json.dumps({'status': state['status'], 'tag': tag, 'evaluation_started': False}), flush=True)
    except BaseException as error:
        state.update(status='failed', updated_utc=now(), error=repr(error))
        save(state_path, state)
        raise


if __name__ == '__main__':
    main()
