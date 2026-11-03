#!/usr/bin/env python3
"""Launch the already-reviewed fixed 42-run plan on three idle physical GPUs.

Run only after the root review gate. Each seed process runs all 14 configurations.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = json.loads((ROOT / 'FOLLOWUP_SOURCE_FROZEN.json').read_text())
    for relative, digest in frozen['file_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f'Frozen file mismatch: {relative}')
    for name in ['equivalence_cpu.json', 'equivalence_cuda.json', 'training_kernel_cpu.json', 'training_kernel_cuda.json']:
        proof = json.loads((ROOT / 'validation' / name).read_text())
        if not proof['all_passed']:
            raise RuntimeError(f'Validation failed: {name}')
    preflight = json.loads((ROOT / 'validation/preflight_seed0/manifest.json').read_text())
    if preflight['status'] != 'preflight_passed_no_training':
        raise RuntimeError('Real-input preflight did not pass')
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    identifiers = {int(line.split(',')[0]): line.split(',')[1].strip() for line in gpus.splitlines()}
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()
    if any(identifiers[index] in busy for index in [0, 1, 2]):
        raise RuntimeError('A planned GPU has another compute process; refusing to launch')
    results = ROOT / 'results'
    results.mkdir(exist_ok=False)
    records = []
    for seed in [0, 1, 2]:
        command = [str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'scripts/run_factorial_fitval.py'),
                   '--protocol', str(ROOT / 'plans/protocol.json'), '--base-code', str(ROOT / 'base_code'),
                   '--cache', str(ROOT / 'inputs/cache_physics_train.pt'),
                   '--captions', str(ROOT / 'data/ret_clean_20261007/Caption_train.csv'),
                   '--split-dir', str(ROOT / 'data/ret_clean_20261007'),
                   '--out', str(results / f'seed_{seed}'), '--seed', str(seed)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(seed), PARE_DEVICE='cuda:0',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (results / f'seed_{seed}.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        records.append({'seed': seed, 'physical_gpu': seed, 'pid': process.pid, 'command': command,
                        'log': f'results/seed_{seed}.log', 'started_unix': time.time()})
        (results / 'launch_manifest.json').write_text(json.dumps({
            'source_freeze_sha256': sha(ROOT / 'FOLLOWUP_SOURCE_FROZEN.json'),
            'protocol_sha256': frozen['file_sha256']['plans/protocol.json'], 'processes': records}, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
