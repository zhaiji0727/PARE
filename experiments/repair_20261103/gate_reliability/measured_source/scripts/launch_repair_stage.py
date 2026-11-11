#!/usr/bin/env python3
"""Explicit, persistent three-seed launcher. Never automatically advances stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'selected-control', 'check'], required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--execute', action='store_true', help='Without this option only print the exact commands')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'protocol.json').read_text())
    frozen = ROOT / 'REPAIR_SOURCE_FROZEN.json'
    if not frozen.exists():
        raise ValueError('Root must review and freeze sources before launching')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_repair import check_freeze, validate_plan
    validate_plan(plan)
    check_freeze(argparse.Namespace(freeze=frozen, protocol=ROOT / 'protocol.json', split_dir=ROOT / 'plans/splits'), plan)
    if args.stage != 'train' and not (ROOT / 'results/SELECTION_LOCK.json').exists():
        raise ValueError('Later stages require the locked global selection')
    commands = []
    for seed in plan['seeds']:
        command = [args.python, str(ROOT / 'scripts/run_repair.py'), args.stage, '--seed', str(seed), '--gpu', str(seed)]
        if args.resume: command.append('--resume')
        if args.retry_incomplete: command.append('--retry-incomplete')
        commands.append({'seed': seed, 'gpu': seed, 'argv': command})
    if not args.execute:
        print(json.dumps({'planned_only': True, 'commands': commands}, indent=2))
        return
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    states = {int(parts[0]): (int(parts[1]), int(parts[2])) for line in gpu.splitlines() if (parts := line.split(','))}
    for item in commands:
        used, util = states[item['gpu']]
        if used > 256 or util != 0:
            raise RuntimeError(f'GPU {item["gpu"]} is occupied: {used} MiB, utilization {util}%')
    logdir = ROOT / 'logs'
    logdir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record = {'stage': args.stage, 'started_utc': stamp, 'protocol_sha256': sha(ROOT / 'protocol.json'), 'freeze_sha256': sha(frozen), 'gpu_before': gpu, 'processes': []}
    record_path = logdir / f'launch_{args.stage}_{stamp}.json'
    for item in commands:
        logfile = logdir / f'{args.stage}_seed_{item["seed"]}_{stamp}.log'
        with logfile.open('xb') as output:
            process = subprocess.Popen(item['argv'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record['processes'].append({**item, 'pid': process.pid, 'log': str(logfile.relative_to(ROOT))})
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
