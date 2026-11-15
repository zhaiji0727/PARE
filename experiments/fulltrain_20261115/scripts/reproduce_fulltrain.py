"""Canonical staged reproduction entry. Default only prints a plan; never computes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {'inputs': ('raw_inputs', 'prepare_inputs.py'),
          'train': ('training', 'train_replay.py'),
          'evaluate': ('evaluation', 'evaluate.py')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Explicitly run one requested stage')
    parser.add_argument('stage', choices=STAGES, nargs='?')
    args, forwarded = parser.parse_known_args(argv)
    if args.stage is None:
        parser.print_help(); return 0
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    package, script = STAGES[args.stage]
    directory = ROOT/'software'/package
    lock = json.loads((directory/'SOURCE_LOCK.json').read_text())
    for name, expected in lock['files'].items():
        path = (directory/name).resolve()
        if directory.resolve() not in path.parents or digest(path) != expected:
            raise ValueError('Source lock differs: '+name)
    if args.stage == 'train' and '--protocol' not in forwarded:
        protocol = directory/'PROTOCOL.json'
        forwarded = ['--protocol', str(protocol), '--protocol-sha256', digest(protocol)] + forwarded
    if args.stage == 'inputs' and '--execute' not in forwarded:
        forwarded = ['--execute'] + forwarded
    command = [sys.executable, str(directory/script), *forwarded]
    if not args.execute:
        print(json.dumps({'status': 'planned_only', 'stage': args.stage, 'command': command,
                          'models_constructed': False, 'training_or_scoring_started': False,
                          'instruction': 'Use --execute before the stage name to run this stage.'}, indent=2))
        return 0
    if '--device' not in forwarded and '--help' not in forwarded:
        raise ValueError('Execution requires explicit --device cpu or --device cuda')
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
