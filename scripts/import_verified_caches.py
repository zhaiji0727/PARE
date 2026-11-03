#!/usr/bin/env python3
"""Import completed frozen caches with explicit, checked source provenance.

No cache bytes are copied or rewritten, and no cache-generation command is run.
The new run uses symlinks and preserves the original execution record.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--review-report', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source_run = args.source_run.resolve()
    output = (args.out or workspace / 'results' / source_run.name).resolve()
    script = workspace / 'scripts/run_repaired_experiment.py'
    spec = importlib.util.spec_from_file_location('verified_import_runner', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if 'batch_scoring.py' not in runner.SOURCE_FILES:
        raise ValueError('Training runner must bind the reviewed batch-scoring helper')
    if 'imported_verified_cache' not in runner.COMPLETED_STATUSES:
        raise ValueError('Runner does not support explicit imported-cache status')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    if (source_run / '.runner.lock').exists():
        raise ValueError('Original cache runner must finish before import')
    source_manifest = source_run / 'run_manifest.json'
    source_sha = runner.sha256(source_manifest)
    source = json.loads(source_manifest.read_text())
    original_config = source['config']
    if original_config['smoke'] or source['status'] not in ('partial', 'complete'):
        raise ValueError('Only completed formal caches may be imported')
    originals = {step['name']: step for step in source['steps'] if step['stage'] == 'cache'}
    if len(originals) != 4 or any(step['status'] != 'complete' for step in originals.values()):
        raise ValueError('All four original cache steps must be complete')
    review_path = args.review_report.resolve()
    review = json.loads(review_path.read_text())
    if review.get('all_passed') is not True:
        raise ValueError('Optimization equivalence report must record all_passed=true')
    review_sha = runner.sha256(review_path)
    command = [sys.executable, str(script), '--workspace', str(workspace), '--out', str(output),
               '--tag', original_config['tag'], '--device', original_config['device'], '--dry-run',
               '--model', original_config['model'], '--pretrained', original_config['pretrained']]
    if original_config.get('remoteclip'):
        command += ['--remoteclip', original_config['remoteclip']]
    plan_run = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(plan_run.stdout)
    config = plan['config']
    for key in ('model', 'pretrained', 'remoteclip_sha256', 'smoke', 'seed', 'split_manifest_sha256'):
        if config[key] != original_config[key]:
            raise ValueError(f'Imported protocol/input mismatch: {key}')
    for name in ('cache_primitives.py', 'saqre_phys.py', 'primitive_controls.py', 'runtime.py', 'artifacts.py'):
        if config['source_sha256'][name] != original_config['source_sha256'][name]:
            raise ValueError(f'Cache-generating implementation changed: {name}')
    for step in originals.values():
        runner.verify_completed(step)
        if runner.sha256(step['log']) != step['log_sha256']:
            raise ValueError(f'Original cache log changed: {step["name"]}')
    env = runner.environment(workspace, output, config['device'])
    manifest = {
        'status': 'partial', 'created_utc': runner.now(), 'config': config,
        'purpose': 'Formal repaired-protocol experiment; explicitly imported verified caches and reviewed optimized training',
        'environment': {'platform': platform.platform(), 'python': sys.version,
                        'packages': {name: importlib.metadata.version(name) for name in
                                     ('torch', 'open_clip_torch', 'numpy', 'Pillow', 'scipy')},
                        'variables': {key: env[key] for key in ('PARE_DEVICE', 'HF_HOME', 'HF_HUB_OFFLINE',
                                     'TRANSFORMERS_OFFLINE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM')}},
        'optimization_review': {'path': str(review_path), 'sha256': review_sha, 'all_passed': True},
        'cache_import': {'status': 'imported_verified_cache', 'importer_path': str(Path(__file__).resolve()),
                         'importer_sha256': runner.sha256(__file__), 'source_run_manifest': str(source_manifest),
                         'source_run_manifest_sha256': source_sha, 'cache_bytes_rewritten': False},
        'steps': [{**step, 'status': 'pending'} for step in plan['steps']],
        'all_required_steps_complete': False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    lock = output / '.cache_import.lock'
    lock.write_text(str(os.getpid()))
    try:
        for step in manifest['steps']:
            if step['stage'] != 'cache':
                continue
            old = originals[step['name']]
            if len(old['outputs']) != len(step['outputs']):
                raise ValueError('Cache output topology mismatch')
            for old_path, new_path in zip(old['outputs'], step['outputs']):
                Path(new_path).symlink_to(Path(old_path).resolve())
            for key in ('started_utc', 'finished_utc', 'returncode', 'executed_command', 'log', 'log_sha256'):
                step[key] = old[key]
            step.update({
                'status': 'imported_verified_cache', 'imported_utc': runner.now(),
                'cache_generation_executed_in_this_run': False,
                'planned_command_not_executed': step['command'],
                'import_provenance': {'source_run_manifest': str(source_manifest),
                                     'source_run_manifest_sha256': source_sha,
                                     'source_step_name': old['name'],
                                     'original_command': old['command'],
                                     'original_executed_command': old['executed_command'],
                                     'original_outputs': old['outputs'],
                                     'original_artifact_sha256': old['artifact_sha256'],
                                     'original_log': old['log'], 'original_log_sha256': old['log_sha256'],
                                     'original_verification': old['verification']},
            })
            step['verification'] = runner.verify_step(step, config)
            step['artifact_sha256'] = runner.artifact_hashes(step['outputs'])
            runner.verify_completed(step)
        if runner.sha256(source_manifest) != source_sha or runner.sha256(review_path) != review_sha:
            raise ValueError('Source manifest or equivalence report changed during import')
        if config['source_sha256'] != {name: runner.sha256(workspace / 'code' / name) for name in runner.SOURCE_FILES}:
            raise ValueError('Training source changed during import')
        manifest['updated_utc'] = runner.now()
        runner.write_json(output / 'run_manifest.json', manifest)
        print(json.dumps({'status': 'imported_verified_cache', 'manifest': str(output / 'run_manifest.json'),
                          'cache_count': 4, 'original_run_manifest_sha256': source_sha,
                          'cache_generation_executed': False}, indent=2))
    except BaseException as error:
        runner.write_json(output / 'IMPORT_FAILED.json', {'status': 'failed', 'error': repr(error),
                          'source_run_manifest': str(source_manifest), 'failed_utc': runner.now()})
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
