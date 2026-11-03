import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cache_import_test_runner', ROOT / 'scripts/run_repaired_experiment.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_step(tmp_path):
    cache = tmp_path / 'cache.pt'
    cache.write_bytes(b'cache byte fixture')
    source = tmp_path / 'source_manifest.json'
    source.write_text(json.dumps({'steps': [{'name': 'cache_physics_train', 'stage': 'cache', 'status': 'complete'}]}))
    return {'name': 'cache_physics_train', 'stage': 'cache', 'status': 'imported_verified_cache',
            'outputs': [str(cache)], 'artifact_sha256': {str(cache): runner.sha256(cache)},
            'import_provenance': {'source_run_manifest': str(source),
                                  'source_run_manifest_sha256': runner.sha256(source),
                                  'source_step_name': 'cache_physics_train'}}


def test_imported_cache_accepts_bound_original_record(tmp_path):
    runner.verify_completed(make_step(tmp_path))


def test_imported_cache_detects_mutated_cache(tmp_path):
    step = make_step(tmp_path)
    Path(step['outputs'][0]).write_bytes(b'changed cache')
    with pytest.raises(ValueError, match='changed'):
        runner.verify_completed(step)


def test_imported_cache_detects_mutated_source_manifest(tmp_path):
    step = make_step(tmp_path)
    Path(step['import_provenance']['source_run_manifest']).write_text('{}')
    with pytest.raises(ValueError, match='source manifest changed'):
        runner.verify_completed(step)


def test_imported_status_cannot_skip_training(tmp_path):
    step = make_step(tmp_path)
    step['stage'] = 'train'
    with pytest.raises(ValueError, match='Only cache'):
        runner.verify_completed(step)


def test_imported_cache_requires_completed_source_step(tmp_path):
    step = make_step(tmp_path)
    source = Path(step['import_provenance']['source_run_manifest'])
    source.write_text(json.dumps({'steps': [{'name': 'cache_physics_train', 'stage': 'cache', 'status': 'pending'}]}))
    step['import_provenance']['source_run_manifest_sha256'] = runner.sha256(source)
    with pytest.raises(ValueError, match='completed original'):
        runner.verify_completed(step)
