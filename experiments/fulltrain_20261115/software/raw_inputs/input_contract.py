"""Raw input identities and plans. Standard library only; no model imports."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from data_contract import load_captions, training_index, query_sets, objsha

POPULATIONS = {
    'train': (9181, 'aebf2530ab350d94f3a325b514ba872ed1fa0e434ed9d7973c2c209a502b74f5'),
    'test': (3688, 'e1982fc39e596abb1d63bf4bd78cf77a482696af42f2b1c3c2f5ead087fc26d7'),
}
CAPTION_SHA256 = {
    'train': '1c8c524eabb0e7f5e246cc0bb0f0ec774692c233428674acb57af28d1e10589d',
    'test': '7a1878610e0a5089380b2fb7fd825d43edf8682cba513a76ca36661d3094724d',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_sources():
    lock = json.loads((ROOT / 'SOURCE_LOCK.json').read_text())
    required = {'prepare_inputs.py', 'input_contract.py', 'raw_builder.py',
                'runtime/v9_helpers.py', 'runtime/cache_primitives.py', 'runtime/saqre_phys.py',
                'runtime/text_helpers.py', 'runtime/data_contract.py', 'BACKBONES.json'}
    require(required <= set(lock['files']), 'Incomplete input-builder source lock')
    for name, digest in lock['files'].items():
        path = (ROOT / name).resolve()
        require(ROOT in path.parents and not Path(name).is_absolute(), 'Unsafe source lock entry')
        require(sha(path) == digest, 'Source changed: ' + name)
    return lock


def model_files(models, manifest, backbone, config):
    """Read the existing public prepare_models.py report; no searching/downloading."""
    models = models.resolve()
    report = json.loads(manifest.read_text())
    require(report['status'] == 'downloaded_and_hashed', 'Model preparation is incomplete')
    rows = {row['tag']: row for row in report['assets']}
    require(len(rows) == len(report['assets']), 'Duplicate model asset tag')
    result = {}
    for label, tag, expected in (
        ('pretrained', config['pretrained_manifest_tag'], config['expected_pretrained_sha256']),
        ('external', backbone, config['expected_external_sha256']),
    ):
        if expected is None:
            result[label] = None
            continue
        require(tag in rows, 'Missing model asset: ' + tag)
        row = rows[tag]
        # Public report paths are relative to the repository root, starting models/.
        relative = Path(row['path'])
        require(not relative.is_absolute() and relative.parts[:1] == ('models',)
                and '..' not in relative.parts, 'Expected models/... path in public model report')
        path = (models / Path(*relative.parts[1:])).resolve()
        require(models in path.parents, 'Model asset escapes --models')
        require(row['sha256'] == expected and sha(path) == expected, 'Backbone weight bytes differ: ' + tag)
        require(path.stat().st_size == row['bytes'], 'Model byte size differs')
        result[label] = binding(path)
    return result


def make_plan(args):
    verify_sources()
    data, models = args.data.resolve(), args.models.resolve()
    require(data.is_dir() and models.is_dir(), 'Existing --data and --models directories required')
    require(not args.out.exists(), 'Refuse existing output; no overwrite/resume')
    pop = args.population
    ids_path = data / 'ret_clean_20261007' / (pop + '_ids.json')
    captions_path = data / 'ret_clean_20261007' / ('Caption_' + pop + '.csv')
    require(sha(captions_path) == CAPTION_SHA256[pop], 'Caption CSV differs from the fixed public-data protocol')
    ids = json.loads(ids_path.read_text())
    n, ids_digest = POPULATIONS[pop]
    require(len(ids) == len(set(ids)) == n and objsha(ids) == ids_digest,
            'Ordered content-disjoint population differs')
    require(all(isinstance(name, str) and Path(name).name == name for name in ids), 'Unsafe image IDs')
    captions = load_captions(captions_path)
    identity = training_index(ids, captions) if pop == 'train' else query_sets(ids, captions)
    config = json.loads((ROOT / 'BACKBONES.json').read_text())['backbones'][args.backbone]
    manifest = (args.model_manifest or models.parent / 'reports/models_manifest.json').resolve()
    weights = model_files(models, manifest, args.backbone, config)
    wanted = set(ids)
    image_paths = {}
    for directory, _, names in os.walk(data / 'SARimages_pre'):
        for name in names:
            if name in wanted:
                require(name not in image_paths, 'Duplicate image basename: ' + name)
                image_paths[name] = Path(directory) / name
    require(set(image_paths) == wanted, 'Missing requested original image files')
    count = args.max_images or n
    require(1 <= count <= n, 'Invalid --max-images prefix')
    # A smoke binds only the actual prefix image bytes, but the complete IDs/caption identity.
    images = {name: binding(image_paths[name]) for name in ids[:count]}
    plan = {
        'schema': 'pare.portable-raw-input-plan.v1', 'status': 'cpu_plan_only',
        'backbone': args.backbone, 'population': pop, 'dimension': config['dimension'],
        'config': config, 'data': str(data), 'models': str(models),
        'model_manifest': binding(manifest), 'weights': weights,
        'ids': binding(ids_path), 'captions': binding(captions_path),
        'ids_sha256': objsha(ids), 'selected_ids': ids[:count], 'image_files': images,
        'smoke': count != n, 'images_to_encode': count,
        'source_lock_sha256': sha(ROOT / 'SOURCE_LOCK.json'),
        'policy': {
            'scene_coarse': 'old cache_primitives.encode: per-image mixed [scene,coarse,spatial-fine], bs128; keep scene/coarse',
            'fine': 'v9 legacy raster selection; fixed8 FP32 forward; black RGB24 padding discarded',
            'text': 'old shared_text loop; independent bs256 streams; training strings sorted unique',
            'new_reproduction_not_archived_bytes': True, 'download': False, 'extract_archives': False,
        },
        'models_constructed': False, 'retrieval_scoring_performed': False,
    }
    return plan, ids, captions, identity


def recheck_plan(plan):
    require(sha(ROOT / 'SOURCE_LOCK.json') == plan['source_lock_sha256'], 'Source lock changed')
    verify_sources()
    items = [plan['ids'], plan['captions'], plan['model_manifest'],
             *plan['image_files'].values(), *filter(None, plan['weights'].values())]
    for item in items:
        require(sha(item['path']) == item['sha256'], 'Bound input changed: ' + item['path'])
