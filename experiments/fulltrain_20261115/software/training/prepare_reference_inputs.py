"""CPU-only export for a later replay of the author's bound training inputs.

This is a reference-input adapter, not a raw-image cache builder or evidence
that a new training run has succeeded. It does not copy the large image cache.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from train_replay import checked_json, bind, sha, write, validate_tensors, require

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--catalog-sha256', required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Do not overwrite an input export')
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    lock = checked_json(ROOT / 'SOURCE_LOCK.json', protocol['source_lock_sha256'])
    for name, digest in lock['files'].items():
        require(sha(ROOT / name) == digest, 'Export source differs: ' + name)
    catalog = checked_json(a.catalog, a.catalog_sha256)['training']
    require(catalog['status'] == 'BOUND_FOR_FINAL_TRAINING', 'Reference catalog status')
    row = catalog['backbones'][a.backbone]
    route = row['v2_inputs'] if a.backbone == 'vitb32' else row['v9_inputs']
    files = {'cache': bind(route['cache'], a.catalog.parent),
             'text': bind(row['shared_text'], a.catalog.parent),
             'text_manifest': bind(row['shared_text_manifest'], a.catalog.parent),
             'captions': bind(catalog['common']['train_captions'], a.catalog.parent)}
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3')
    sys.path.insert(0, str(ROOT / 'runtime'))
    import torch
    from data_contract import load_captions, training_index
    from train_backend import state_sha
    torch.set_num_threads(3)
    cache = torch.load(files['cache'], map_location='cpu', weights_only=True)
    shared = torch.load(files['text'], map_location='cpu', weights_only=True)
    text_manifest = json.loads(files['text_manifest'].read_text())
    require(text_manifest['artifact_sha256'] == row['shared_text']['sha256'], 'Text file/manifest mismatch')
    require(all(shared[k] == v for k, v in text_manifest['identity'].items()), 'Text identity mismatch')
    data = training_index(cache['ids'], load_captions(files['captions']))
    text = {'embeddings': shared['train'], 'backbone': a.backbone,
            'training_text_sha256': data['training_text_sha256']}
    require(shared['backbone'] == a.backbone and shared['training_text_sha256'] == data['training_text_sha256'],
            'Training text order/backbone mismatch')
    require(state_sha({'train': text['embeddings']}) == text_manifest['train_tensor_sha256'], 'Train tensor mismatch')
    signature_path = ['parent_extraction_signature'] if a.backbone == 'vitb32' else ['base_extraction_signature']
    signature = cache['provenance']['signature']
    encoder = signature
    for key in signature_path:
        encoder = encoder[key]
    inputs = {'schema': 'pare.portable-full-training-inputs.v1', 'status': 'BOUND',
        'backbone': a.backbone, 'dimension': cache['D'],
        'cache': {'path': str(files['cache']), 'sha256': route['cache']['sha256']},
        'captions': {'path': str(files['captions']), 'sha256': catalog['common']['train_captions']['sha256']},
        'training_ids_sha256': data['training_ids_sha256'],
        'pair_order_sha256': data['pair_order_sha256'], 'training_text_sha256': data['training_text_sha256'],
        'training_tensor_sha256': text_manifest['train_tensor_sha256'],
        'cache_signature_sha256': cache['provenance']['signature_sha256'],
        'encoder_signature_path': signature_path, 'encoder_signature': encoder,
        'origin': {
            'catalog': {'path': str(a.catalog.resolve()), 'sha256': a.catalog_sha256},
            'shared_text': {'path': str(files['text']), 'sha256': row['shared_text']['sha256']},
            'text_manifest': {'path': str(files['text_manifest']),
                              'sha256': row['shared_text_manifest']['sha256']},
            'exporter_sha256': sha(__file__),
            'source_lock_sha256': protocol['source_lock_sha256']}}
    validate_tensors(cache, text, inputs)
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / 'train_only.pt').open('xb') as stream:
        torch.save(text, stream)
    inputs['training_text'] = {'path': 'train_only.pt', 'sha256': sha(a.output / 'train_only.pt')}
    for key, path in files.items():
        expected = route['cache']['sha256'] if key == 'cache' else (
            row['shared_text']['sha256'] if key == 'text' else row['shared_text_manifest']['sha256']
            if key == 'text_manifest' else catalog['common']['train_captions']['sha256'])
        require(sha(path) == expected, 'Reference input changed during export')
    require(not torch.cuda.is_initialized(), 'CPU input export initialized CUDA')
    require(sha(a.catalog) == a.catalog_sha256, 'Catalog changed during export')
    for name, digest in lock['files'].items():
        require(sha(ROOT / name) == digest, 'Export source changed: ' + name)
    # Publish a BOUND manifest only after every end-of-export check succeeds.
    write(a.output / 'INPUTS.json', inputs)
    write(a.output / 'EXPORT_RECEIPT.json', {'status': 'cpu_inputs_exported_no_training',
        'catalog_sha256': a.catalog_sha256, 'exporter_sha256': sha(__file__),
        'inputs_sha256': sha(a.output / 'INPUTS.json'), 'cuda_initialized': False,
        'training_performed': False, 'retrieval_scoring_performed': False,
        'test_text_vectors_exported': False})
    print(json.dumps({'backbone': a.backbone, 'status': 'cpu_inputs_exported_no_training',
                      'inputs_sha256': sha(a.output / 'INPUTS.json')}))


if __name__ == '__main__':
    main()
