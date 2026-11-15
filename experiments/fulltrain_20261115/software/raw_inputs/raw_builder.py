"""Assemble raw inputs with the unchanged old mixed-batch and v9 fine helpers."""
import json
from pathlib import Path
import shutil

from input_contract import ROOT, binding, objsha, require, recheck_plan, sha, write


def empty_cache(ids, dimension):
    import torch
    n = len(ids)
    out = {'ids': list(ids), 'D': dimension, 'Mc': 6, 'Mf': 8}
    for key in ('scene', 'agg'):
        out[key] = torch.zeros(n, dimension)
    for key in ('nu', 'L', 'c_star'):
        out[key] = torch.zeros(n)
    for branch, slots in [('c', 6), ('f', 8)]:
        out[branch + '_emb'] = torch.zeros(n, slots, dimension)
        out[branch + '_mu'] = torch.zeros(n, slots)
        out[branch + '_t'] = torch.zeros(n, slots)
        out[branch + '_tf'] = torch.zeros(n, slots, 4)
        out[branch + '_n'] = torch.zeros(n, dtype=torch.long)
    return out


def fill_image(cache, row, image, model, preprocess, device):
    import numpy as np
    import torch
    import cache_primitives as old
    import saqre_phys as extractor
    import v9_helpers as v9
    prim = extractor.extract_primitives(image, max_regions=6, max_sc=8)
    # Preserve the old MIXED forward shape, including the old spatial fine crops.
    # Encoding only scene+coarse would silently change the archived batch policy.
    crops = [prim['scene'][0]] + [p for p, _ in prim['regions']] + [p for p, _ in prim['scatterers']]
    embs = old.encode(model, preprocess, device, crops, bs=128)
    require(embs.shape == (len(crops), cache['D']), 'Mixed embedding shape differs')
    cache['scene'][row] = embs[0]
    cache['c_n'][row] = len(prim['regions'])
    for i, (_, features) in enumerate(prim['regions']):
        cache['c_emb'][row, i] = embs[1 + i]
        cache['c_mu'][row, i] = features['mu']
        cache['c_t'][row, i] = features['t']
        cache['c_tf'][row, i] = torch.as_tensor(features['typefeat'])
    for key in ('nu', 'L', 'c_star'):
        cache[key][row] = prim[key]
    g = extractor.to_intensity(image)
    det = extractor.ca_cfar(g)
    boxes = v9.legacy_boxes(g, det, max_sc=8, half=12)
    # v9 computes c_star from float64 ENL before storing a float32 scalar.
    c_star = 1.0 / np.sqrt(extractor.estimate_enl(g))
    features = v9.fine_features(g, det, c_star, boxes, extractor)
    fine = v9.encode_fine(model, preprocess, device, image, boxes, slots=8)
    v9.assign_fine(cache, row, fine, features)
    require(all(bool(torch.isfinite(v[row]).all()) for v in cache.values() if torch.is_tensor(v)),
            'Nonfinite cache values')
    return {'legacy_boxes': boxes, 'mixed_forward_batch_size': len(crops), 'fine_forward_batch_size': 8}


def setup_encoder(plan, device):
    import importlib.metadata
    import torch
    import open_clip
    import artifacts
    import runtime
    import saqre_phys
    require(saqre_phys.EXTRACTOR_VERSION == '2026-10-07-float64-spatial-ties', 'Wrong numerical extractor')
    config = plan['config']
    resolved = artifacts.resolve_pretrained_path(config['model'], config['pretrained'])
    require(sha(resolved) == plan['weights']['pretrained']['sha256'], 'Runtime resolves a different weight')
    external = plan['weights']['external']['path'] if plan['weights']['external'] else None
    encoder = artifacts.extraction_signature(config['model'], config['pretrained'], external, 'physics', 6, 8)
    model, _, preprocess = runtime.create_backbone(config['model'], config['pretrained'], external)
    model = model.float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'Expected FP32 encoder')
    tokenizer = open_clip.get_tokenizer(config['model'])
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'open_clip_torch', 'numpy', 'Pillow', 'scipy', 'timm', 'transformers', 'tokenizers')}
    return model, preprocess, tokenizer, encoder, packages


def build(plan, ids, captions, out, device):
    from datetime import datetime, timezone
    import io
    import torch
    from PIL import Image
    import artifacts
    import v9_helpers as v9
    from data_contract import training_index, query_sets
    from text_helpers import encode_text, state_sha
    model, preprocess, tokenizer, encoder, packages = setup_encoder(plan, device)
    selected = plan['selected_ids']
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'PLAN.json', plan)
    try:
        cache = empty_cache(selected, plan['dimension'])
        records = []
        for row, name in enumerate(selected):
            source = plan['image_files'][name]
            raw = Path(source['path']).read_bytes()
            import hashlib
            require(hashlib.sha256(raw).hexdigest() == source['sha256'], 'Image changed before decode')
            with Image.open(io.BytesIO(raw)) as opened:
                image = opened.convert('RGB')
            record = fill_image(cache, row, image, model, preprocess, device)
            records.append({'image_id': name, **record})
            if (row + 1) % 100 == 0 or row + 1 == len(selected):
                print(json.dumps({'images_encoded': row + 1, 'images_total': len(selected)}), flush=True)
        signature = {
            'schema': 'pare.portable-raw-cache.v1', 'origin': 'new_raw_image_reproduction',
            'encoder': encoder, 'fine_tie_selection': 'legacy_raster',
            'policy': plan['policy'], 'source_lock_sha256': plan['source_lock_sha256'],
            'preprocess': v9.normalized_transform(repr(preprocess)),
            'population': plan['population'], 'smoke': plan['smoke'],
            'ids_sha256': objsha(selected),
        }
        cache['config'] = {'schema': signature['schema'], 'fine_slots': 8, 'fine_half_pixels': 12}
        cache['provenance'] = {
            'schema_version': signature['schema'], 'signature': signature,
            'signature_sha256': objsha(signature), 'ids_sha256': objsha(selected),
            'image_file_sha256': {k: v['sha256'] for k, v in plan['image_files'].items()},
            'eval_transform': repr(preprocess), 'created_utc': datetime.now(timezone.utc).isoformat(),
        }
        pop = plan['population']
        caption_copy = out / ('Caption_' + pop + '.csv')
        shutil.copyfile(plan['captions']['path'], caption_copy)
        write(out / (pop + '_ids.json'), selected)
        with (out / 'cache_legacy_raster.pt').open('xb') as stream:
            torch.save(cache, stream)
        with torch.no_grad():
            if pop == 'train':
                identity = training_index(selected, captions, validate_official=not plan['smoke'])
                vectors = encode_text(identity['texts'], model, tokenizer, device)
                require(vectors.shape == (len(identity['texts']), plan['dimension']), 'Training text dimensions differ')
                text = {'embeddings': vectors, 'backbone': plan['backbone'],
                        'training_text_sha256': identity['training_text_sha256']}
                with (out / 'training_text.pt').open('xb') as stream:
                    torch.save(text, stream)
                manifest = training_manifest(plan, cache, identity, vectors, out, state_sha)
            else:
                identity = query_sets(selected, captions, validate_official=not plan['smoke'])
                text = {name: encode_text([r['caption'] for r in identity[name]], model, tokenizer, device)
                        for name in ('first', 'extra') if identity[name]}
                if 'extra' not in text:
                    text['extra'] = torch.zeros(0, plan['dimension'])
                require(all(text[name].shape == (len(identity[name]), plan['dimension']) for name in ('first', 'extra')),
                        'Query text dimensions differ')
                with (out / 'query_text.pt').open('xb') as stream:
                    torch.save(text, stream)
                write(out / 'QUERY_IDENTITY.json', identity)
                manifest = {'schema': 'pare.portable-retrieval-inputs.v1',
                            'status': 'SMOKE_ONLY' if plan['smoke'] else 'BOUND',
                            'backbone': plan['backbone'], 'dimension': plan['dimension'],
                            'cache': relative_binding(out / 'cache_legacy_raster.pt'),
                            'query_text': relative_binding(out / 'query_text.pt'),
                            'query_identity': relative_binding(out / 'QUERY_IDENTITY.json'),
                            'encoder_signature': encoder,
                            'origin': 'new_raw_image_reproduction; not the archived final-score input catalog'}
        require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all())
                    for v in text.values() if torch.is_tensor(v)), 'Invalid text embeddings')
        reread = torch.load(out / 'cache_legacy_raster.pt', map_location='cpu', weights_only=True)
        require(state_sha({k: v for k, v in cache.items() if torch.is_tensor(v)}) ==
                state_sha({k: v for k, v in reread.items() if torch.is_tensor(v)}), 'Cache roundtrip changed values')
        text_file = out / ('training_text.pt' if pop == 'train' else 'query_text.pt')
        reread_text = torch.load(text_file, map_location='cpu', weights_only=True)
        require(state_sha({k: v for k, v in text.items() if torch.is_tensor(v)}) ==
                state_sha({k: v for k, v in reread_text.items() if torch.is_tensor(v)}), 'Text roundtrip changed values')
        external = plan['weights']['external']['path'] if plan['weights']['external'] else None
        require(encoder == artifacts.extraction_signature(plan['config']['model'], plan['config']['pretrained'],
                                                          external, 'physics', 6, 8), 'Encoder identity changed')
        recheck_plan(plan)
        require(sha(caption_copy) == plan['captions']['sha256'], 'Caption copy changed')
        write(out / 'IMAGE_RECORDS.json', records)
        receipt = {'schema': 'pare.portable-raw-input-build.v1', 'status': 'completed',
                   'population': pop, 'smoke': plan['smoke'], 'source_lock_sha256': plan['source_lock_sha256'],
                   'packages': packages, 'torch_device': device,
                   'gpu_name': torch.cuda.get_device_name() if device == 'cuda' else None,
                   'input_image_count': len(selected), 'new_reproduction_not_archived_bytes': True,
                   'reference_tensor_alignment_tested': False, 'training_performed': False,
                   'retrieval_scores_computed': False,
                   'artifacts': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
        write(out / 'BUILD_RECEIPT.json', receipt)
        # Only publish BOUND input manifests after all byte/source checks succeed.
        write(out / ('SMOKE_INPUTS.json' if plan['smoke'] else 'INPUTS.json'), manifest)
    except BaseException as error:
        write(out / 'FAILED.json', {'type': type(error).__name__, 'message': str(error)})
        raise


def relative_binding(path):
    return {'path': path.name, 'sha256': sha(path)}


def training_manifest(plan, cache, identity, vectors, out, state_sha):
    signature = cache['provenance']['signature']
    return {
        'schema': 'pare.portable-full-training-inputs.v1',
        'status': 'SMOKE_ONLY' if plan['smoke'] else 'BOUND',
        'backbone': plan['backbone'], 'dimension': plan['dimension'],
        'cache': relative_binding(out / 'cache_legacy_raster.pt'),
        'captions': relative_binding(out / 'Caption_train.csv'),
        'training_text': relative_binding(out / 'training_text.pt'),
        'training_ids_sha256': identity['training_ids_sha256'],
        'pair_order_sha256': identity['pair_order_sha256'],
        'training_text_sha256': identity['training_text_sha256'],
        'training_tensor_sha256': state_sha({'train': vectors}),
        'cache_signature_sha256': objsha(signature), 'encoder_signature_path': ['encoder'],
        'encoder_signature': signature['encoder'],
        'origin': {'kind': 'new_raw_image_reproduction', 'archived_byte_identity_claimed': False,
                   'source_lock_sha256': plan['source_lock_sha256']},
    }
