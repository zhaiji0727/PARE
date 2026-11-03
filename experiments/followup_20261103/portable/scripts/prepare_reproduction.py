#!/usr/bin/env python3
"""Validate local inputs, then create a narrowly rebound reproduction protocol.

This does not train, download weights, access test data, or change the original
executed protocol. Use a fresh output directory for each independent reproduction.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

from protocol_adapter import (ORIGINAL_PROTOCOL_SHA256, check_cache_contract,
                              load_original, make_reproduction_protocol, sha)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'original'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True, help='new compatible physics training cache')
    parser.add_argument('--captions', type=Path, required=True, help='clean Caption_train.csv, exact original bytes')
    parser.add_argument('--split-dir', type=Path, required=True, help='fixed fit/val/train IDs and local manifest.json')
    parser.add_argument('--model-root', type=Path, required=True, help='prepared offline OpenCLIP weights directory')
    parser.add_argument('--out', type=Path, required=True, help='new protocol/preparation output directory')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh reproduction output directory')
    original = load_original(ORIGINAL)
    bindings = original['input_bindings']
    base = ORIGINAL / 'base_code'
    for name, digest in bindings['frozen_source_sha256'].items():
        if sha(base / name) != digest:
            raise ValueError('Original scientific implementation changed: ' + name)
    if sha(ROOT / 'code/factorial_head.py') != sha(ORIGINAL / 'code/factorial_head.py'):
        raise ValueError('Portable factor adapter must have the original executed bytes')
    paths = {'training_cache': args.cache, 'training_captions': args.captions,
             'local_split_manifest': args.split_dir / 'manifest.json',
             **{name: args.split_dir / name for name in bindings['split_file_sha256']}}
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes['training_captions'] != bindings['clean_training_captions_sha256']:
        raise ValueError('Clean training captions differ from the original fixed input')
    for name, digest in bindings['split_file_sha256'].items():
        if hashes[name] != digest:
            raise ValueError('Fixed split IDs changed: ' + name)
    if not isinstance(json.loads(paths['local_split_manifest'].read_text()), dict):
        raise ValueError('Local split metadata must be a JSON object')
    train = json.loads((args.split_dir / 'train_ids.json').read_text())
    fit = json.loads((args.split_dir / 'fit_ids.json').read_text())
    val = json.loads((args.split_dir / 'val_ids.json').read_text())
    if len(fit) != 8263 or len(set(fit)) != 8263 or len(val) != 918 or len(set(val)) != 918:
        raise ValueError('Expected 8263 distinct fit IDs and 918 distinct validation IDs')
    if set(fit) & set(val) or set(fit) | set(val) != set(train):
        raise ValueError('Fit/validation must disjointly partition the fixed training IDs')
    os.environ['RS_RAG_MODEL_ROOT'] = str(args.model_root.resolve())
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(base))
    import torch
    from artifacts import validate_cache, resolve_pretrained_path
    from train_saqre import load_captions
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    signature = validate_cache(cache, original['backbone']['model'], original['backbone']['pretrained'])
    check_cache_contract(cache, signature, original, train)
    weights = resolve_pretrained_path(original['backbone']['model'], original['backbone']['pretrained'])
    paths['pretrained_weight'] = weights
    hashes['pretrained_weight'] = sha(weights)
    if hashes['pretrained_weight'] != original['backbone']['pretrained_weight_sha256']:
        raise ValueError('Backbone weights differ from the original fixed input')
    caps = load_captions(args.captions)
    fit_set, val_set = set(fit), set(val)
    if sum(len(captions) for name, captions in caps.items() if name in fit_set) != 28759:
        raise ValueError('Expected all 28759 fit caption pairs')
    if {name for name in caps if name in fit_set} != fit_set or {name for name in caps if name in val_set} != val_set:
        raise ValueError('Every fit and validation image must have a caption')
    if any(sha(path) != hashes[name] for name, path in paths.items()):
        raise RuntimeError('An input file changed during preparation')
    created = datetime.now(timezone.utc).isoformat()
    plan = make_reproduction_protocol(original, hashes['training_cache'], hashes['local_split_manifest'], created)
    args.out.mkdir(parents=True)
    protocol = args.out / 'protocol.json'
    protocol.write_text(json.dumps(plan, indent=2) + '\n')
    shutil.copyfile(ORIGINAL / 'plans/protocol.md', args.out / 'protocol.md')
    protocol_sha = sha(protocol)
    commands = []
    for seed in original['seeds']:
        command = [sys.executable, str(ROOT / 'scripts/run_factorial_fitval.py'),
                   '--protocol', str(protocol.resolve()), '--expected-protocol-sha256', protocol_sha,
                   '--base-code', str(base), '--cache', str(args.cache.resolve()),
                   '--captions', str(args.captions.resolve()), '--split-dir', str(args.split_dir.resolve()),
                   '--out', str((args.out / 'results' / f'seed_{seed}').resolve()), '--seed', str(seed)]
        commands.append(command)
    manifest = {'schema': 'pare.portable-preparation.v1', 'created_utc': created,
                'parent_protocol_sha256': ORIGINAL_PROTOCOL_SHA256, 'protocol_sha256': protocol_sha,
                'input_sha256': hashes, 'cache_signature': signature,
                'counts': {'cache_images': 9181, 'fit_images': 8263, 'fit_caption_pairs': 28759,
                           'validation_images': 918, 'configurations': 14, 'seeds': 3},
                'portable_source_sha256': {str(path.relative_to(ROOT)): sha(path)
                                          for path in [ROOT / 'scripts/protocol_adapter.py',
                                                       ROOT / 'scripts/prepare_reproduction.py',
                                                       ROOT / 'scripts/run_factorial_fitval.py',
                                                       ROOT / 'code/factorial_head.py']},
                'scientific_design_unchanged': True, 'training_started': False,
                'run_commands': commands}
    (args.out / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Validated local inputs; no training started.')
    print('Protocol SHA-256:', protocol_sha)
    print('Set RS_RAG_MODEL_ROOT to', shlex.quote(str(args.model_root.resolve())))
    print('Choose an available physical CUDA GPU using CUDA_VISIBLE_DEVICES and PARE_DEVICE=cuda:0.')
    print('Preflight each command first by adding --preflight-only and a separate --out directory.')
    for command in commands:
        print(shlex.join(command))


if __name__ == '__main__':
    main()
