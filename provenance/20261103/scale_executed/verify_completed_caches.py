#!/usr/bin/env python3
"""Read back completed scale artifacts and verify their recorded file hashes."""
import argparse
import importlib.util
import json
from pathlib import Path

from scale_contract import file_hash
from stress_cache import load_frozen_code, validate_scale_cache, verify_frozen_imports
from evaluate_scale import validate_extra_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    load_frozen_code(args.code_root)
    import torch
    spec = importlib.util.spec_from_file_location('verification_evaluator', args.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    if file_hash(args.reference) != args.reference_sha256:
        raise ValueError('Reference differs from independently verified original cache')
    reference = torch.load(args.reference, map_location='cpu', weights_only=True)
    summary = {'reference_cache_sha256': args.reference_sha256, 'n_gallery': len(reference['ids']),
               'is_real_gsd_validation': False, 'conditions': {}}
    if len(reference['ids']) != 3688:
        raise ValueError('Not the full formal gallery')
    for factor in ('1.5', '2'):
        path = args.directory / f'cache_spacing_{factor}.pt'
        manifest = json.loads(path.with_suffix('.pt.manifest.json').read_text())
        digest = file_hash(path)
        if digest != manifest['cache_sha256']:
            raise ValueError('Serialized scale artifact differs from completed manifest')
        cache = torch.load(path, map_location='cpu', weights_only=True)
        validate_scale_cache(cache, reference, args.reference_sha256, factor,
                             'ViT-B-32', 'laion2b_s34b_b79k')
        verify_frozen_imports(args.code_root)
        evaluator.validate_tensor_layout(cache)
        validate_extra_tensors(cache)
        if cache['scale_test']['image_records'] != manifest['image_records']:
            raise ValueError('Manifest records differ from serialized records')
        rows = list(manifest['image_records'].values())
        ratios = [value for row in rows for value in row['effective_spacing_ratio_xy']]
        sizes = [value for row in rows for value in row['transformed_size_wh']]
        summary['conditions'][factor] = {
            'cache_sha256': digest, 'bytes': path.stat().st_size,
            'signature_sha256': cache['provenance']['signature_sha256'],
            'n_image_records': len(rows), 'actual_axis_spacing_ratio_range': [min(ratios), max(ratios)],
            'transformed_image_side_range_pixels': [min(sizes), max(sizes)],
            'extraction_elapsed_seconds': manifest['elapsed_seconds'],
            'all_contract_and_tensor_checks_passed': True}
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
